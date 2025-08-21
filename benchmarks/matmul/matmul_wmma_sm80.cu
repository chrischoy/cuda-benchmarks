#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <type_traits>
using namespace nvcuda;

// -------------------- Type traits & converters --------------------

template <typename T>
struct IsInputSupported : std::false_type
{
};
template <>
struct IsInputSupported<half> : std::true_type
{
};

template <typename T>
struct IsOutputSupported : std::false_type
{
};
template <>
struct IsOutputSupported<float> : std::true_type
{
};
template <>
struct IsOutputSupported<half> : std::true_type
{
};
#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct IsOutputSupported<nv_bfloat16> : std::true_type
{
};
#endif

// WMMA A/B type tags for half / bf16
template <typename T>
struct WmmaInputTag;
template <>
struct WmmaInputTag<half>
{
    using type = half;
};
// Enable this only if your CUDA WMMA headers provide bfloat16 support
#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(WMMA_BF16_AVAILABLE)
template <>
struct WmmaInputTag<nv_bfloat16>
{
    using type = wmma::precision::bfloat16;
};
#endif

// to-float for input/bias
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }

// from-float to output
__device__ __forceinline__ float from_float(float x, float *) { return x; }
__device__ __forceinline__ half from_float(float x, half *) { return __float2half_rn(x); }
#if defined(__CUDA_BF16_TYPES_EXIST__)
__device__ __forceinline__ nv_bfloat16 from_float(float x, nv_bfloat16 *) { return __float2bfloat16(x); }
#endif

// -------------------- Ampere cp.async helpers (16B) --------------------

static __device__ __forceinline__ void cp_async_16B(void *smem_ptr, const void *gmem_ptr, bool pred)
{
#if __CUDA_ARCH__ >= 800
    if (pred)
    {
        unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::
                         "r"(smem_addr),
                     "l"(gmem_ptr), "n"(16));
    }
    else
    {
        uint4 z = {0, 0, 0, 0};
        *reinterpret_cast<uint4 *>(smem_ptr) = z;
    }
#else
    if (pred)
    {
        uint4 v = *reinterpret_cast<const uint4 *>(gmem_ptr);
        *reinterpret_cast<uint4 *>(smem_ptr) = v;
    }
    else
    {
        uint4 z = {0, 0, 0, 0};
        *reinterpret_cast<uint4 *>(smem_ptr) = z;
    }
#endif
}
static __device__ __forceinline__ void cp_async_commit()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}
static __device__ __forceinline__ void cp_async_wait_all()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n");
#endif
}

// -------------------- Kernel (templated InT, OutT, UseAtomic) --------------------
// One warp per block (blockDim.x == 32).
template <typename InT, typename OutT, bool UseAtomic = true>
__global__ void implicit_gemm_wmma_db_ampere(
    const InT *__restrict__ A,       // [M,K], row-major
    const InT *__restrict__ B,       // [K,N], row-major
    const InT *__restrict__ Cbias,   // [M,N] optional (may be nullptr)
    const long *__restrict__ a_inds, // [P]
    const long *__restrict__ d_inds, // [P]
    OutT *__restrict__ D,            // [Q,N], row-major
    int M, int K, int N, int P, int Q)
{
    static_assert(IsInputSupported<InT>::value, "InT must be half or nv_bfloat16");
    static_assert(IsOutputSupported<OutT>::value, "OutT must be float, half, or nv_bfloat16");

    // Require one warp per block
    if (blockDim.x != 32)
        return;

    constexpr int TILE_M = 16, TILE_N = 16, TILE_K = 16;

    const int lane = threadIdx.x & 31;
    const int tile_n = blockIdx.x; // 16-wide tile index along N
    const int n_base = tile_n * TILE_N;

    // stride P-tiles along gridDim.y like original
    for (int tile_p = blockIdx.y; tile_p < (P + TILE_M - 1) / TILE_M; tile_p += gridDim.y)
    {

        // Gather/scatter row indices
        __shared__ long a_row_idx[TILE_M];
        __shared__ long d_row_idx[TILE_M];
        if (lane < TILE_M)
        {
            const int p = tile_p * TILE_M + lane;
            a_row_idx[lane] = (p < P) ? a_inds[p] : -1;
            d_row_idx[lane] = (p < P) ? d_inds[p] : -1;
        }
        __syncthreads();

        // Double-buffered smem tiles (each 16x16 of InT)
        __shared__ __align__(16) InT Asmem[2][TILE_M * TILE_K];
        __shared__ __align__(16) InT Bsmem[2][TILE_K * TILE_N];
        __shared__ float OutTile[TILE_M * TILE_N]; // accumulation dump buffer

        // WMMA fragments (f32 accumulate)
        using WmmaT = typename WmmaInputTag<InT>::type;
        wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, WmmaT, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, WmmaT, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        // Async stage loader for one K-slice
        auto stage_load_tiles = [&](int k0, int stage)
        {
            const int row = lane & 15; // 0..15
            const int seg = lane >> 4; // 0..1 (8 elements => 16B per seg)

            // A (gather)
            long arow = a_row_idx[row];
            const InT *gA = (arow >= 0 && arow < M) ? (A + (size_t)arow * K + k0 + seg * 8) : nullptr;
            bool predA = (arow >= 0 && arow < M) && (k0 + seg * 8 + 7 < K);
            InT *sA = &Asmem[stage][row * TILE_K + seg * 8];
            cp_async_16B((void *)sA, (const void *)gA, predA);

            // B (contiguous)
            const int kb = k0 + row;
            const int nb = n_base + seg * 8;
            const InT *gB = (kb < K && nb < N) ? (B + (size_t)kb * N + nb) : nullptr;
            bool predB = (kb < K) && (nb + 7 < N);
            InT *sB = &Bsmem[stage][row * TILE_N + seg * 8];
            cp_async_16B((void *)sB, (const void *)gB, predB);
        };

        // Preload stage 0
        int stage = 0;
        stage_load_tiles(/*k0=*/0, /*stage=*/stage);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // K-loop with ping-pong
        for (int k0 = 0; k0 < K; k0 += TILE_K)
        {
            const int next_stage = stage ^ 1;

            if (k0 + TILE_K < K)
            {
                stage_load_tiles(k0 + TILE_K, next_stage);
                cp_async_commit();
            }

            // Compute
            wmma::load_matrix_sync(a_frag, &Asmem[stage][0], TILE_K);
            wmma::load_matrix_sync(b_frag, &Bsmem[stage][0], TILE_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

            if (k0 + TILE_K < K)
                cp_async_wait_all();
            __syncthreads();
            stage = next_stage;
        }

        // Dump accumulator to smem (row-major)
        wmma::store_matrix_sync(OutTile, acc_frag, TILE_N, wmma::mem_row_major);
        __syncthreads();

        // ---------------- Epilogue (bias add + convert + store/add) ----------------
        // Strategy:
        //  - OutT == float  : 2 × float4 (8 floats) per lane → 64 chunks/tile
        //  - OutT != float  : 1 × 8-wide (8 values) per lane → 32 chunks/tile
        if constexpr (std::is_same<OutT, float>::value)
        {
#pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                int chunk_id = lane + i * 32;     // 0..63
                int r = (chunk_id * 4) / TILE_N;  // row
                int c4 = (chunk_id * 4) % TILE_N; // col start (multiple of 4)
                int n = n_base + c4;
                int p = tile_p * TILE_M + r;

                long arow = a_row_idx[r];
                long drow = d_row_idx[r];
                if (!(p < P && arow >= 0 && arow < M && drow >= 0 && drow < Q))
                    continue;

                const float f0 = OutTile[r * TILE_N + c4 + 0];
                const float f1 = OutTile[r * TILE_N + c4 + 1];
                const float f2 = OutTile[r * TILE_N + c4 + 2];
                const float f3 = OutTile[r * TILE_N + c4 + 3];

                if constexpr (UseAtomic)
                {
                    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
                    if (Cbias)
                    {
                        if (n + 3 < N)
                        {
                            b0 = to_float(Cbias[(size_t)arow * N + (n + 0)]);
                            b1 = to_float(Cbias[(size_t)arow * N + (n + 1)]);
                            b2 = to_float(Cbias[(size_t)arow * N + (n + 2)]);
                            b3 = to_float(Cbias[(size_t)arow * N + (n + 3)]);
                        }
                        else
                        {
                            if (n + 0 < N)
                                b0 = to_float(Cbias[(size_t)arow * N + (n + 0)]);
                            if (n + 1 < N)
                                b1 = to_float(Cbias[(size_t)arow * N + (n + 1)]);
                            if (n + 2 < N)
                                b2 = to_float(Cbias[(size_t)arow * N + (n + 2)]);
                            if (n + 3 < N)
                                b3 = to_float(Cbias[(size_t)arow * N + (n + 3)]);
                        }
                    }
                    float *__restrict__ Drow = &D[(size_t)drow * N + n];
                    if (n + 3 < N)
                    {
                        atomicAdd(&Drow[0], f0 + b0);
                        atomicAdd(&Drow[1], f1 + b1);
                        atomicAdd(&Drow[2], f2 + b2);
                        atomicAdd(&Drow[3], f3 + b3);
                    }
                    else
                    {
                        if (n + 0 < N)
                            atomicAdd(&Drow[0], f0 + b0);
                        if (n + 1 < N)
                            atomicAdd(&Drow[1], f1 + b1);
                        if (n + 2 < N)
                            atomicAdd(&Drow[2], f2 + b2);
                        if (n + 3 < N)
                            atomicAdd(&Drow[3], f3 + b3);
                    }
                }
                else
                {
                    // Vectorized 16B store when fully in-bounds; tail-safe fallback otherwise
                    float v0 = f0, v1 = f1, v2 = f2, v3 = f3;
                    if (Cbias)
                    {
                        if (n + 3 < N)
                        {
                            v0 += to_float(Cbias[(size_t)arow * N + (n + 0)]);
                            v1 += to_float(Cbias[(size_t)arow * N + (n + 1)]);
                            v2 += to_float(Cbias[(size_t)arow * N + (n + 2)]);
                            v3 += to_float(Cbias[(size_t)arow * N + (n + 3)]);
                        }
                        else
                        {
                            if (n + 0 < N)
                                v0 += to_float(Cbias[(size_t)arow * N + (n + 0)]);
                            if (n + 1 < N)
                                v1 += to_float(Cbias[(size_t)arow * N + (n + 1)]);
                            if (n + 2 < N)
                                v2 += to_float(Cbias[(size_t)arow * N + (n + 2)]);
                            if (n + 3 < N)
                                v3 += to_float(Cbias[(size_t)arow * N + (n + 3)]);
                        }
                    }
                    float *__restrict__ Dst = &D[(size_t)drow * N + n];
                    const bool aligned16 = ((reinterpret_cast<uintptr_t>(Dst) & 0xF) == 0);
                    if ((n + 3 < N) && aligned16)
                    {
                        float4 v = make_float4(v0, v1, v2, v3);
                        *reinterpret_cast<float4 *>(Dst) = v;
                    }
                    else
                    {
                        if (n + 0 < N)
                            D[(size_t)drow * N + (n + 0)] = v0;
                        if (n + 1 < N)
                            D[(size_t)drow * N + (n + 1)] = v1;
                        if (n + 2 < N)
                            D[(size_t)drow * N + (n + 2)] = v2;
                        if (n + 3 < N)
                            D[(size_t)drow * N + (n + 3)] = v3;
                    }
                }
            }
        }
        else
        {
            // half / bf16: one 8-wide chunk per lane (16B total)
            int chunk_id = lane;              // 0..31
            int r = (chunk_id * 8) / TILE_N;  // row
            int c8 = (chunk_id * 8) % TILE_N; // col start (0 or 8)
            int n = n_base + c8;
            int p = tile_p * TILE_M + r;

            long arow = a_row_idx[r];
            long drow = d_row_idx[r];

            if (p < P && arow >= 0 && arow < M && drow >= 0 && drow < Q)
            {
                if (n + 7 < N)
                {
                    if constexpr (UseAtomic)
                    {
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                        {
                            float val = OutTile[r * TILE_N + c8 + j];
                            if (Cbias)
                                val += to_float(Cbias[(size_t)arow * N + (n + j)]);
                            atomicAdd(&D[(size_t)drow * N + (n + j)], from_float(val, (OutT *)nullptr));
                        }
                    }
                    else
                    {
                        // Convert 8 floats -> OutT packed 16B and store as a single vector (if aligned)
                        alignas(16) OutT tmp8[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                        {
                            float val = OutTile[r * TILE_N + c8 + j];
                            if (Cbias)
                                val += to_float(Cbias[(size_t)arow * N + (n + j)]);
                            tmp8[j] = from_float(val, (OutT *)nullptr);
                        }
                        OutT *__restrict__ Dst = &D[(size_t)drow * N + n];
                        const bool aligned16 = ((reinterpret_cast<uintptr_t>(Dst) & 0xF) == 0);
                        if (aligned16)
                        {
                            *reinterpret_cast<uint4 *>(Dst) = *reinterpret_cast<const uint4 *>(tmp8);
                        }
                        else
                        {
                            // Fallback to scalar stores if not aligned
#pragma unroll
                            for (int j = 0; j < 8; ++j)
                            {
                                Dst[j] = tmp8[j];
                            }
                        }
                    }
                }
                else
                {
                    // tail-safe (N not multiple of 8)
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        int nc = n + j;
                        if (nc >= N)
                            break;
                        float val = OutTile[r * TILE_N + c8 + j];
                        if (Cbias)
                            val += to_float(Cbias[(size_t)arow * N + nc]);
                        if constexpr (UseAtomic)
                        {
                            atomicAdd(&D[(size_t)drow * N + nc], from_float(val, (OutT *)nullptr));
                        }
                        else
                        {
                            D[(size_t)drow * N + nc] = from_float(val, (OutT *)nullptr);
                        }
                    }
                }
            }
        }
    }
}

// -------------------- Launch helpers --------------------

template <typename InT, typename OutT, bool UseAtomic = true>
void launch_implicit_gemm_ampere(
    dim3 grid, dim3 block, cudaStream_t stream,
    const InT *A, const InT *B, const InT *Cbias,
    const long *a_inds, const long *d_inds,
    OutT *D, int M, int K, int N, int P, int Q)
{
    implicit_gemm_wmma_db_ampere<InT, OutT, UseAtomic>
        <<<grid, block, 0, stream>>>(A, B, Cbias, a_inds, d_inds, D, M, K, N, P, Q);
}

// -------------------- Explicit launch wrappers (for linking from matmul.cu) --------------------
extern "C"
{
    void launch_wmma_db_ampere_f16_f32(
        dim3 grid, dim3 block, cudaStream_t stream,
        const half *A, const half *B, const half *Cbias,
        const long *a_inds, const long *d_inds,
        float *D, int M, int K, int N, int P, int Q)
    {
        launch_implicit_gemm_ampere<half, float, true>(grid, block, stream, A, B, Cbias, a_inds, d_inds, D, M, K, N, P, Q);
    }

    void launch_wmma_db_ampere_f16_f16(
        dim3 grid, dim3 block, cudaStream_t stream,
        const half *A, const half *B, const half *Cbias,
        const long *a_inds, const long *d_inds,
        half *D, int M, int K, int N, int P, int Q)
    {
        launch_implicit_gemm_ampere<half, half, true>(grid, block, stream, A, B, Cbias, a_inds, d_inds, D, M, K, N, P, Q);
    }

#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(WMMA_BF16_AVAILABLE)
    void launch_wmma_db_ampere_bf16_f32(
        dim3 grid, dim3 block, cudaStream_t stream,
        const nv_bfloat16 *A, const nv_bfloat16 *B, const nv_bfloat16 *Cbias,
        const long *a_inds, const long *d_inds,
        float *D, int M, int K, int N, int P, int Q)
    {
        launch_implicit_gemm_ampere<nv_bfloat16, float, true>(grid, block, stream, A, B, Cbias, a_inds, d_inds, D, M, K, N, P, Q);
    }

    void launch_wmma_db_ampere_bf16_bf16(
        dim3 grid, dim3 block, cudaStream_t stream,
        const nv_bfloat16 *A, const nv_bfloat16 *B, const nv_bfloat16 *Cbias,
        const long *a_inds, const long *d_inds,
        nv_bfloat16 *D, int M, int K, int N, int P, int Q)
    {
        launch_implicit_gemm_ampere<nv_bfloat16, nv_bfloat16, true>(grid, block, stream, A, B, Cbias, a_inds, d_inds, D, M, K, N, P, Q);
    }
#endif
}

// Optional non-atomic wrappers (write-only epilogue)
extern "C"
{
    void launch_wmma_db_ampere_f16_f32_store(
        dim3 grid, dim3 block, cudaStream_t stream,
        const half *A, const half *B, const half *Cbias,
        const long *a_inds, const long *d_inds,
        float *D, int M, int K, int N, int P, int Q)
    {
        launch_implicit_gemm_ampere<half, float, false>(grid, block, stream, A, B, Cbias, a_inds, d_inds, D, M, K, N, P, Q);
    }

    void launch_wmma_db_ampere_f16_f16_store(
        dim3 grid, dim3 block, cudaStream_t stream,
        const half *A, const half *B, const half *Cbias,
        const long *a_inds, const long *d_inds,
        half *D, int M, int K, int N, int P, int Q)
    {
        launch_implicit_gemm_ampere<half, half, false>(grid, block, stream, A, B, Cbias, a_inds, d_inds, D, M, K, N, P, Q);
    }

#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(WMMA_BF16_AVAILABLE)
    void launch_wmma_db_ampere_bf16_f32_store(
        dim3 grid, dim3 block, cudaStream_t stream,
        const nv_bfloat16 *A, const nv_bfloat16 *B, const nv_bfloat16 *Cbias,
        const long *a_inds, const long *d_inds,
        float *D, int M, int K, int N, int P, int Q)
    {
        launch_implicit_gemm_ampere<nv_bfloat16, float, false>(grid, block, stream, A, B, Cbias, a_inds, d_inds, D, M, K, N, P, Q);
    }

    void launch_wmma_db_ampere_bf16_bf16_store(
        dim3 grid, dim3 block, cudaStream_t stream,
        const nv_bfloat16 *A, const nv_bfloat16 *B, const nv_bfloat16 *Cbias,
        const long *a_inds, const long *d_inds,
        nv_bfloat16 *D, int M, int K, int N, int P, int Q)
    {
        launch_implicit_gemm_ampere<nv_bfloat16, nv_bfloat16, false>(grid, block, stream, A, B, Cbias, a_inds, d_inds, D, M, K, N, P, Q);
    }
#endif
}

// -------------------- Example usage --------------------
// dim3 block(32); // one warp
// dim3 grid((N+15)/16, (P+15)/16);
// // FP16 in -> FP32 out
// launch_implicit_gemm_ampere<half, float>(grid, block, stream, A_h, B_h, C_h, a_inds, d_inds, D_f, M,K,N,P,Q);
// // BF16 in -> BF16 out
// launch_implicit_gemm_ampere<nv_bfloat16, nv_bfloat16>(grid, block, stream, A_bf, B_bf, C_bf, a_inds, d_inds, D_bf, M,K,N,P,Q);
// // FP16 in -> FP16 out
// launch_implicit_gemm_ampere<half, half>(grid, block, stream, A_h, B_h, C_h, a_inds, d_inds, D_h, M,K,N,P,Q);

// Compile with:  nvcc -arch=sm_80 -O3 ...
