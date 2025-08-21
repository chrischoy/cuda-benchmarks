#include <cstdint>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <vector>

#include "../../include/vector_types.cuh"
#include "matmul_common.cuh"

using namespace cuda_benchmark::matmul;
using namespace nvcuda;
#if 1
// Forward declarations to link to wrappers in matmul_wmma_sm80.cu
extern "C" {
void launch_wmma_db_ampere_f16_f32(dim3, dim3, cudaStream_t, const half *,
                                   const half *, const half *, const long *,
                                   const long *, float *, int, int, int, int,
                                   int);
void launch_wmma_db_ampere_f16_f16(dim3, dim3, cudaStream_t, const half *,
                                   const half *, const half *, const long *,
                                   const long *, half *, int, int, int, int,
                                   int);
// Non-atomic store variants
void launch_wmma_db_ampere_f16_f32_store(dim3, dim3, cudaStream_t, const half *,
                                         const half *, const half *,
                                         const long *, const long *, float *,
                                         int, int, int, int, int);
void launch_wmma_db_ampere_f16_f16_store(dim3, dim3, cudaStream_t, const half *,
                                         const half *, const half *,
                                         const long *, const long *, half *,
                                         int, int, int, int, int);
}
#endif

// Compute D[d_inds, :] += A[a_inds, :] @ B + C[a_inds, :]

// Naive FP32 baseline: one block per output row p, threads iterate N
__global__ void implicit_gemm_naive_f32(const float *__restrict__ A, // [M, K]
                                        const float *__restrict__ B, // [K, N]
                                        const float *__restrict__ C, // [M, N]
                                        const long *__restrict__ a_inds, // [P]
                                        const long *__restrict__ d_inds, // [P]
                                        float *__restrict__ D, // [Q, N]
                                        int M, int K, int N, int P, int Q) {
  for (int p = blockIdx.y; p < P; p += gridDim.y) {
    int n0 = blockIdx.x * blockDim.x + threadIdx.x;
    long a = a_inds[p];
    long d = d_inds[p];
    if (a < 0 || a >= M || d < 0 || d >= Q)
      continue;

    for (int n = n0; n < N; n += blockDim.x * gridDim.x) {
      float acc = 0.0f;
      const float *a_row = A + a * K;
      const float *b_col = B + n; // column-major access simulated via stride N
      for (int k = 0; k < K; ++k) {
        acc += a_row[k] * b_col[k * N];
      }
      float c_val = C ? C[a * N + n] : 0.0f;
      atomicAdd(&D[d * N + n], acc + c_val);
    }
  }
}

// WMMA FP16 inputs, FP32 accumulate, 16x16x16 tiles; gather/scatter per tile
__global__ void implicit_gemm_wmma_f16_acc_f32(
    const half *__restrict__ A,      // [M, K]
    const half *__restrict__ B,      // [K, N]
    const half *__restrict__ C,      // [M, N] optional, can be null
    const long *__restrict__ a_inds, // [P]
    const long *__restrict__ d_inds, // [P]
    float *__restrict__ D,           // [Q, N]
    int M, int K, int N, int P, int Q) {
  // Tile indices
  int tile_n = blockIdx.x; // along N
  for (int tile_p = blockIdx.y; tile_p < (P + 15) / 16; tile_p += gridDim.y) {
    int warp_lane = threadIdx.x % 32;

    // Load up to 16 a/d indices participating in this tile row
    __shared__ long a_row_idx[16];
    __shared__ long d_row_idx[16];
    int row_in_tile = warp_lane;
    if (row_in_tile < 16) {
      int p = tile_p * 16 + row_in_tile;
      long aval = (p < P) ? a_inds[p] : -1;
      long dval = (p < P) ? d_inds[p] : -1;
      a_row_idx[row_in_tile] = aval;
      d_row_idx[row_in_tile] = dval;
    }
    __syncthreads();

    // Shared tiles and WMMA fragments
    __shared__ half A_tile[16 * 16];
    __shared__ half B_tile[16 * 16];
    __shared__ float out_tile[16 * 16];

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int n_base = tile_n * 16;
    // Iterate over K in 16 chunks
    for (int k0 = 0; k0 < K; k0 += 16) {
      // Each lane cooperatively loads 256 elements in 8 iterations
      for (int t = threadIdx.x % 32; t < 256; t += 32) {
        int r = t / 16;
        int c = t % 16;
        long arow = a_row_idx[r];
        half av = __float2half(0.0f);
        if (arow >= 0 && arow < M && (k0 + c) < K) {
          av = A[arow * K + (k0 + c)];
        }
        A_tile[r * 16 + c] = av;

        // For B_tile column-major layout expected by WMMA, write at (c*ld + r)
        half bv = __float2half(0.0f);
        int kb = k0 + r;
        int nb = n_base + c;
        if (kb < K && nb < N) {
          bv = B[kb * N + nb];
        }
        B_tile[c * 16 + r] = bv;
      }
      __syncthreads();

      // Load fragments
      wmma::load_matrix_sync(a_frag, A_tile, 16);
      wmma::load_matrix_sync(b_frag, B_tile, 16);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      __syncthreads();
    }

    // Epilogue: add optional bias C[a_inds,:] and scatter-add to D[d_inds,:]
    // Store accumulator to smem then atomically add
    wmma::store_matrix_sync(out_tile, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    // Each lane writes 8 elements via atomicAdd
    for (int t = threadIdx.x % 32; t < 256; t += 32) {
      int r = t / 16;
      int c = t % 16;
      int n = n_base + c;
      int p = tile_p * 16 + r;
      long arow = a_row_idx[r];
      long drow = d_row_idx[r];
      if (p < P && arow >= 0 && arow < M && drow >= 0 && drow < Q && n < N) {
        float val = out_tile[r * 16 + c];
        if (C) {
          val += __half2float(C[arow * N + n]);
        }
        atomicAdd(&D[drow * N + n], val);
      }
    }
  }
}

#if 1
// ---- Ampere cp.async helpers (16B at a time) --------------------------------
static __device__ __forceinline__ void
cp_async_16B(void *smem_ptr, const void *gmem_ptr, bool pred) {
#if __CUDA_ARCH__ >= 800
  if (pred) {
    unsigned smem_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_addr),
                 "l"(gmem_ptr), "n"(16));
  } else {
    uint4 z = {0, 0, 0, 0};
    *reinterpret_cast<uint4 *>(smem_ptr) = z;
  }
#else
  if (pred) {
    uint4 v = *reinterpret_cast<const uint4 *>(gmem_ptr);
    *reinterpret_cast<uint4 *>(smem_ptr) = v;
  } else {
    uint4 z = {0, 0, 0, 0};
    *reinterpret_cast<uint4 *>(smem_ptr) = z;
  }
#endif
}

static __device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}
static __device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n");
#endif
}

// ---- Kernel: WMMA f16×f16→f32, 16x16 tiles, double-buffered over K ----------
// Assumes blockDim.x == 32 (single warp per tile).
__global__ void implicit_gemm_wmma_f16_acc_f32_db_ampere(
    const half *__restrict__ A,      // [M, K], row-major
    const half *__restrict__ B,      // [K, N], row-major
    const half *__restrict__ Cbias,  // [M, N] optional (may be nullptr)
    const long *__restrict__ a_inds, // [P] (rows to gather from A/C)
    const long *__restrict__ d_inds, // [P] (rows to scatter-add into D)
    float *__restrict__ D,           // [Q, N]
    int M, int K, int N, int P, int Q) {
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x;
  const int n_base = tile_n * 16;
  for (int tile_p = blockIdx.y; tile_p < (P + 15) / 16; tile_p += gridDim.y) {

    __shared__ long a_row_idx[16];
    __shared__ long d_row_idx[16];
    if (lane < 16) {
      const int p = tile_p * 16 + lane;
      a_row_idx[lane] = (p < P) ? a_inds[p] : -1;
      d_row_idx[lane] = (p < P) ? d_inds[p] : -1;
    }
    __syncthreads();

    __shared__ __align__(16) half Asmem[2][16 * 16];
    __shared__ __align__(16) half Bsmem[2][16 * 16];
    __shared__ float OutTile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Helper to stage tiles for a given k0 into buffer `stage`
    auto stage_load_tiles = [&](int k0, int stage_param) {
      const int row = lane & 15;
      const int seg = lane >> 4;

      long arow = a_row_idx[row];
      const half *gA = (arow >= 0 && arow < M)
                           ? (A + arow * (size_t)K + k0 + seg * 8)
                           : nullptr;
      bool predA = (arow >= 0 && arow < M) && (k0 + seg * 8 + 7 < K);
      half *sA = &Asmem[stage_param][row * 16 + seg * 8];
      cp_async_16B((void *)sA, (const void *)gA, predA);

      const int kb = k0 + row;
      const int nb = n_base + seg * 8;
      const half *gB = (kb < K && nb < N) ? (B + (size_t)kb * N + nb) : nullptr;
      bool predB = (kb < K) && (nb + 7 < N);
      half *sB = &Bsmem[stage_param][row * 16 + seg * 8];
      cp_async_16B((void *)sB, (const void *)gB, predB);
    };

    int stage = 0;
    stage_load_tiles(/*k0=*/0, /*stage=*/stage);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += 16) {
      const int next_stage = stage ^ 1;
      if (k0 + 16 < K) {
        stage_load_tiles(k0 + 16, next_stage);
        cp_async_commit();
      }

      wmma::load_matrix_sync(a_frag, &Asmem[stage][0], 16);
      wmma::load_matrix_sync(b_frag, &Bsmem[stage][0], 16);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      if (k0 + 16 < K) {
        cp_async_wait_all();
      }
      __syncthreads();
      stage = next_stage;
    }

    wmma::store_matrix_sync(OutTile, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();

    for (int t = lane; t < 256; t += 32) {
      int r = t / 16;
      int c = t % 16;
      int n = n_base + c;
      int p = tile_p * 16 + r;
      long arow = a_row_idx[r];
      long drow = d_row_idx[r];

      if (p < P && arow >= 0 && arow < M && drow >= 0 && drow < Q && n < N) {
        float val = OutTile[r * 16 + c];
        if (Cbias) {
          val += __half2float(Cbias[(size_t)arow * N + n]);
        }
        atomicAdd(&D[(size_t)drow * N + n], val);
      }
    }
  }
}
#endif

#if 0
// BF16 WMMA kernel intentionally disabled
__global__ void implicit_gemm_wmma_bf16_acc_f32(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C,
    const long* __restrict__ a_inds,
    const long* __restrict__ d_inds,
    float* __restrict__ D,
    int M, int K, int N, int P, int Q) {}
#endif

// Inline-PTX assisted FP32 kernel: each thread computes 4 contiguous columns
// using ld.global.v4.f32 for B and C
__global__ void
implicit_gemm_f32_ptx_v4(const float *__restrict__ A,     // [M, K]
                         const float *__restrict__ B,     // [K, N]
                         const float *__restrict__ C,     // [M, N] optional
                         const long *__restrict__ a_inds, // [P]
                         const long *__restrict__ d_inds, // [P]
                         float *__restrict__ D,           // [Q, N]
                         int M, int K, int N, int P, int Q) {
  int thread_group_cols = 4;
  int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * thread_group_cols;
  if (n0 >= N)
    return;

  for (int p = blockIdx.y; p < P; p += gridDim.y) {
    long a = a_inds[p];
    long d = d_inds[p];
    if (a < 0 || a >= M || d < 0 || d >= Q)
      continue;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const float *a_row = A + a * K;

    // Iterate K and accumulate 4 outputs (columns n0..n0+3)
    for (int k = 0; k < K; ++k) {
      float a_val = a_row[k];
      const float *b_ptr = B + k * N + n0;
      float b0, b1, b2, b3;
      bool in_bounds4 = (n0 + 3) < N;
      if (in_bounds4 && ((reinterpret_cast<uintptr_t>(b_ptr) & 0xF) == 0)) {
        asm volatile("{\n\t"
                     "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n\t"
                     "}\n"
                     : "=f"(b0), "=f"(b1), "=f"(b2), "=f"(b3)
                     : "l"(b_ptr));
      } else {
        b0 = (n0 + 0) < N ? b_ptr[0] : 0.0f;
        b1 = (n0 + 1) < N ? b_ptr[1] : 0.0f;
        b2 = (n0 + 2) < N ? b_ptr[2] : 0.0f;
        b3 = (n0 + 3) < N ? b_ptr[3] : 0.0f;
      }
      acc0 += a_val * b0;
      acc1 += a_val * b1;
      acc2 += a_val * b2;
      acc3 += a_val * b3;
    }

    if (C) {
      const float *c_ptr = C + a * N + n0;
      float c0, c1, c2, c3;
      bool in_bounds4 = (n0 + 3) < N;
      if (in_bounds4 && ((reinterpret_cast<uintptr_t>(c_ptr) & 0xF) == 0)) {
        asm volatile("{\n\t"
                     "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n\t"
                     "}\n"
                     : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
                     : "l"(c_ptr));
      } else {
        c0 = (n0 + 0) < N ? c_ptr[0] : 0.0f;
        c1 = (n0 + 1) < N ? c_ptr[1] : 0.0f;
        c2 = (n0 + 2) < N ? c_ptr[2] : 0.0f;
        c3 = (n0 + 3) < N ? c_ptr[3] : 0.0f;
      }
      acc0 += c0;
      acc1 += c1;
      acc2 += c2;
      acc3 += c3;
    }

    float *d_ptr = D + d * N + n0;
    if ((n0 + 0) < N)
      atomicAdd(&d_ptr[0], acc0);
    if ((n0 + 1) < N)
      atomicAdd(&d_ptr[1], acc1);
    if ((n0 + 2) < N)
      atomicAdd(&d_ptr[2], acc2);
    if ((n0 + 3) < N)
      atomicAdd(&d_ptr[3], acc3);
  }
}

// ---- CUB-based gather-scatter matmul (templated datatypes)
// -------------------
namespace {

template <typename T> struct ToFloat;

template <> struct ToFloat<float> {
  static __device__ __forceinline__ float convert(float x) { return x; }
};

template <> struct ToFloat<half> {
  static __device__ __forceinline__ float convert(half x) {
    return __half2float(x);
  }
};

template <> struct ToFloat<__nv_bfloat16> {
  static __device__ __forceinline__ float convert(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
};

template <typename T> struct FromFloat;

template <> struct FromFloat<float> {
  static __device__ __forceinline__ float convert(float x) { return x; }
};

template <> struct FromFloat<half> {
  static __device__ __forceinline__ half convert(float x) {
    return __float2half(x);
  }
};

template <> struct FromFloat<__nv_bfloat16> {
  static __device__ __forceinline__ __nv_bfloat16 convert(float x) {
    return __float2bfloat16(x);
  }
};

// Atomic add for float/half/bfloat16
static __device__ __forceinline__ void atomicAddTyped(float *addr, float val) {
  atomicAdd(addr, val);
}

static __device__ __forceinline__ void atomicAddTyped(half *addr, float val) {
  // Implement via CAS on 32-bit word containing the target half
  uintptr_t int_addr = reinterpret_cast<uintptr_t>(addr);
  unsigned int *base = reinterpret_cast<unsigned int *>(int_addr & ~0x3ULL);
  bool high = (int_addr & 0x2ULL) != 0ULL;
  unsigned int old = *base;
  unsigned int assumed;
  do {
    assumed = old;
    unsigned short hbits = high
                               ? static_cast<unsigned short>(assumed >> 16)
                               : static_cast<unsigned short>(assumed & 0xFFFFu);
    half hval = __ushort_as_half(hbits);
    float f = __half2float(hval) + val;
    unsigned short new_hbits = __half_as_ushort(__float2half(f));
    unsigned int new_word =
        high ? ((assumed & 0x0000FFFFu) |
                (static_cast<unsigned int>(new_hbits) << 16))
             : ((assumed & 0xFFFF0000u) | static_cast<unsigned int>(new_hbits));
    old = atomicCAS(base, assumed, new_word);
  } while (old != assumed);
}

static __device__ __forceinline__ void atomicAddTyped(__nv_bfloat16 *addr,
                                                      float val) {
  uintptr_t int_addr = reinterpret_cast<uintptr_t>(addr);
  unsigned int *base = reinterpret_cast<unsigned int *>(int_addr & ~0x3ULL);
  bool high = (int_addr & 0x2ULL) != 0ULL;
  unsigned int old = *base;
  unsigned int assumed;
  do {
    assumed = old;
    unsigned short bbits = high
                               ? static_cast<unsigned short>(assumed >> 16)
                               : static_cast<unsigned short>(assumed & 0xFFFFu);
    __nv_bfloat16 bval = __ushort_as_bfloat16(bbits);
    float f = __bfloat162float(bval) + val;
    unsigned short new_bbits = __bfloat16_as_ushort(__float2bfloat16(f));
    unsigned int new_word =
        high ? ((assumed & 0x0000FFFFu) |
                (static_cast<unsigned int>(new_bbits) << 16))
             : ((assumed & 0xFFFF0000u) | static_cast<unsigned int>(new_bbits));
    old = atomicCAS(base, assumed, new_word);
  } while (old != assumed);
}

template <typename TA, typename TB, typename TC, typename TD, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
__global__ void
implicit_gemm_cub_blockload(const TA *__restrict__ A,    // [M, K]
                            const TB *__restrict__ B,    // [K, N]
                            const TC *__restrict__ Copt, // [M, N] or nullptr
                            const long *__restrict__ a_inds, // [P]
                            const long *__restrict__ d_inds, // [P]
                            TD *__restrict__ D,              // [Q, N]
                            int M, int K, int N, int P, int Q) {
  constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;
  int n0 = blockIdx.x * TILE;
  int tid = threadIdx.x;

  using BlockLoadB = cub::BlockLoad<TB, BLOCK_THREADS, ITEMS_PER_THREAD,
                                    cub::BLOCK_LOAD_VECTORIZE>;
  using BlockLoadC = cub::BlockLoad<TC, BLOCK_THREADS, ITEMS_PER_THREAD,
                                    cub::BLOCK_LOAD_VECTORIZE>;
  __shared__ typename BlockLoadB::TempStorage temp_storage_b;
  __shared__ typename BlockLoadC::TempStorage temp_storage_c;
  for (int p = blockIdx.y; p < P; p += gridDim.y) {
    long a = a_inds[p];
    long d = d_inds[p];
    if (a < 0 || a >= M || d < 0 || d >= Q)
      continue;

    float acc[ITEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
      acc[i] = 0.0f;

    // Iterate over K
    for (int k = 0; k < K; ++k) {
      float aval = ToFloat<TA>::convert(A[a * (size_t)K + k]);

      // Load a TILE segment from B[k, n0:n0+TILE)
      TB b_vals[ITEMS_PER_THREAD];
      int valid = 0;
      if (n0 < N) {
        int remaining = N - n0;
        valid = remaining > TILE ? TILE : remaining;
      }
      BlockLoadB(temp_storage_b)
          .Load(B + (size_t)k * N + n0, b_vals, valid, TB());
      __syncthreads();

#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        acc[i] += aval * ToFloat<TB>::convert(b_vals[i]);
      }
    }

    // Optional bias C[a, n]
    if (Copt) {
      TC c_vals[ITEMS_PER_THREAD];
      int valid = 0;
      if (n0 < N) {
        int remaining = N - n0;
        valid = remaining > TILE ? TILE : remaining;
      }
      BlockLoadC(temp_storage_c)
          .Load(Copt + (size_t)a * N + n0, c_vals, valid, TC());
      __syncthreads();
#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        acc[i] += ToFloat<TC>::convert(c_vals[i]);
      }
    }

    // Scatter-add into D[d, n]
    int base_n = n0 + tid * ITEMS_PER_THREAD;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      int n = base_n + i;
      if (n < N) {
        atomicAddTyped(D + (size_t)d * N + n, acc[i]);
      }
    }
  }
}

} // anonymous namespace

// Host entry
std::vector<float>
benchmark_implicit_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                        torch::Tensor a_inds, torch::Tensor d_inds,
                        torch::Tensor D, int method, int iterations = 100,
                        bool use_store = false) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  int P = a_inds.size(0);
  int Q = D.size(0);

  // Move to GPU/contig
  A = A.cuda().contiguous();
  B = B.cuda().contiguous();
  if (C.defined() && C.numel() > 0)
    C = C.cuda().contiguous();
  a_inds = a_inds.cuda().contiguous();
  d_inds = d_inds.cuda().contiguous();
  D = D.cuda().contiguous();

  dim3 block = calculate_block_size(method);
  dim3 grid = calculate_grid_size(P, N, method, block);

  // Warmup
  for (int i = 0; i < 5; ++i) {
    if (method == NAIVE_F32) {
      implicit_gemm_naive_f32<<<grid, block>>>(
          A.data_ptr<float>(), B.data_ptr<float>(),
          C.defined() ? C.data_ptr<float>() : nullptr, a_inds.data_ptr<long>(),
          d_inds.data_ptr<long>(), D.data_ptr<float>(), M, K, N, P, Q);
    } else if (method == WMMA_F16_ACC_F32) {
      implicit_gemm_wmma_f16_acc_f32<<<grid, dim3(32, 1, 1)>>>(
          reinterpret_cast<half *>(A.data_ptr<at::Half>()),
          reinterpret_cast<half *>(B.data_ptr<at::Half>()),
          C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                      : nullptr,
          a_inds.data_ptr<long>(), d_inds.data_ptr<long>(), D.data_ptr<float>(),
          M, K, N, P, Q);
    } else if (method == F32_PTX_V4) {
      implicit_gemm_f32_ptx_v4<<<grid, block>>>(
          A.data_ptr<float>(), B.data_ptr<float>(),
          C.defined() ? C.data_ptr<float>() : nullptr, a_inds.data_ptr<long>(),
          d_inds.data_ptr<long>(), D.data_ptr<float>(), M, K, N, P, Q);
    } else if (method == WMMA_F16_ACC_F32_DB_AMPERE) {
      implicit_gemm_wmma_f16_acc_f32_db_ampere<<<grid, dim3(32, 1, 1)>>>(
          reinterpret_cast<half *>(A.data_ptr<at::Half>()),
          reinterpret_cast<half *>(B.data_ptr<at::Half>()),
          C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                      : nullptr,
          a_inds.data_ptr<long>(), d_inds.data_ptr<long>(), D.data_ptr<float>(),
          M, K, N, P, Q);
    } else if (method == WMMA_DB_AMPERE_GENERIC ||
               method == WMMA_DB_AMPERE_GENERIC_STORE) {
      // Use SM80 WMMA double-buffer kernel; optional non-atomic store via
      // use_store flag
      if (method == WMMA_DB_AMPERE_GENERIC_STORE)
        use_store = true;
      auto Adtype = A.scalar_type();
      auto Ddtype = D.scalar_type();
      if (Adtype == at::kHalf && Ddtype == at::kFloat) {
        if (use_store) {
          launch_wmma_db_ampere_f16_f32_store(
              grid, dim3(32, 1, 1), nullptr,
              reinterpret_cast<half *>(A.data_ptr<at::Half>()),
              reinterpret_cast<half *>(B.data_ptr<at::Half>()),
              C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                          : nullptr,
              a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
              D.data_ptr<float>(), M, K, N, P, Q);
        } else {
          launch_wmma_db_ampere_f16_f32(
              grid, dim3(32, 1, 1), nullptr,
              reinterpret_cast<half *>(A.data_ptr<at::Half>()),
              reinterpret_cast<half *>(B.data_ptr<at::Half>()),
              C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                          : nullptr,
              a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
              D.data_ptr<float>(), M, K, N, P, Q);
        }
      } else if (Adtype == at::kHalf && Ddtype == at::kHalf) {
        if (use_store) {
          launch_wmma_db_ampere_f16_f16_store(
              grid, dim3(32, 1, 1), nullptr,
              reinterpret_cast<half *>(A.data_ptr<at::Half>()),
              reinterpret_cast<half *>(B.data_ptr<at::Half>()),
              C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                          : nullptr,
              a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
              reinterpret_cast<half *>(D.data_ptr<at::Half>()), M, K, N, P, Q);
        } else {
          launch_wmma_db_ampere_f16_f16(
              grid, dim3(32, 1, 1), nullptr,
              reinterpret_cast<half *>(A.data_ptr<at::Half>()),
              reinterpret_cast<half *>(B.data_ptr<at::Half>()),
              C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                          : nullptr,
              a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
              reinterpret_cast<half *>(D.data_ptr<at::Half>()), M, K, N, P, Q);
        }
      }
    } else if (method == CUB_F32_BLOCKLOAD) {
      auto Adtype = A.scalar_type();
      auto Bdtype = B.scalar_type();
      auto Ddtype = D.scalar_type();
      // Support: A/B/C in half or bfloat16; D in float/half/bfloat16
      if (Adtype == at::kHalf && Bdtype == at::kHalf) {
        if (Ddtype == at::kFloat) {
          implicit_gemm_cub_blockload<half, half, half, float, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<half *>(A.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                  C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  D.data_ptr<float>(), M, K, N, P, Q);
        } else if (Ddtype == at::kHalf) {
          implicit_gemm_cub_blockload<half, half, half, half, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<half *>(A.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                  C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  reinterpret_cast<half *>(D.data_ptr<at::Half>()), M, K, N, P,
                  Q);
        } else if (Ddtype == at::kBFloat16) {
          implicit_gemm_cub_blockload<half, half, half, __nv_bfloat16, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<half *>(A.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                  C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  reinterpret_cast<__nv_bfloat16 *>(D.data_ptr<at::BFloat16>()),
                  M, K, N, P, Q);
        }
      } else if (Adtype == at::kBFloat16 && Bdtype == at::kBFloat16) {
        if (Ddtype == at::kFloat) {
          implicit_gemm_cub_blockload<__nv_bfloat16, __nv_bfloat16,
                                      __nv_bfloat16, float, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<__nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
                  reinterpret_cast<__nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                  C.defined() ? reinterpret_cast<__nv_bfloat16 *>(
                                    C.data_ptr<at::BFloat16>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  D.data_ptr<float>(), M, K, N, P, Q);
        } else if (Ddtype == at::kBFloat16) {
          implicit_gemm_cub_blockload<__nv_bfloat16, __nv_bfloat16,
                                      __nv_bfloat16, __nv_bfloat16, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<__nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
                  reinterpret_cast<__nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                  C.defined() ? reinterpret_cast<__nv_bfloat16 *>(
                                    C.data_ptr<at::BFloat16>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  reinterpret_cast<__nv_bfloat16 *>(D.data_ptr<at::BFloat16>()),
                  M, K, N, P, Q);
        } else if (Ddtype == at::kHalf) {
          implicit_gemm_cub_blockload<__nv_bfloat16, __nv_bfloat16,
                                      __nv_bfloat16, half, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<__nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
                  reinterpret_cast<__nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                  C.defined() ? reinterpret_cast<__nv_bfloat16 *>(
                                    C.data_ptr<at::BFloat16>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  reinterpret_cast<half *>(D.data_ptr<at::Half>()), M, K, N, P,
                  Q);
        }
      }
    }
  }
  cudaDeviceSynchronize();
  // Reset D to avoid warmup accumulation affecting correctness/timing
  D.zero_();
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  std::vector<float> times;
  times.reserve(iterations);

  for (int it = 0; it < iterations; ++it) {
    cudaEventRecord(start);
    if (method == NAIVE_F32) {
      implicit_gemm_naive_f32<<<grid, block>>>(
          A.data_ptr<float>(), B.data_ptr<float>(),
          C.defined() ? C.data_ptr<float>() : nullptr, a_inds.data_ptr<long>(),
          d_inds.data_ptr<long>(), D.data_ptr<float>(), M, K, N, P, Q);
    } else if (method == WMMA_F16_ACC_F32) {
      implicit_gemm_wmma_f16_acc_f32<<<grid, dim3(32, 1, 1)>>>(
          reinterpret_cast<half *>(A.data_ptr<at::Half>()),
          reinterpret_cast<half *>(B.data_ptr<at::Half>()),
          C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                      : nullptr,
          a_inds.data_ptr<long>(), d_inds.data_ptr<long>(), D.data_ptr<float>(),
          M, K, N, P, Q);
    } else if (method == F32_PTX_V4) {
      implicit_gemm_f32_ptx_v4<<<grid, block>>>(
          A.data_ptr<float>(), B.data_ptr<float>(),
          C.defined() ? C.data_ptr<float>() : nullptr, a_inds.data_ptr<long>(),
          d_inds.data_ptr<long>(), D.data_ptr<float>(), M, K, N, P, Q);
    } else if (method == WMMA_F16_ACC_F32_DB_AMPERE) {
      implicit_gemm_wmma_f16_acc_f32_db_ampere<<<grid, dim3(32, 1, 1)>>>(
          reinterpret_cast<half *>(A.data_ptr<at::Half>()),
          reinterpret_cast<half *>(B.data_ptr<at::Half>()),
          C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                      : nullptr,
          a_inds.data_ptr<long>(), d_inds.data_ptr<long>(), D.data_ptr<float>(),
          M, K, N, P, Q);
    } else if (method == WMMA_DB_AMPERE_GENERIC ||
               method == WMMA_DB_AMPERE_GENERIC_STORE) {
      if (method == WMMA_DB_AMPERE_GENERIC_STORE)
        use_store = true;
      auto Adtype = A.scalar_type();
      auto Ddtype = D.scalar_type();
      if (Adtype == at::kHalf && Ddtype == at::kFloat) {
        if (use_store) {
          launch_wmma_db_ampere_f16_f32_store(
              grid, dim3(32, 1, 1), nullptr,
              reinterpret_cast<half *>(A.data_ptr<at::Half>()),
              reinterpret_cast<half *>(B.data_ptr<at::Half>()),
              C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                          : nullptr,
              a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
              D.data_ptr<float>(), M, K, N, P, Q);
        } else {
          launch_wmma_db_ampere_f16_f32(
              grid, dim3(32, 1, 1), nullptr,
              reinterpret_cast<half *>(A.data_ptr<at::Half>()),
              reinterpret_cast<half *>(B.data_ptr<at::Half>()),
              C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                          : nullptr,
              a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
              D.data_ptr<float>(), M, K, N, P, Q);
        }
      } else if (Adtype == at::kHalf && Ddtype == at::kHalf) {
        if (use_store) {
          launch_wmma_db_ampere_f16_f16_store(
              grid, dim3(32, 1, 1), nullptr,
              reinterpret_cast<half *>(A.data_ptr<at::Half>()),
              reinterpret_cast<half *>(B.data_ptr<at::Half>()),
              C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                          : nullptr,
              a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
              reinterpret_cast<half *>(D.data_ptr<at::Half>()), M, K, N, P, Q);
        } else {
          launch_wmma_db_ampere_f16_f16(
              grid, dim3(32, 1, 1), nullptr,
              reinterpret_cast<half *>(A.data_ptr<at::Half>()),
              reinterpret_cast<half *>(B.data_ptr<at::Half>()),
              C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                          : nullptr,
              a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
              reinterpret_cast<half *>(D.data_ptr<at::Half>()), M, K, N, P, Q);
        }
      }
    } else if (method == CUB_F32_BLOCKLOAD) {
      auto Adtype = A.scalar_type();
      auto Bdtype = B.scalar_type();
      auto Ddtype = D.scalar_type();
      if (Adtype == at::kHalf && Bdtype == at::kHalf) {
        if (Ddtype == at::kFloat) {
          implicit_gemm_cub_blockload<half, half, half, float, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<half *>(A.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                  C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  D.data_ptr<float>(), M, K, N, P, Q);
        } else if (Ddtype == at::kHalf) {
          implicit_gemm_cub_blockload<half, half, half, half, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<half *>(A.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                  C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  reinterpret_cast<half *>(D.data_ptr<at::Half>()), M, K, N, P,
                  Q);
        } else if (Ddtype == at::kBFloat16) {
          implicit_gemm_cub_blockload<half, half, half, __nv_bfloat16, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<half *>(A.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                  C.defined() ? reinterpret_cast<half *>(C.data_ptr<at::Half>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  reinterpret_cast<__nv_bfloat16 *>(D.data_ptr<at::BFloat16>()),
                  M, K, N, P, Q);
        }
      } else if (Adtype == at::kBFloat16 && Bdtype == at::kBFloat16) {
        if (Ddtype == at::kFloat) {
          implicit_gemm_cub_blockload<__nv_bfloat16, __nv_bfloat16,
                                      __nv_bfloat16, float, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<__nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
                  reinterpret_cast<__nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                  C.defined() ? reinterpret_cast<__nv_bfloat16 *>(
                                    C.data_ptr<at::BFloat16>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  D.data_ptr<float>(), M, K, N, P, Q);
        } else if (Ddtype == at::kBFloat16) {
          implicit_gemm_cub_blockload<__nv_bfloat16, __nv_bfloat16,
                                      __nv_bfloat16, __nv_bfloat16, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<__nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
                  reinterpret_cast<__nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                  C.defined() ? reinterpret_cast<__nv_bfloat16 *>(
                                    C.data_ptr<at::BFloat16>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  reinterpret_cast<__nv_bfloat16 *>(D.data_ptr<at::BFloat16>()),
                  M, K, N, P, Q);
        } else if (Ddtype == at::kHalf) {
          implicit_gemm_cub_blockload<__nv_bfloat16, __nv_bfloat16,
                                      __nv_bfloat16, half, 128, 4>
              <<<grid, dim3(128, 1, 1)>>>(
                  reinterpret_cast<__nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
                  reinterpret_cast<__nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                  C.defined() ? reinterpret_cast<__nv_bfloat16 *>(
                                    C.data_ptr<at::BFloat16>())
                              : nullptr,
                  a_inds.data_ptr<long>(), d_inds.data_ptr<long>(),
                  reinterpret_cast<half *>(D.data_ptr<at::Half>()), M, K, N, P,
                  Q);
        }
      }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    times.push_back(ms);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return times;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("benchmark_implicit_gemm", &benchmark_implicit_gemm,
        "Benchmark implicit GEMM: D[d]=A[a]@B + C[a]", py::arg("A"),
        py::arg("B"), py::arg("C"), py::arg("a_inds"), py::arg("d_inds"),
        py::arg("D"), py::arg("method"), py::arg("iterations") = 100,
        py::arg("use_store") = false);
}
