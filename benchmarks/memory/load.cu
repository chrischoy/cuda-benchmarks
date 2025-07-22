#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <utility>
#include <tuple>
#include "matrix_loading_common.cuh"
#include "vector_types.cuh"

// CUB includes for optimal loading
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_store.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

using namespace cuda_benchmark::memory;

// =============================================================================
// MATRIX LOADING KERNELS
// =============================================================================

// Element-wise loading kernel - pure loading test
template <typename T, bool store = false>
__global__ void matrix_load_elementwise(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ T shared_cache[1024]; // Shared memory cache

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        long idx = row * cols + col;
        // Pure loading test - store in shared memory
        T value = input[idx];
        T processed_value = value * static_cast<T>(1.001f); // Minimal computation to prevent optimization
        shared_cache[tid] = processed_value;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = processed_value;
        }
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Base template for vectorized operations
template <typename T, bool store = false>
__global__ void matrix_load_vectorized2(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ T shared_cache[1024]; // Shared memory cache

    // Default implementation for non-specialized types - fall back to elementwise
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        long idx = row * cols + col;
        T processed_value = input[idx] * T(1.001f);
        shared_cache[tid] = processed_value;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = processed_value;
        }
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Vectorized loading using float2 (64-bit loads) - pure loading test
template <bool store = false>
__global__ void matrix_load_vectorized2(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ float2 shared_cache[1024]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Grid-stride loop with float2 vectorized loads
    for (long idx = tid * 2; idx + 1 < total_elements; idx += (blockDim.x * gridDim.x) * 2)
    {
        // Load 2 floats at once using vectorized instruction
        float2 values = __ldg(reinterpret_cast<const float2 *>(&input[idx]));

        // Minimal processing to prevent optimization
        values.x *= 1.001f;
        values.y *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[local_tid] = values;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = values.x;
            output[idx + 1] = values.y;
        }
        __syncthreads();
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Vectorized loading using float4 (128-bit loads) - pure loading test
template <bool store = false>
__global__ void matrix_load_vectorized4(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ float4 shared_cache[1024]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Grid-stride loop with float4 vectorized loads
    for (long idx = tid * 4; idx + 3 < total_elements; idx += (blockDim.x * gridDim.x) * 4)
    {
        // Load 4 floats at once using vectorized instruction
        float4 values = __ldg(reinterpret_cast<const float4 *>(&input[idx]));

        // Minimal processing to prevent optimization
        values.x *= 1.001f;
        values.y *= 1.001f;
        values.z *= 1.001f;
        values.w *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[local_tid] = values;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = values.x;
            output[idx + 1] = values.y;
            output[idx + 2] = values.z;
            output[idx + 3] = values.w;
        }
        __syncthreads();
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Base template for vectorized8 operations
template <typename T, bool store = false>
__global__ void matrix_load_vectorized8(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ T shared_cache[1024]; // Shared memory cache

    // Default implementation for non-specialized types - fall back to elementwise
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        long idx = row * cols + col;
        T processed_value = input[idx] * T(1.001f);
        shared_cache[tid] = processed_value;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = processed_value;
        }
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Vectorized loading using float8 (256-bit loads as 2x float4) - pure loading test
template <bool store = false>
__global__ void matrix_load_vectorized8(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ float8 shared_cache[1024]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Grid-stride loop with float8 vectorized loads
    for (long idx = tid * 8; idx + 7 < total_elements; idx += (blockDim.x * gridDim.x) * 8)
    {
        // Load 8 floats at once using two vectorized float4 instructions
        float8 values = load_float8(&input[idx]);

        // Minimal processing to prevent optimization
        values.lo.x *= 1.001f;
        values.lo.y *= 1.001f;
        values.lo.z *= 1.001f;
        values.lo.w *= 1.001f;
        values.hi.x *= 1.001f;
        values.hi.y *= 1.001f;
        values.hi.z *= 1.001f;
        values.hi.w *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[local_tid] = values;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = values.lo.x;
            output[idx + 1] = values.lo.y;
            output[idx + 2] = values.lo.z;
            output[idx + 3] = values.lo.w;
            output[idx + 4] = values.hi.x;
            output[idx + 5] = values.hi.y;
            output[idx + 6] = values.hi.z;
            output[idx + 7] = values.hi.w;
        }
        __syncthreads();
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Row-wise coalesced access pattern - pure loading test
template <typename T, bool store = false>
__global__ void matrix_load_coalesced_row(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ T shared_cache[1024]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Grid-stride loop for optimal memory coalescing
    for (long idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x)
    {
        T value = __ldg(&input[idx]);
        T processed_value = value * static_cast<T>(1.001);
        // Fair comparison: ALL threads store their loaded data in shared memory
        shared_cache[local_tid] = processed_value;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = processed_value;
        }
        __syncthreads();
    }
}

// Column-wise access (non-coalesced for comparison)
template <typename T, bool store = false>
__global__ void matrix_load_coalesced_column(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ T shared_cache[256]; // Shared memory cache

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        // Column-wise strided access pattern
        for (long r = row; r < rows; r += blockDim.y * gridDim.y)
        {
            long idx = r * cols + col;
            T value = __ldg(&input[idx]);
            T processed_value = value * static_cast<T>(1.001);
            shared_cache[tid] = processed_value;

            // Conditionally write to output for debugging
            if constexpr (store)
            {
                output[idx] = processed_value;
            }
        }
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Coalesced float4 loading - pure loading test with 1D grid
template <bool store = false>
__global__ void matrix_load_coalesced_float4(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ float4 shared_cache[1024]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Grid-stride loop with float4 vectorized loads
    for (long idx = tid * 4; idx + 3 < total_elements; idx += (blockDim.x * gridDim.x) * 4)
    {
        // Load 4 floats at once using vectorized instruction
        float4 values = __ldg(reinterpret_cast<const float4 *>(&input[idx]));

        // Minimal processing to prevent optimization
        values.x *= 1.001f;
        values.y *= 1.001f;
        values.z *= 1.001f;
        values.w *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[local_tid] = values;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = values.x;
            output[idx + 1] = values.y;
            output[idx + 2] = values.z;
            output[idx + 3] = values.w;
        }
        __syncthreads();
    }
}

// Coalesced float8 loading - pure loading test with 1D grid
template <bool store = false>
__global__ void matrix_load_coalesced_float8(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ float8 shared_cache[1024]; // Shared memory cache

    ulong tid = blockIdx.x * blockDim.x + threadIdx.x;
    ulong local_tid = threadIdx.x;
    ulong total_elements = rows * cols;

    // Grid-stride loop with float8 vectorized loads
    for (ulong idx = tid * 8; idx + 7 < total_elements; idx += (blockDim.x * gridDim.x) * 8)
    {
        // Load 8 floats at once using two vectorized float4 instructions
        float8 values = load_float8(&input[idx]);

        // Minimal processing on all 8 elements
        values.lo.x *= 1.001f;
        values.lo.y *= 1.001f;
        values.lo.z *= 1.001f;
        values.lo.w *= 1.001f;
        values.hi.x *= 1.001f;
        values.hi.y *= 1.001f;
        values.hi.z *= 1.001f;
        values.hi.w *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[local_tid] = values;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = values.lo.x;
            output[idx + 1] = values.lo.y;
            output[idx + 2] = values.lo.z;
            output[idx + 3] = values.lo.w;
            output[idx + 4] = values.hi.x;
            output[idx + 5] = values.hi.y;
            output[idx + 6] = values.hi.z;
            output[idx + 7] = values.hi.w;
        }
        __syncthreads();
    }
}

// Shared memory tiled loading - pure loading test
template <typename T, int TILE_SIZE = 32, bool store = false>
__global__ void matrix_load_shared_memory(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ T tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    long row = blockIdx.y * TILE_SIZE + threadIdx.y;
    long col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Cooperative loading into shared memory
    if (row < rows && col < cols)
    {
        tile[threadIdx.y][threadIdx.x] = __ldg(&input[row * cols + col]);
    }
    else
    {
        tile[threadIdx.y][threadIdx.x] = static_cast<T>(0);
    }

    __syncthreads();

    // Process data from shared memory (pure loading test - no write back)
    T value = tile[threadIdx.y][threadIdx.x] * static_cast<T>(1.001);

    // Store processed value back in tile to ensure computation isn't optimized away
    tile[threadIdx.y][threadIdx.x] = value;

    // Conditionally write to output for debugging
    if constexpr (store)
    {
        if (row < rows && col < cols)
        {
            output[row * cols + col] = value;
        }
    }

    __syncthreads();
}

// CUB Device-level loading operations - optimized with cache modifiers
template <typename T, bool store = false>
__global__ void matrix_load_cub_device(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ T shared_cache[1024]; // Shared memory cache

    // Use CUB's thread-level primitives with optimal cache modifiers
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    for (long idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x)
    {
        // Use LOAD_LDG for read-only data (cached global load)
        T value = cub::ThreadLoad<cub::LOAD_LDG>(&input[idx]);
        T processed_value = value * static_cast<T>(1.001);
        shared_cache[threadIdx.x] = processed_value;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = processed_value;
        }
    }
}

// CUB Device-level loading with cache-modified iterators
template <typename T, bool store = false>
__global__ void matrix_load_cub_device_cache_modified(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ T shared_cache[1024]; // Shared memory cache

    // Use cache-modified iterators for optimal memory access patterns
    using CacheModifiedInputIterator = cub::CacheModifiedInputIterator<cub::LOAD_LDG, T, size_t>;
    CacheModifiedInputIterator cache_modified_input(input);

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    for (long idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x)
    {
        T value = cache_modified_input[idx];
        T processed_value = value * static_cast<T>(1.001);
        shared_cache[threadIdx.x] = processed_value;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = processed_value;
        }
    }
}

// CUB Block-level loading operations - optimized with vectorization
template <typename T, int BLOCK_SIZE = 256, bool store = false>
__global__ void matrix_load_cub_block(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    // Use BLOCK_LOAD_VECTORIZE for optimal performance when data is aligned
    // and ITEMS_PER_THREAD is even (4 items per thread)
    typedef cub::BlockLoad<T, BLOCK_SIZE, 4, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;

    __shared__ typename BlockLoad::TempStorage load_temp_storage;

    long block_offset = blockIdx.x * BLOCK_SIZE * 4;
    long total_elements = rows * cols;

    if (block_offset < total_elements)
    {
        T thread_data[4];

        // Cooperative block loading using CUB with vectorization
        long valid_items = min(BLOCK_SIZE * 4, int(total_elements - block_offset));
        BlockLoad(load_temp_storage).Load(&input[block_offset], thread_data, valid_items);

        __syncthreads();

// Process loaded data
#pragma unroll
        for (long i = 0; i < 4; ++i)
        {
            thread_data[i] *= static_cast<T>(1.001);
        }

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            long thread_idx = threadIdx.x;
            for (long i = 0; i < 4; ++i)
            {
                long idx = block_offset + thread_idx * 4 + i;
                if (idx < total_elements)
                {
                    output[idx] = thread_data[i];
                }
            }
        }
    }
}

// CUB Block-level loading with warp transpose for better coalescing
template <typename T, int BLOCK_SIZE = 256, bool store = false>
__global__ void matrix_load_cub_block_warp_transpose(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    // Use BLOCK_LOAD_WARP_TRANSPOSE for optimal memory coalescing
    // when BLOCK_THREADS is a multiple of warp size (32)
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size (32)");
    typedef cub::BlockLoad<T, BLOCK_SIZE, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;

    __shared__ typename BlockLoad::TempStorage load_temp_storage;

    long block_offset = blockIdx.x * BLOCK_SIZE * 4;
    long total_elements = rows * cols;

    if (block_offset < total_elements)
    {
        T thread_data[4];

        // Cooperative block loading using warp-striped pattern then transpose
        long valid_items = min(BLOCK_SIZE * 4, int(total_elements - block_offset));
        BlockLoad(load_temp_storage).Load(&input[block_offset], thread_data, valid_items);

        __syncthreads();

// Process loaded data
#pragma unroll
        for (long i = 0; i < 4; ++i)
        {
            thread_data[i] *= static_cast<T>(1.001);
        }

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            long thread_idx = threadIdx.x;
            for (long i = 0; i < 4; ++i)
            {
                long idx = block_offset + thread_idx * 4 + i;
                if (idx < total_elements)
                {
                    output[idx] = thread_data[i];
                }
            }
        }
    }
}

// CUB Block-level loading with striped transpose for maximum coalescing
template <typename T, int BLOCK_SIZE = 256, bool store = false>
__global__ void matrix_load_cub_block_striped_transpose(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    // Use BLOCK_LOAD_TRANSPOSE for maximum memory coalescing regardless of items per thread
    typedef cub::BlockLoad<T, BLOCK_SIZE, 4, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;

    __shared__ typename BlockLoad::TempStorage load_temp_storage;

    long block_offset = blockIdx.x * BLOCK_SIZE * 4;
    long total_elements = rows * cols;

    if (block_offset < total_elements)
    {
        T thread_data[4];

        // Cooperative block loading using striped pattern then transpose to blocked
        long valid_items = min(BLOCK_SIZE * 4, int(total_elements - block_offset));
        BlockLoad(load_temp_storage).Load(&input[block_offset], thread_data, valid_items);

        __syncthreads();

// Process loaded data
#pragma unroll
        for (long i = 0; i < 4; ++i)
        {
            thread_data[i] *= static_cast<T>(1.001);
        }

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            long thread_idx = threadIdx.x;
            for (long i = 0; i < 4; ++i)
            {
                long idx = block_offset + thread_idx * 4 + i;
                if (idx < total_elements)
                {
                    output[idx] = thread_data[i];
                }
            }
        }
    }
}

// CUB Warp-level loading operations - optimized with vectorization
template <typename T, bool store = false>
__global__ void matrix_load_cub_warp(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    // Use WARP_LOAD_VECTORIZE for optimal performance when data is aligned
    typedef cub::WarpLoad<T, 4, cub::WARP_LOAD_VECTORIZE> WarpLoad;

    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_offset = warp_id * WARP_SIZE * 4;
    long total_elements = rows * cols;

    if (warp_offset < total_elements)
    {
        T thread_data[4];

        // Warp-level cooperative loading with vectorization
        long valid_items = min(WARP_SIZE * 4, int(total_elements - warp_offset));
        WarpLoad().Load(&input[warp_offset], thread_data, valid_items);

// Process loaded data
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            thread_data[i] *= static_cast<T>(1.001);
        }

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            for (int i = 0; i < 4; ++i)
            {
                long idx = warp_offset + lane_id * 4 + i;
                if (idx < total_elements)
                {
                    output[idx] = thread_data[i];
                }
            }
        }
    }
}

// CUB Warp-level loading with striped pattern for better coalescing
template <typename T, bool store = false>
__global__ void matrix_load_cub_warp_striped(
    const T *__restrict__ input,
    T *__restrict__ output,
    size_t rows,
    size_t cols)
{
    // Use WARP_LOAD_STRIPED for optimal memory coalescing
    typedef cub::WarpLoad<T, 4, cub::WARP_LOAD_STRIPED> WarpLoad;

    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_offset = warp_id * WARP_SIZE * 4;
    long total_elements = rows * cols;

    if (warp_offset < total_elements)
    {
        T thread_data[4];

        // Warp-level cooperative loading with striped pattern
        long valid_items = min(WARP_SIZE * 4, int(total_elements - warp_offset));
        WarpLoad().Load(&input[warp_offset], thread_data, valid_items);

// Process loaded data
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            thread_data[i] *= static_cast<T>(1.001);
        }

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            for (int i = 0; i < 4; ++i)
            {
                long idx = warp_offset + lane_id * 4 + i;
                if (idx < total_elements)
                {
                    output[idx] = thread_data[i];
                }
            }
        }
    }
}

// Texture memory loading (for float only)
template <bool store = false>
__global__ void matrix_load_texture_float(
    cudaTextureObject_t tex_obj,
    float *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ float shared_cache[1024]; // Shared memory cache

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        long idx = row * cols + col;
        float value = tex1Dfetch<float>(tex_obj, idx);
        float processed_value = value * 1.001f;
        shared_cache[tid] = processed_value;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = processed_value;
        }
    }
}

// PTX float4 loading kernel using ld.global.v4.f32
template <bool store = false>
__global__ void matrix_load_ptx_float4(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ float4 shared_cache[1024]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Grid-stride loop with PTX float4 vectorized loads
    for (long idx = tid * 4; idx + 3 < total_elements; idx += (blockDim.x * gridDim.x) * 4)
    {
        // Load 4 floats using PTX ld.global.v4.f32 instruction
        float4 values;
        asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(values.x), "=f"(values.y), "=f"(values.z), "=f"(values.w)
                     : "l"((const float *)&input[idx]));

        // Minimal processing to prevent optimization
        values.x *= 1.001f;
        values.y *= 1.001f;
        values.z *= 1.001f;
        values.w *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[local_tid] = values;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = values.x;
            output[idx + 1] = values.y;
            output[idx + 2] = values.z;
            output[idx + 3] = values.w;
        }
        __syncthreads();
    }

    __syncthreads(); // Ensure shared memory write completes
}

// PTX float4 loading kernel using ld.global.nc.v4.f32 (non-cached)
template <bool store = false>
__global__ void matrix_load_ptx_float4_nc(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t cols)
{
    __shared__ float4 shared_cache[1024]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Grid-stride loop with PTX float4 vectorized loads (non-cached)
    for (long idx = tid * 4; idx + 3 < total_elements; idx += (blockDim.x * gridDim.x) * 4)
    {
        // Load 4 floats using PTX ld.global.nc.v4.f32 instruction (non-cached)
        float4 values;
        asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(values.x), "=f"(values.y), "=f"(values.z), "=f"(values.w)
                     : "l"((const float *)&input[idx]));

        // Minimal processing to prevent optimization
        values.x *= 1.001f;
        values.y *= 1.001f;
        values.z *= 1.001f;
        values.w *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[local_tid] = values;

        // Conditionally write to output for debugging
        if constexpr (store)
        {
            output[idx] = values.x;
            output[idx + 1] = values.y;
            output[idx + 2] = values.z;
            output[idx + 3] = values.w;
        }
        __syncthreads();
    }

    __syncthreads(); // Ensure shared memory write completes
}

// =============================================================================
// BENCHMARK FUNCTION
// =============================================================================

// Host function to benchmark matrix loading
std::tuple<std::vector<float>, torch::Tensor> benchmark_matrix_loading(
    torch::Tensor input,
    int method,
    int iterations = 100,
    bool store = false)
{
    auto output = torch::zeros_like(input);
    auto rows = input.size(0);
    auto cols = input.size(1);

    // Ensure tensors are on GPU and contiguous
    input = input.cuda().contiguous();
    output = output.cuda().contiguous();

    // Calculate grid and block dimensions using common utilities
    dim3 block_size = calculate_optimal_block_size(method);
    dim3 grid_size = calculate_optimal_grid_size(rows, cols, method, block_size);

    // Warmup runs
    for (int i = 0; i < 10; ++i)
    {
        switch (method)
        {
        case ELEMENTWISE:
            if (store)
            {
                matrix_load_elementwise<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_elementwise<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case VECTORIZED_FLOAT2:
            if (store)
            {
                matrix_load_vectorized2<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_vectorized2<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case VECTORIZED_FLOAT4:
            if (store)
            {
                matrix_load_vectorized4<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_vectorized4<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case VECTORIZED_FLOAT8:
            if (store)
            {
                matrix_load_vectorized8<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_vectorized8<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case COALESCED_ROW:
            if (store)
            {
                matrix_load_coalesced_row<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_coalesced_row<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case COALESCED_COLUMN:
            if (store)
            {
                matrix_load_coalesced_column<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_coalesced_column<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case COALESCED_FLOAT4:
            if (store)
            {
                matrix_load_coalesced_float4<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_coalesced_float4<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case COALESCED_FLOAT8:
            if (store)
            {
                matrix_load_coalesced_float8<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_coalesced_float8<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case SHARED_MEMORY_TILED:
            if (store)
            {
                matrix_load_shared_memory<float, 32, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_shared_memory<float, 32, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_DEVICE_LOAD:
            if (store)
            {
                matrix_load_cub_device<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_device<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_DEVICE_CACHE_MODIFIED:
            if (store)
            {
                matrix_load_cub_device_cache_modified<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_device_cache_modified<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_BLOCK_LOAD:
            if (store)
            {
                matrix_load_cub_block<float, 256, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_block<float, 256, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_BLOCK_WARP_TRANSPOSE:
            if (store)
            {
                matrix_load_cub_block_warp_transpose<float, 256, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_block_warp_transpose<float, 256, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_BLOCK_STRIPED_TRANSPOSE:
            if (store)
            {
                matrix_load_cub_block_striped_transpose<float, 256, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_block_striped_transpose<float, 256, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_WARP_LOAD:
            if (store)
            {
                matrix_load_cub_warp<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_warp<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_WARP_STRIPED:
            if (store)
            {
                matrix_load_cub_warp_striped<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_warp_striped<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case PTX_FLOAT4:
            if (store)
            {
                matrix_load_ptx_float4<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_ptx_float4<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case PTX_FLOAT4_NC:
            if (store)
            {
                matrix_load_ptx_float4_nc<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_ptx_float4_nc<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        default:
            if (store)
            {
                matrix_load_elementwise<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_elementwise<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        }
    }
    cudaDeviceSynchronize();

    // Benchmark timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i)
    {
        cudaEventRecord(start);

        switch (method)
        {
        case ELEMENTWISE:
            if (store)
            {
                matrix_load_elementwise<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_elementwise<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case VECTORIZED_FLOAT2:
            if (store)
            {
                matrix_load_vectorized2<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_vectorized2<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case VECTORIZED_FLOAT4:
            if (store)
            {
                matrix_load_vectorized4<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_vectorized4<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case VECTORIZED_FLOAT8:
            if (store)
            {
                matrix_load_vectorized8<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_vectorized8<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case COALESCED_ROW:
            if (store)
            {
                matrix_load_coalesced_row<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_coalesced_row<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case COALESCED_COLUMN:
            if (store)
            {
                matrix_load_coalesced_column<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_coalesced_column<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case COALESCED_FLOAT4:
            if (store)
            {
                matrix_load_coalesced_float4<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_coalesced_float4<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case COALESCED_FLOAT8:
            if (store)
            {
                matrix_load_coalesced_float8<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_coalesced_float8<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case SHARED_MEMORY_TILED:
            if (store)
            {
                matrix_load_shared_memory<float, 32, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_shared_memory<float, 32, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_DEVICE_LOAD:
            if (store)
            {
                matrix_load_cub_device<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_device<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_DEVICE_CACHE_MODIFIED:
            if (store)
            {
                matrix_load_cub_device_cache_modified<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_device_cache_modified<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_BLOCK_LOAD:
            if (store)
            {
                matrix_load_cub_block<float, 256, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_block<float, 256, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_BLOCK_WARP_TRANSPOSE:
            if (store)
            {
                matrix_load_cub_block_warp_transpose<float, 256, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_block_warp_transpose<float, 256, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_BLOCK_STRIPED_TRANSPOSE:
            if (store)
            {
                matrix_load_cub_block_striped_transpose<float, 256, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_block_striped_transpose<float, 256, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_WARP_LOAD:
            if (store)
            {
                matrix_load_cub_warp<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_warp<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case CUB_WARP_STRIPED:
            if (store)
            {
                matrix_load_cub_warp_striped<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_cub_warp_striped<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case PTX_FLOAT4:
            if (store)
            {
                matrix_load_ptx_float4<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_ptx_float4<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        case PTX_FLOAT4_NC:
            if (store)
            {
                matrix_load_ptx_float4_nc<true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_ptx_float4_nc<false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        default:
            if (store)
            {
                matrix_load_elementwise<float, true><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            else
            {
                matrix_load_elementwise<float, false><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
            }
            break;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        times.push_back(elapsed_ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return std::make_tuple(times, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("benchmark_matrix_loading", &benchmark_matrix_loading,
          "Benchmark matrix loading operations - returns (times, output_tensor)",
          py::arg("input"), py::arg("method"), py::arg("iterations") = 100, py::arg("store") = false);
}
