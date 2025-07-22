#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "matrix_loading_common.cuh"
#include "vector_types.cuh"

// CUB includes for optimal storing
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_store.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

using namespace cuda_benchmark::memory;

// =============================================================================
// MATRIX STORING KERNELS
// =============================================================================

// Element-wise storing kernel - pure storing test
template<typename T>
__global__ void matrix_store_elementwise(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    const int MAX_THREADS = 1024; // Maximum threads per block
    __shared__ T shared_cache[MAX_THREADS]; // Dynamic shared memory for data generation

    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    // Generate data in shared memory first
    if (tid < MAX_THREADS) {
        shared_cache[tid] = static_cast<T>(1.0f);
    }
    __syncthreads();

    // Loop-based approach to handle large matrices that exceed grid limits
    long thread_id = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
                     blockIdx.x * blockDim.x * blockDim.y + tid;
    long total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    for (long idx = thread_id; idx < total_elements; idx += total_threads) {
        if (tid < MAX_THREADS) {
            output[idx] = shared_cache[tid];
        }
    }
}

// Base template for vectorized operations
template<typename T>
__global__ void matrix_store_vectorized2(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    const int MAX_THREADS = 1024;
    __shared__ T shared_cache[MAX_THREADS]; // Shared memory for data generation

    // Default implementation for non-specialized types - fall back to elementwise
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long block_size = blockDim.x * blockDim.y * blockDim.z;

    if (row < rows && col < cols && tid < block_size) {
        long idx = row * cols + col;
        shared_cache[tid] = static_cast<T>(1.0f);
        __syncthreads();
        output[idx] = shared_cache[tid];
    }
}

// Vectorized storing using float2 (64-bit stores) - pure storing test
template<>
__global__ void matrix_store_vectorized2<float>(
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    const int MAX_THREADS = 1024;
    __shared__ float2 shared_cache[MAX_THREADS]; // Proper block size handling

    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    // Generate data in shared memory
    if (tid < MAX_THREADS) {
        shared_cache[tid] = make_float2(1.0f, 1.0f);
    }
    __syncthreads();

    // Loop-based approach with vectorized access
    long thread_id = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
                     blockIdx.x * blockDim.x * blockDim.y + tid;
    long total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    // Each thread processes 2 elements at a time
    for (long base_idx = thread_id * 2; base_idx + 1 < total_elements; base_idx += total_threads * 2) {
        if (tid < MAX_THREADS) {
            *reinterpret_cast<float2*>(&output[base_idx]) = shared_cache[tid];
        }
    }
}

// Base template for vectorized4 operations
template<typename T>
__global__ void matrix_store_vectorized4(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory for data generation

    // Default implementation for non-specialized types - fall back to elementwise
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        long idx = row * cols + col;
        shared_cache[tid] = static_cast<T>(1.0f);
        __syncthreads();
        output[idx] = shared_cache[tid];
    }
}

// Vectorized storing using float4 (128-bit stores) - pure storing test
template<>
__global__ void matrix_store_vectorized4<float>(
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ float4 shared_cache[256]; // Match block size (16x16=256 threads)

    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    // Generate data in shared memory
    if (tid < 256) {
        shared_cache[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    __syncthreads();

    // Loop-based approach with vectorized access
    long thread_id = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
                     blockIdx.x * blockDim.x * blockDim.y + tid;
    long total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    // Each thread processes 4 elements at a time
    for (long base_idx = thread_id * 4; base_idx + 3 < total_elements; base_idx += total_threads * 4) {
        if (tid < 256) {
            *reinterpret_cast<float4*>(&output[base_idx]) = shared_cache[tid];
        }
    }
}

// Base template for vectorized8 operations
template<typename T>
__global__ void matrix_store_vectorized8(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory for data generation

    // Default implementation for non-specialized types - fall back to elementwise
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        long idx = row * cols + col;
        shared_cache[tid] = static_cast<T>(1.0f);
        __syncthreads();
        output[idx] = shared_cache[tid];
    }
}

// Vectorized storing using float8 (256-bit stores as 2x float4) - pure storing test
template<>
__global__ void matrix_store_vectorized8<float>(
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    const int MAX_THREADS = 1024;
    __shared__ float8 shared_cache[MAX_THREADS]; // Proper block size handling

    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    // Generate data in shared memory
    if (tid < MAX_THREADS) {
        shared_cache[tid] = make_float8(1.0f);
    }
    __syncthreads();

    // Loop-based approach with vectorized access
    long thread_id = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
                     blockIdx.x * blockDim.x * blockDim.y + tid;
    long total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    // Each thread processes 8 elements at a time
    for (long base_idx = thread_id * 8; base_idx + 7 < total_elements; base_idx += total_threads * 8) {
        if (tid < MAX_THREADS) {
            store_float8(&output[base_idx], shared_cache[tid]);
        }
    }
}

// Row-wise coalesced access pattern - pure storing test
template<typename T>
__global__ void matrix_store_coalesced_row(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    const int MAX_THREADS = 1024;
    __shared__ T shared_cache[MAX_THREADS]; // Shared memory for data generation

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;
    long block_size = blockDim.x * blockDim.y * blockDim.z;

    // Generate data in shared memory
    if (local_tid < block_size) {
        shared_cache[local_tid] = static_cast<T>(1.0f);
    }
    __syncthreads();

    // Grid-stride loop for optimal memory coalescing
    for (long idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        output[idx] = shared_cache[local_tid];
        __syncthreads();
    }
}

// Column-wise access (non-coalesced for comparison)
template<typename T>
__global__ void matrix_store_coalesced_column(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory for data generation

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Generate data in shared memory
    shared_cache[tid] = static_cast<T>(1.0f);
    __syncthreads();

    if (row < rows && col < cols) {
        // Column-wise strided access pattern
        for (long r = row; r < rows; r += blockDim.y * gridDim.y) {
            long idx = r * cols + col;
            output[idx] = shared_cache[tid];
        }
    }
}

// Coalesced float4 storing - pure storing test with 1D grid
__global__ void matrix_store_coalesced_float4(
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ float4 shared_cache[256]; // Shared memory for data generation

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Generate data in shared memory
    shared_cache[local_tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    __syncthreads();

    // Grid-stride loop with float4 vectorized stores
    for (long idx = tid * 4; idx + 3 < total_elements; idx += (blockDim.x * gridDim.x) * 4) {
        // Store 4 floats at once using vectorized instruction
        *reinterpret_cast<float4*>(&output[idx]) = shared_cache[local_tid];
        __syncthreads();
    }
}

// Coalesced float8 storing - pure storing test with 1D grid
__global__ void matrix_store_coalesced_float8(
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    const int MAX_THREADS = 1024;
    __shared__ float8 shared_cache[MAX_THREADS]; // Shared memory for data generation

    ulong tid = blockIdx.x * blockDim.x + threadIdx.x;
    ulong local_tid = threadIdx.x;
    ulong total_elements = rows * cols;
    ulong block_size = blockDim.x * blockDim.y * blockDim.z;

    // Generate data in shared memory
    if (local_tid < block_size) {
        shared_cache[local_tid] = make_float8(1.0f);
    }
    __syncthreads();

    // Grid-stride loop with float8 vectorized stores
    for (ulong idx = tid * 8; idx + 7 < total_elements; idx += (blockDim.x * gridDim.x) * 8) {
        // Store 8 floats at once using two vectorized float4 instructions
        store_float8(&output[idx], shared_cache[local_tid]);
        __syncthreads();
    }
}

// Shared memory tiled storing - pure storing test
template<typename T, int TILE_SIZE = 32>
__global__ void matrix_store_shared_memory(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    long row = blockIdx.y * TILE_SIZE + threadIdx.y;
    long col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Generate data in shared memory
    tile[threadIdx.y][threadIdx.x] = static_cast<T>(1.0f);
    __syncthreads();

    // Cooperative storing from shared memory
    if (row < rows && col < cols) {
        output[row * cols + col] = tile[threadIdx.y][threadIdx.x];
    }
}

// CUB Device-level storing operations - optimized with cache modifiers
template<typename T>
__global__ void matrix_store_cub_device(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[1024]; // Shared memory for data generation

    // Use CUB's thread-level primitives with optimal cache modifiers
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    // Generate data in shared memory
    shared_cache[threadIdx.x] = static_cast<T>(1.0f);
    __syncthreads();

    for (long idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        // Use STORE_WB for write-back cache policy (optimal for storing)
        cub::ThreadStore<cub::STORE_WB>(&output[idx], shared_cache[threadIdx.x]);
    }
}

// CUB Device-level storing with cache-modified operations
template<typename T>
__global__ void matrix_store_cub_device_cache_modified(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[1024]; // Shared memory for data generation

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    // Generate data in shared memory
    shared_cache[threadIdx.x] = static_cast<T>(1.0f);
    __syncthreads();

    for (long idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        // Use ThreadStore with cache modifier for optimal storing
        cub::ThreadStore<cub::STORE_WB>(&output[idx], shared_cache[threadIdx.x]);
    }
}

// CUB Block-level storing operations - optimized with vectorization
template<typename T, int BLOCK_SIZE = 256>
__global__ void matrix_store_cub_block(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    // Use BLOCK_STORE_VECTORIZE for optimal performance when data is aligned
    // and ITEMS_PER_THREAD is even (4 items per thread)
    typedef cub::BlockStore<T, BLOCK_SIZE, 4, cub::BLOCK_STORE_VECTORIZE> BlockStore;

    __shared__ typename BlockStore::TempStorage store_temp_storage;

    long block_offset = blockIdx.x * BLOCK_SIZE * 4;
    long total_elements = rows * cols;

    if (block_offset < total_elements) {
        T thread_data[4];

        // Generate data to store
        #pragma unroll
        for (long i = 0; i < 4; ++i) {
            thread_data[i] = static_cast<T>(1.0f);
        }

        __syncthreads();

        // Cooperative block storing using CUB with vectorization
        long valid_items = min(BLOCK_SIZE * 4, int(total_elements - block_offset));
        BlockStore(store_temp_storage).Store(&output[block_offset], thread_data, valid_items);
    }
}

// CUB Block-level storing with warp transpose for better coalescing
template<typename T, int BLOCK_SIZE = 256>
__global__ void matrix_store_cub_block_warp_transpose(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    // Use BLOCK_STORE_WARP_TRANSPOSE for optimal memory coalescing
    // when BLOCK_THREADS is a multiple of warp size (32)
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size (32)");
    typedef cub::BlockStore<T, BLOCK_SIZE, 4, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStore;

    __shared__ typename BlockStore::TempStorage store_temp_storage;

    long block_offset = blockIdx.x * BLOCK_SIZE * 4;
    long total_elements = rows * cols;

    if (block_offset < total_elements) {
        T thread_data[4];

        // Generate data to store
        #pragma unroll
        for (long i = 0; i < 4; ++i) {
            thread_data[i] = static_cast<T>(1.0f);
        }

        __syncthreads();

        // Cooperative block storing using warp-striped pattern then transpose
        long valid_items = min(BLOCK_SIZE * 4, int(total_elements - block_offset));
        BlockStore(store_temp_storage).Store(&output[block_offset], thread_data, valid_items);
    }
}

// CUB Block-level storing with striped transpose for maximum coalescing
template<typename T, int BLOCK_SIZE = 256>
__global__ void matrix_store_cub_block_striped_transpose(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    // Use BLOCK_STORE_TRANSPOSE for maximum memory coalescing regardless of items per thread
    typedef cub::BlockStore<T, BLOCK_SIZE, 4, cub::BLOCK_STORE_TRANSPOSE> BlockStore;

    __shared__ typename BlockStore::TempStorage store_temp_storage;

    long block_offset = blockIdx.x * BLOCK_SIZE * 4;
    long total_elements = rows * cols;

    if (block_offset < total_elements) {
        T thread_data[4];

        // Generate data to store
        #pragma unroll
        for (long i = 0; i < 4; ++i) {
            thread_data[i] = static_cast<T>(1.0f);
        }

        __syncthreads();

        // Cooperative block storing using striped pattern then transpose to blocked
        long valid_items = min(BLOCK_SIZE * 4, int(total_elements - block_offset));
        BlockStore(store_temp_storage).Store(&output[block_offset], thread_data, valid_items);
    }
}

// CUB Warp-level storing operations - optimized with vectorization
template<typename T>
__global__ void matrix_store_cub_warp(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    // Use WARP_STORE_VECTORIZE for optimal performance when data is aligned
    typedef cub::WarpStore<T, 4, cub::WARP_STORE_VECTORIZE> WarpStore;

    const int WARP_SIZE = 32;
    long warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    long warp_offset = warp_id * WARP_SIZE * 4;
    long total_elements = rows * cols;

    if (warp_offset < total_elements) {
        T thread_data[4];

        // Generate data to store
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            thread_data[i] = static_cast<T>(1.0f);
        }

        // Warp-level cooperative storing with vectorization
        long valid_items = min(WARP_SIZE * 4, int(total_elements - warp_offset));
        WarpStore().Store(&output[warp_offset], thread_data, valid_items);
    }
}

// CUB Warp-level storing with striped pattern for better coalescing
template<typename T>
__global__ void matrix_store_cub_warp_striped(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    // Use WARP_STORE_STRIPED for optimal memory coalescing
    typedef cub::WarpStore<T, 4, cub::WARP_STORE_STRIPED> WarpStore;

    const int WARP_SIZE = 32;
    long warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    long warp_offset = warp_id * WARP_SIZE * 4;
    long total_elements = rows * cols;

    if (warp_offset < total_elements) {
        T thread_data[4];

        // Generate data to store
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            thread_data[i] = static_cast<T>(1.0f);
        }

        // Warp-level cooperative storing with striped pattern
        long valid_items = min(WARP_SIZE * 4, int(total_elements - warp_offset));
        WarpStore().Store(&output[warp_offset], thread_data, valid_items);
    }
}

// PTX float4 storing kernel using st.global.v4.f32
__global__ void matrix_store_ptx_float4(
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ float4 shared_cache[1024]; // Match block size (16x16=256 threads)

    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    // Generate data in shared memory
    if (tid < 1024) {
        shared_cache[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    __syncthreads();

    // Loop-based approach with PTX instructions
    long thread_id = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
                     blockIdx.x * blockDim.x * blockDim.y + tid;
    long total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    // Each thread processes 4 elements at a time
    for (long base_idx = thread_id * 4; base_idx + 3 < total_elements; base_idx += total_threads * 4) {
        if (tid < 1024) {
            // Store 4 floats using PTX st.global.v4.f32 instruction
            asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                         :: "l"((float*)&output[base_idx]), "f"(shared_cache[tid].x),
                            "f"(shared_cache[tid].y), "f"(shared_cache[tid].z), "f"(shared_cache[tid].w));
        }
    }
}

// PTX float4 storing kernel using st.global.wb.v4.f32 (write-back)
__global__ void matrix_store_ptx_float4_wb(
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ float4 shared_cache[1024]; // Match block size (16x16=256 threads)

    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    // Generate data in shared memory
    if (tid < 1024) {
        shared_cache[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    __syncthreads();

    // Loop-based approach with PTX instructions
    long thread_id = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
                     blockIdx.x * blockDim.x * blockDim.y + tid;
    long total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    // Each thread processes 4 elements at a time
    for (long base_idx = thread_id * 4; base_idx + 3 < total_elements; base_idx += total_threads * 4) {
        if (tid < 1024) {
            // Store 4 floats using PTX st.global.wb.v4.f32 instruction (write-back)
            asm volatile("st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};"
                         :: "l"((float*)&output[base_idx]), "f"(shared_cache[tid].x),
                            "f"(shared_cache[tid].y), "f"(shared_cache[tid].z), "f"(shared_cache[tid].w));
        }
    }
}

// =============================================================================
// BENCHMARK FUNCTION
// =============================================================================

// Host function to benchmark matrix storing
std::vector<float> benchmark_matrix_storing(
    torch::Tensor output,
    int method,
    int iterations = 100
) {
    auto rows = output.size(0);
    auto cols = output.size(1);

    // Ensure tensor is on GPU and contiguous
    output = output.cuda().contiguous();

    // Calculate grid and block dimensions using common utilities
    dim3 block_size = calculate_optimal_block_size(method);
    dim3 grid_size = calculate_optimal_grid_size(rows, cols, method, block_size);

    // Warmup runs
    for (int i = 0; i < 10; ++i) {
        switch(method) {
            case ELEMENTWISE:
                matrix_store_elementwise<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT2:
                matrix_store_vectorized2<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT4:
                matrix_store_vectorized4<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT8:
                matrix_store_vectorized8<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_ROW:
                matrix_store_coalesced_row<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_COLUMN:
                matrix_store_coalesced_column<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_FLOAT4:
                matrix_store_coalesced_float4<<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_FLOAT8:
                matrix_store_coalesced_float8<<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case SHARED_MEMORY_TILED:
                matrix_store_shared_memory<float, 32><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_DEVICE_LOAD:
                matrix_store_cub_device<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_DEVICE_CACHE_MODIFIED:
                matrix_store_cub_device_cache_modified<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_BLOCK_LOAD:
                matrix_store_cub_block<float, 256><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_BLOCK_WARP_TRANSPOSE:
                matrix_store_cub_block_warp_transpose<float, 256><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_BLOCK_STRIPED_TRANSPOSE:
                matrix_store_cub_block_striped_transpose<float, 256><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_WARP_LOAD:
                matrix_store_cub_warp<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_WARP_STRIPED:
                matrix_store_cub_warp_striped<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case PTX_FLOAT4:
                matrix_store_ptx_float4<<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case PTX_FLOAT4_NC:
                matrix_store_ptx_float4_wb<<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            default:
                matrix_store_elementwise<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
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

    for (int i = 0; i < iterations; ++i) {
        cudaEventRecord(start);

        switch(method) {
            case ELEMENTWISE:
                matrix_store_elementwise<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT2:
                matrix_store_vectorized2<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT4:
                matrix_store_vectorized4<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT8:
                matrix_store_vectorized8<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_ROW:
                matrix_store_coalesced_row<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_COLUMN:
                matrix_store_coalesced_column<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_FLOAT4:
                matrix_store_coalesced_float4<<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_FLOAT8:
                matrix_store_coalesced_float8<<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case SHARED_MEMORY_TILED:
                matrix_store_shared_memory<float, 32><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_DEVICE_LOAD:
                matrix_store_cub_device<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_DEVICE_CACHE_MODIFIED:
                matrix_store_cub_device_cache_modified<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_BLOCK_LOAD:
                matrix_store_cub_block<float, 256><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_BLOCK_WARP_TRANSPOSE:
                matrix_store_cub_block_warp_transpose<float, 256><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_BLOCK_STRIPED_TRANSPOSE:
                matrix_store_cub_block_striped_transpose<float, 256><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_WARP_LOAD:
                matrix_store_cub_warp<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_WARP_STRIPED:
                matrix_store_cub_warp_striped<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case PTX_FLOAT4:
                matrix_store_ptx_float4<<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case PTX_FLOAT4_NC:
                matrix_store_ptx_float4_wb<<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            default:
                matrix_store_elementwise<float><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
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

    return times;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("benchmark_matrix_storing", &benchmark_matrix_storing,
          "Benchmark matrix storing operations",
          py::arg("output"), py::arg("method"), py::arg("iterations") = 100);
}
