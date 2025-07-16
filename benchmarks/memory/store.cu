#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "matrix_loading_common.cuh"
#include "vector_types.cuh"

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

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long block_size = blockDim.x * blockDim.y * blockDim.z;

    if (row < rows && col < cols && tid < block_size) {
        long idx = row * cols + col;
        // Generate data in shared memory first
        shared_cache[tid] = static_cast<T>(1.0f);
        __syncthreads();

        // Pure storing test - write 1s to global memory
        output[idx] = shared_cache[tid];
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

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long block_size = blockDim.x * blockDim.y * blockDim.z;

    if (row < rows && col + 1 < cols && tid < block_size) {
        long idx = row * cols + col;

        // Generate data in shared memory
        shared_cache[tid] = make_float2(1.0f, 1.0f);
        __syncthreads();

        // Store 2 floats at once using vectorized instruction
        *reinterpret_cast<float2*>(&output[idx]) = shared_cache[tid];
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

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col + 3 < cols) {
        long idx = row * cols + col;

        // Generate data in shared memory
        shared_cache[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        __syncthreads();

        // Store 4 floats at once using vectorized instruction
        *reinterpret_cast<float4*>(&output[idx]) = shared_cache[tid];
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

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;
    long block_size = blockDim.x * blockDim.y * blockDim.z;

    if (row < rows && col + 7 < cols && tid < block_size) {
        long idx = row * cols + col;

        // Generate data in shared memory
        shared_cache[tid] = make_float8(1.0f);
        __syncthreads();

        // Store 8 floats at once using two vectorized float4 instructions
        store_float8(&output[idx], shared_cache[tid]);
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

// CUB Device-level storing operations
template<typename T>
__global__ void matrix_store_cub_device(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    const int MAX_THREADS = 1024;
    __shared__ T shared_cache[MAX_THREADS]; // Shared memory for data generation

    // Use CUB's thread-level primitives for efficient storing
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x; // Use LOCAL thread ID for shared memory
    long total_elements = rows * cols;
    long block_size = blockDim.x * blockDim.y * blockDim.z;

    // Generate data in shared memory using LOCAL thread ID
    if (local_tid < block_size) {
        shared_cache[local_tid] = static_cast<T>(1.0f);
    }
    __syncthreads();

    // Use the local thread's data for all global memory stores
    for (long idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        cub::ThreadStore<cub::STORE_WB>(&output[idx], shared_cache[local_tid]);
    }
}

// CUB Block-level storing operations
template<typename T, int BLOCK_SIZE = 256>
__global__ void matrix_store_cub_block(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
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

        // Cooperative block storing using CUB
        long valid_items = min(BLOCK_SIZE * 4, int(total_elements - block_offset));
        BlockStore(store_temp_storage).Store(&output[block_offset], thread_data, valid_items);
    }
}

// CUB Warp-level storing operations
template<typename T>
__global__ void matrix_store_cub_warp(
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    typedef cub::WarpStore<T, 4, cub::WARP_STORE_VECTORIZE> WarpStore;

    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_offset = warp_id * WARP_SIZE * 4;
    long total_elements = rows * cols;

    if (warp_offset < total_elements) {
        T thread_data[4];

        // Generate data to store
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            thread_data[i] = static_cast<T>(1.0f);
        }

        // Warp-level cooperative storing
        long valid_items = min(WARP_SIZE * 4, int(total_elements - warp_offset));
        WarpStore().Store(&output[warp_offset], thread_data, valid_items);
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
            case CUB_BLOCK_LOAD:
                matrix_store_cub_block<float, 256><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_WARP_LOAD:
                matrix_store_cub_warp<float><<<grid_size, block_size>>>(
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
            case CUB_BLOCK_LOAD:
                matrix_store_cub_block<float, 256><<<grid_size, block_size>>>(
                    output.data_ptr<float>(), rows, cols);
                break;
            case CUB_WARP_LOAD:
                matrix_store_cub_warp<float><<<grid_size, block_size>>>(
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
