#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "matrix_loading_common.cuh"
#include "vector_types.cuh"

using namespace cuda_benchmark::memory;

// =============================================================================
// MATRIX LOADING KERNELS
// =============================================================================

// Element-wise loading kernel - pure loading test
template<typename T>
__global__ void matrix_load_elementwise(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory cache

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols && tid < 256) {
        long idx = row * cols + col;
        // Pure loading test - store in shared memory
        T value = input[idx];
        shared_cache[tid] = value * static_cast<T>(1.001f); // Minimal computation to prevent optimization
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Base template for vectorized operations
template<typename T>
__global__ void matrix_load_vectorized2(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory cache

    // Default implementation for non-specialized types - fall back to elementwise
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        long idx = row * cols + col;
        shared_cache[tid] = input[idx] * T(1.001f);
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Vectorized loading using float2 (64-bit loads) - pure loading test
template<>
__global__ void matrix_load_vectorized2<float>(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ float2 shared_cache[256]; // Match block size (16x16=256 threads)

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col + 1 < cols) {
        long idx = row * cols + col;

        // Load 2 floats at once using vectorized instruction
        float2 values = __ldg(reinterpret_cast<const float2*>(&input[idx]));

        // Minimal processing and store in shared memory
        values.x *= 1.001f;
        values.y *= 1.001f;
        shared_cache[tid] = values;
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Base template for vectorized4 operations
template<typename T>
__global__ void matrix_load_vectorized4(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory cache

    // Default implementation for non-specialized types - fall back to elementwise
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        long idx = row * cols + col;
        shared_cache[tid] = input[idx] * T(1.001f);
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Vectorized loading using float4 (128-bit loads) - pure loading test
template<>
__global__ void matrix_load_vectorized4<float>(
    const float* __restrict__ input,
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

        // Load 4 floats at once using vectorized instruction
        float4 values = __ldg(reinterpret_cast<const float4*>(&input[idx]));

        // Minimal processing and store in shared memory
        values.x *= 1.001f;
        values.y *= 1.001f;
        values.z *= 1.001f;
        values.w *= 1.001f;
        shared_cache[tid] = values;
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Base template for vectorized8 operations
template<typename T>
__global__ void matrix_load_vectorized8(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory cache

    // Default implementation for non-specialized types - fall back to elementwise
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        long idx = row * cols + col;
        shared_cache[tid] = input[idx] * T(1.001f);
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Vectorized loading using float8 (256-bit loads as 2x float4) - pure loading test
template<>
__global__ void matrix_load_vectorized8<float>(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ float8 shared_cache[256]; // Match block size (16x16=256 threads)

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col + 7 < cols) {
        long idx = row * cols + col;

        // Load 8 floats at once using two vectorized float4 instructions
        float8 values = load_float8(&input[idx]);

        // Minimal processing on all 8 elements
        values.lo.x *= 1.001f; values.lo.y *= 1.001f; values.lo.z *= 1.001f; values.lo.w *= 1.001f;
        values.hi.x *= 1.001f; values.hi.y *= 1.001f; values.hi.z *= 1.001f; values.hi.w *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[tid] = values;
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Row-wise coalesced access pattern - pure loading test
template<typename T>
__global__ void matrix_load_coalesced_row(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Grid-stride loop for optimal memory coalescing
    for (long idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        T value = __ldg(&input[idx]);
        // Fair comparison: ALL threads store their loaded data in shared memory
        shared_cache[local_tid] = value * static_cast<T>(1.001);
        __syncthreads();
    }
}

// Column-wise access (non-coalesced for comparison)
template<typename T>
__global__ void matrix_load_coalesced_column(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory cache

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Column-wise strided access pattern
        for (long r = row; r < rows; r += blockDim.y * gridDim.y) {
            long idx = r * cols + col;
            T value = __ldg(&input[idx]);
            shared_cache[tid] = value * static_cast<T>(1.001);
        }
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Coalesced float4 loading - pure loading test with 1D grid
__global__ void matrix_load_coalesced_float4(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ float4 shared_cache[256]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_elements = rows * cols;

    // Grid-stride loop with float4 vectorized loads
    for (long idx = tid * 4; idx + 3 < total_elements; idx += (blockDim.x * gridDim.x) * 4) {
        // Load 4 floats at once using vectorized instruction
        float4 values = __ldg(reinterpret_cast<const float4*>(&input[idx]));

        // Minimal processing to prevent optimization
        values.x *= 1.001f; values.y *= 1.001f; values.z *= 1.001f; values.w *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[local_tid] = values;
        __syncthreads();
    }
}

// Coalesced float8 loading - pure loading test with 1D grid
__global__ void matrix_load_coalesced_float8(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ float8 shared_cache[256]; // Shared memory cache

    ulong tid = blockIdx.x * blockDim.x + threadIdx.x;
    ulong local_tid = threadIdx.x;
    ulong total_elements = rows * cols;

    // Grid-stride loop with float8 vectorized loads
    for (ulong idx = tid * 8; idx + 7 < total_elements; idx += (blockDim.x * gridDim.x) * 8) {
        // Load 8 floats at once using two vectorized float4 instructions
        float8 values = load_float8(&input[idx]);

        // Minimal processing on all 8 elements
        values.lo.x *= 1.001f; values.lo.y *= 1.001f; values.lo.z *= 1.001f; values.lo.w *= 1.001f;
        values.hi.x *= 1.001f; values.hi.y *= 1.001f; values.hi.z *= 1.001f; values.hi.w *= 1.001f;

        // Store in shared memory for fair comparison
        shared_cache[local_tid] = values;
        __syncthreads();
    }
}

// Shared memory tiled loading - pure loading test
template<typename T, int TILE_SIZE = 32>
__global__ void matrix_load_shared_memory(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    long row = blockIdx.y * TILE_SIZE + threadIdx.y;
    long col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Cooperative loading into shared memory
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = __ldg(&input[row * cols + col]);
    } else {
        tile[threadIdx.y][threadIdx.x] = static_cast<T>(0);
    }

    __syncthreads();

    // Process data from shared memory (pure loading test - no write back)
    T value = tile[threadIdx.y][threadIdx.x] * static_cast<T>(1.001);

    // Store processed value back in tile to ensure computation isn't optimized away
    tile[threadIdx.y][threadIdx.x] = value;

    __syncthreads();
}

// CUB Device-level loading operations
template<typename T>
__global__ void matrix_load_cub_device(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ T shared_cache[256]; // Shared memory cache

    // Use CUB's thread-level primitives for efficient loading
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total_elements = rows * cols;

    for (long idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        T value = cub::ThreadLoad<cub::LOAD_LDG>(&input[idx]);
        shared_cache[tid] = value * static_cast<T>(1.001);
        // cub::ThreadStore<cub::STORE_WB>(&output[idx], value * static_cast<T>(1.001));
    }
}

// CUB Block-level loading operations
template<typename T, int BLOCK_SIZE = 256>
__global__ void matrix_load_cub_block(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    typedef cub::BlockLoad<T, BLOCK_SIZE, 4, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
    typedef cub::BlockStore<T, BLOCK_SIZE, 4, cub::BLOCK_STORE_VECTORIZE> BlockStore;

    __shared__ typename BlockLoad::TempStorage load_temp_storage;
    __shared__ typename BlockStore::TempStorage store_temp_storage;

    long block_offset = blockIdx.x * BLOCK_SIZE * 4;
    long total_elements = rows * cols;

    if (block_offset < total_elements) {
        T thread_data[4];

        // Cooperative block loading using CUB
        long valid_items = min(BLOCK_SIZE * 4, int(total_elements - block_offset));
        BlockLoad(load_temp_storage).Load(&input[block_offset], thread_data, valid_items);

        __syncthreads();

        // Process loaded data
        #pragma unroll
        for (long i = 0; i < 4; ++i) {
            thread_data[i] *= static_cast<T>(1.001);
        }

        // Cooperative block storing using CUB
        // BlockStore(store_temp_storage).Store(&output[block_offset], thread_data, valid_items);
    }
}

// CUB Warp-level loading operations
template<typename T>
__global__ void matrix_load_cub_warp(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t rows,
    size_t cols
) {
    typedef cub::WarpLoad<T, 4, cub::WARP_LOAD_VECTORIZE> WarpLoad;
    typedef cub::WarpStore<T, 4, cub::WARP_STORE_VECTORIZE> WarpStore;

    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_offset = warp_id * WARP_SIZE * 4;
    long total_elements = rows * cols;

    if (warp_offset < total_elements) {
        T thread_data[4];

        // Warp-level cooperative loading
        long valid_items = min(WARP_SIZE * 4, int(total_elements - warp_offset));
        WarpLoad().Load(&input[warp_offset], thread_data, valid_items);

        // Process loaded data
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            thread_data[i] *= static_cast<T>(1.001);
        }

        // Warp-level cooperative storing
        // WarpStore().Store(&output[warp_offset], thread_data, valid_items);
    }
}

// Texture memory loading (for float only)
__global__ void matrix_load_texture_float(
    cudaTextureObject_t tex_obj,
    float* __restrict__ output,
    size_t rows,
    size_t cols
) {
    __shared__ float shared_cache[256]; // Shared memory cache

    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        long idx = row * cols + col;
        float value = tex1Dfetch<float>(tex_obj, idx);
        shared_cache[tid] = value * 1.001f;
    }
}

// PTX float4 loading kernel using ld.global.v4.f32
__global__ void matrix_load_ptx_float4(
    const float* __restrict__ input,
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

        // Load 4 floats using PTX ld.global.v4.f32 instruction
        float4 values;
        asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(values.x), "=f"(values.y), "=f"(values.z), "=f"(values.w)
                     : "l"((const float*)&input[idx]));

        // Minimal processing and store in shared memory
        values.x *= 1.001f;
        values.y *= 1.001f;
        values.z *= 1.001f;
        values.w *= 1.001f;
        shared_cache[tid] = values;
    }

    __syncthreads(); // Ensure shared memory write completes
}

// PTX float4 loading kernel using ld.global.nc.v4.f32 (non-cached)
__global__ void matrix_load_ptx_float4_nc(
    const float* __restrict__ input,
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

        // Load 4 floats using PTX ld.global.nc.v4.f32 instruction (non-cached)
        float4 values;
        asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(values.x), "=f"(values.y), "=f"(values.z), "=f"(values.w)
                     : "l"((const float*)&input[idx]));

        // Minimal processing and store in shared memory
        values.x *= 1.001f;
        values.y *= 1.001f;
        values.z *= 1.001f;
        values.w *= 1.001f;
        shared_cache[tid] = values;
    }

    __syncthreads(); // Ensure shared memory write completes
}

// =============================================================================
// BENCHMARK FUNCTION
// =============================================================================

// Host function to benchmark matrix loading
std::vector<float> benchmark_matrix_loading(
    torch::Tensor input,
    int method,
    int iterations = 100
) {
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
    for (int i = 0; i < 10; ++i) {
        switch(method) {
            case ELEMENTWISE:
                matrix_load_elementwise<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT2:
                matrix_load_vectorized2<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT4:
                matrix_load_vectorized4<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT8:
                matrix_load_vectorized8<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_ROW:
                matrix_load_coalesced_row<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_COLUMN:
                matrix_load_coalesced_column<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_FLOAT4:
                matrix_load_coalesced_float4<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_FLOAT8:
                matrix_load_coalesced_float8<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case SHARED_MEMORY_TILED:
                matrix_load_shared_memory<float, 32><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case CUB_DEVICE_LOAD:
                matrix_load_cub_device<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case CUB_BLOCK_LOAD:
                matrix_load_cub_block<float, 256><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case CUB_WARP_LOAD:
                matrix_load_cub_warp<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case PTX_FLOAT4:
                matrix_load_ptx_float4<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case PTX_FLOAT4_NC:
                matrix_load_ptx_float4_nc<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            default:
                matrix_load_elementwise<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
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
                matrix_load_elementwise<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT2:
                matrix_load_vectorized2<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT4:
                matrix_load_vectorized4<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case VECTORIZED_FLOAT8:
                matrix_load_vectorized8<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_ROW:
                matrix_load_coalesced_row<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_COLUMN:
                matrix_load_coalesced_column<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_FLOAT4:
                matrix_load_coalesced_float4<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case COALESCED_FLOAT8:
                matrix_load_coalesced_float8<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case SHARED_MEMORY_TILED:
                matrix_load_shared_memory<float, 32><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case CUB_DEVICE_LOAD:
                matrix_load_cub_device<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case CUB_BLOCK_LOAD:
                matrix_load_cub_block<float, 256><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case CUB_WARP_LOAD:
                matrix_load_cub_warp<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case PTX_FLOAT4:
                matrix_load_ptx_float4<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            case PTX_FLOAT4_NC:
                matrix_load_ptx_float4_nc<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
                break;
            default:
                matrix_load_elementwise<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
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
    m.def("benchmark_matrix_loading", &benchmark_matrix_loading,
          "Benchmark matrix loading operations",
          py::arg("input"), py::arg("method"), py::arg("iterations") = 100);
}
