#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "matrix_loading.cuh"
#include "vector_types.cuh"

using namespace cuda_benchmark::gather;

// =============================================================================
// MATRIX GATHERING KERNELS
// =============================================================================

// Element-wise gathering kernel - pure gathering test
template<typename T>
__global__ void matrix_gather_elementwise(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    long gather_row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;

    if (gather_row < n_gather && col < cols) {
        long source_row = indices[gather_row];
        if (source_row < rows) {
            long input_idx = source_row * cols + col;
            long output_idx = gather_row * cols + col;

            // Pure gathering test with minimal computation to prevent optimization
            T value = input[input_idx];
            output[output_idx] = value * static_cast<T>(1.001f);
        }
    }
}

// Base template for vectorized gathering operations
template<typename T>
__global__ void matrix_gather_vectorized2(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    // Default implementation for non-specialized types - fall back to elementwise
    long gather_row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;

    if (gather_row < n_gather && col < cols) {
        long source_row = indices[gather_row];
        if (source_row < rows) {
            long input_idx = source_row * cols + col;
            long output_idx = gather_row * cols + col;
            output[output_idx] = input[input_idx] * T(1.001f);
        }
    }
}

// Vectorized gathering using float2 (64-bit loads) - pure gathering test
template<>
__global__ void matrix_gather_vectorized2<float>(
    const float* __restrict__ input,
    const long* __restrict__ indices,
    float* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    long gather_row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (gather_row < n_gather && col < cols) {
        long source_row = indices[gather_row];
        if (source_row < rows) {
            long input_idx = source_row * cols + col;
            long output_idx = gather_row * cols + col;

            if (col + 1 < cols) {
                // Load 2 floats at once using vectorized instruction
                float2 values = __ldg(reinterpret_cast<const float2*>(&input[input_idx]));

                // Minimal processing
                values.x *= 1.001f;
                values.y *= 1.001f;

                // Store to output
                *reinterpret_cast<float2*>(&output[output_idx]) = values;
            } else {
                // Handle the last element if cols is odd
                float value = input[input_idx];
                output[output_idx] = value * 1.001f;
            }
        }
    }
}

// Base template for vectorized4 gathering operations
template<typename T>
__global__ void matrix_gather_vectorized4(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    // Default implementation for non-specialized types - fall back to elementwise
    long gather_row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;

    if (gather_row < n_gather && col < cols) {
        long source_row = indices[gather_row];
        if (source_row < rows) {
            long input_idx = source_row * cols + col;
            long output_idx = gather_row * cols + col;
            output[output_idx] = input[input_idx] * T(1.001f);
        }
    }
}

// Vectorized gathering using float4 (128-bit loads) - pure gathering test
template<>
__global__ void matrix_gather_vectorized4<float>(
    const float* __restrict__ input,
    const long* __restrict__ indices,
    float* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    long gather_row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (gather_row < n_gather && col < cols) {
        long source_row = indices[gather_row];
        if (source_row < rows) {
            long input_idx = source_row * cols + col;
            long output_idx = gather_row * cols + col;

            if (col + 3 < cols) {
                // Load 4 floats at once using vectorized instruction
                float4 values = __ldg(reinterpret_cast<const float4*>(&input[input_idx]));

                // Minimal processing
                values.x *= 1.001f;
                values.y *= 1.001f;
                values.z *= 1.001f;
                values.w *= 1.001f;

                // Store to output
                *reinterpret_cast<float4*>(&output[output_idx]) = values;
            } else {
                // Handle remaining elements if cols is not divisible by 4
                for (int i = 0; i < 4 && col + i < cols; ++i) {
                    float value = input[input_idx + i];
                    output[output_idx + i] = value * 1.001f;
                }
            }
        }
    }
}

// Base template for vectorized8 gathering operations
template<typename T>
__global__ void matrix_gather_vectorized8(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    // Default implementation for non-specialized types - fall back to elementwise
    long gather_row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;

    if (gather_row < n_gather && col < cols) {
        long source_row = indices[gather_row];
        if (source_row < rows) {
            long input_idx = source_row * cols + col;
            long output_idx = gather_row * cols + col;
            output[output_idx] = input[input_idx] * T(1.001f);
        }
    }
}

// Vectorized gathering using float8 (256-bit loads as 2x float4) - pure gathering test
template<>
__global__ void matrix_gather_vectorized8<float>(
    const float* __restrict__ input,
    const long* __restrict__ indices,
    float* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    long gather_row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (gather_row < n_gather && col < cols) {
        long source_row = indices[gather_row];
        if (source_row < rows) {
            long input_idx = source_row * cols + col;
            long output_idx = gather_row * cols + col;

            if (col + 7 < cols) {
                // Load 8 floats at once using two vectorized float4 instructions
                float8 values = load_float8(&input[input_idx]);

                // Minimal processing on all 8 elements
                values.lo.x *= 1.001f; values.lo.y *= 1.001f; values.lo.z *= 1.001f; values.lo.w *= 1.001f;
                values.hi.x *= 1.001f; values.hi.y *= 1.001f; values.hi.z *= 1.001f; values.hi.w *= 1.001f;

                // Store to output
                store_float8(&output[output_idx], values);
            } else {
                // Handle remaining elements if cols is not divisible by 8
                for (int i = 0; i < 8 && col + i < cols; ++i) {
                    float value = input[input_idx + i];
                    output[output_idx + i] = value * 1.001f;
                }
            }
        }
    }
}

// Row-wise coalesced gathering pattern - pure gathering test
template<typename T>
__global__ void matrix_gather_coalesced_row(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    __shared__ T shared_cache[1024]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_gather_elements = n_gather * cols;

    // Grid-stride loop for optimal memory coalescing
    for (long idx = tid; idx < total_gather_elements; idx += blockDim.x * gridDim.x) {
        long gather_row = idx / cols;
        long col = idx % cols;

        if (gather_row < n_gather) {
            long source_row = indices[gather_row];
            if (source_row < rows) {
                long input_idx = source_row * cols + col;
                T value = __ldg(&input[input_idx]);
                // Fair comparison: ALL threads store their loaded data in shared memory
                shared_cache[local_tid] = value * static_cast<T>(1.001);
                output[idx] = shared_cache[local_tid];
                __syncthreads();
            }
        }
    }
}

// Column-wise gathering (non-coalesced for comparison)
template<typename T>
__global__ void matrix_gather_coalesced_column(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    __shared__ T shared_cache[1024]; // Shared memory cache

    long gather_row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (gather_row < n_gather && col < cols) {
        long source_row = indices[gather_row];
        if (source_row < rows) {
            // Column-wise strided access pattern
            for (long g = gather_row; g < n_gather; g += blockDim.y * gridDim.y) {
                long input_idx = indices[g] * cols + col;
                long output_idx = g * cols + col;
                T value = __ldg(&input[input_idx]);
                shared_cache[tid] = value * static_cast<T>(1.001);
                output[output_idx] = shared_cache[tid];
            }
        }
    }

    __syncthreads(); // Ensure shared memory write completes
}

// Coalesced float4 gathering - pure gathering test with 1D grid
__global__ void matrix_gather_coalesced_float4(
    const float* __restrict__ input,
    const long* __restrict__ indices,
    float* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    __shared__ float4 shared_cache[1024]; // Shared memory cache

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long local_tid = threadIdx.x;
    long total_gather_elements = n_gather * cols;

    // Grid-stride loop with float4 vectorized loads
    for (long idx = tid * 4; idx + 3 < total_gather_elements; idx += (blockDim.x * gridDim.x) * 4) {
        long gather_row = idx / cols;
        long col = idx % cols;

        if (gather_row < n_gather && col + 3 < cols) {
            long source_row = indices[gather_row];
            if (source_row < rows) {
                long input_idx = source_row * cols + col;

                // Load 4 floats at once using vectorized instruction
                float4 values = __ldg(reinterpret_cast<const float4*>(&input[input_idx]));

                // Minimal processing to prevent optimization
                values.x *= 1.001f; values.y *= 1.001f; values.z *= 1.001f; values.w *= 1.001f;

                // Store in shared memory for fair comparison
                shared_cache[local_tid] = values;

                // Store to output
                *reinterpret_cast<float4*>(&output[idx]) = values;
                __syncthreads();
            }
        }
    }
}

// Coalesced float8 gathering - pure gathering test with 1D grid
__global__ void matrix_gather_coalesced_float8(
    const float* __restrict__ input,
    const long* __restrict__ indices,
    float* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    __shared__ float8 shared_cache[1024]; // Shared memory cache

    ulong tid = blockIdx.x * blockDim.x + threadIdx.x;
    ulong local_tid = threadIdx.x;
    ulong total_gather_elements = n_gather * cols;

    // Grid-stride loop with float8 vectorized loads
    for (ulong idx = tid * 8; idx + 7 < total_gather_elements; idx += (blockDim.x * gridDim.x) * 8) {
        ulong gather_row = idx / cols;
        ulong col = idx % cols;

        if (gather_row < n_gather && col + 7 < cols) {
            ulong source_row = indices[gather_row];
            if (source_row < rows) {
                ulong input_idx = source_row * cols + col;

                // Load 8 floats at once using two vectorized float4 instructions
                float8 values = load_float8(&input[input_idx]);

                // Minimal processing on all 8 elements
                values.lo.x *= 1.001f; values.lo.y *= 1.001f; values.lo.z *= 1.001f; values.lo.w *= 1.001f;
                values.hi.x *= 1.001f; values.hi.y *= 1.001f; values.hi.z *= 1.001f; values.hi.w *= 1.001f;

                // Store in shared memory for fair comparison
                shared_cache[local_tid] = values;

                // Store to output
                store_float8(&output[idx], values);
                __syncthreads();
            }
        }
    }
}

// Shared memory tiled gathering - pure gathering test
template<typename T, int TILE_SIZE = 32>
__global__ void matrix_gather_shared_memory(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    __shared__ T tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    long gather_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    long col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Cooperative loading into shared memory
    if (gather_row < n_gather && col < cols) {
        long source_row = indices[gather_row];
        if (source_row < rows) {
            long input_idx = source_row * cols + col;
            tile[threadIdx.y][threadIdx.x] = __ldg(&input[input_idx]);
        } else {
            tile[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
    } else {
        tile[threadIdx.y][threadIdx.x] = static_cast<T>(0);
    }

    __syncthreads();

    // Process data from shared memory (pure gathering test - write back)
    T value = tile[threadIdx.y][threadIdx.x] * static_cast<T>(1.001);

    // Store processed value back in tile to ensure computation isn't optimized away
    tile[threadIdx.y][threadIdx.x] = value;

    __syncthreads();

    // Write back to output
    if (gather_row < n_gather && col < cols) {
        long output_idx = gather_row * cols + col;
        output[output_idx] = tile[threadIdx.y][threadIdx.x];
    }
}

// CUB Device-level gathering operations
template<typename T>
__global__ void matrix_gather_cub_device(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    __shared__ T shared_cache[1024]; // Shared memory cache

    // Use CUB's thread-level primitives for efficient loading
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total_gather_elements = n_gather * cols;

    for (long idx = tid; idx < total_gather_elements; idx += blockDim.x * gridDim.x) {
        long gather_row = idx / cols;
        long col = idx % cols;

        if (gather_row < n_gather) {
            long source_row = indices[gather_row];
            if (source_row < rows) {
                long input_idx = source_row * cols + col;
                T value = cub::ThreadLoad<cub::LOAD_LDG>(&input[input_idx]);
                shared_cache[tid] = value * static_cast<T>(1.001);
                output[idx] = shared_cache[tid];
            }
        }
    }
}

// CUB Block-level gathering operations
template<typename T, int BLOCK_SIZE = 256>
__global__ void matrix_gather_cub_block(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    typedef cub::BlockLoad<T, BLOCK_SIZE, 4, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
    typedef cub::BlockStore<T, BLOCK_SIZE, 4, cub::BLOCK_STORE_VECTORIZE> BlockStore;

    __shared__ typename BlockLoad::TempStorage load_temp_storage;
    __shared__ typename BlockStore::TempStorage store_temp_storage;

    long block_offset = blockIdx.x * BLOCK_SIZE * 4;
    long total_gather_elements = n_gather * cols;

    if (block_offset < total_gather_elements) {
        T thread_data[4];
        T input_data[4];

        // Calculate gather indices for this block
        for (int i = 0; i < 4; ++i) {
            long idx = block_offset + threadIdx.x * 4 + i;
            if (idx < total_gather_elements) {
                long gather_row = idx / cols;
                long col = idx % cols;
                if (gather_row < n_gather) {
                    long source_row = indices[gather_row];
                    if (source_row < rows) {
                        long input_idx = source_row * cols + col;
                        input_data[i] = input[input_idx];
                    } else {
                        input_data[i] = static_cast<T>(0);
                    }
                } else {
                    input_data[i] = static_cast<T>(0);
                }
            } else {
                input_data[i] = static_cast<T>(0);
            }
        }

        // Process loaded data
        #pragma unroll
        for (long i = 0; i < 4; ++i) {
            thread_data[i] = input_data[i] * static_cast<T>(1.001);
        }

        // Cooperative block storing using CUB
        long valid_items = min(BLOCK_SIZE * 4, int(total_gather_elements - block_offset));
        BlockStore(store_temp_storage).Store(&output[block_offset], thread_data, valid_items);
    }
}

// CUB Warp-level gathering operations
template<typename T>
__global__ void matrix_gather_cub_warp(
    const T* __restrict__ input,
    const long* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t n_gather
) {
    typedef cub::WarpLoad<T, 4, cub::WARP_LOAD_VECTORIZE> WarpLoad;
    typedef cub::WarpStore<T, 4, cub::WARP_STORE_VECTORIZE> WarpStore;

    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_offset = warp_id * WARP_SIZE * 4;
    long total_gather_elements = n_gather * cols;

    if (warp_offset < total_gather_elements) {
        T thread_data[4];
        T input_data[4];

        // Calculate gather indices for this warp
        for (int i = 0; i < 4; ++i) {
            long idx = warp_offset + lane_id * 4 + i;
            if (idx < total_gather_elements) {
                long gather_row = idx / cols;
                long col = idx % cols;
                if (gather_row < n_gather) {
                    long source_row = indices[gather_row];
                    if (source_row < rows) {
                        long input_idx = source_row * cols + col;
                        input_data[i] = input[input_idx];
                    } else {
                        input_data[i] = static_cast<T>(0);
                    }
                } else {
                    input_data[i] = static_cast<T>(0);
                }
            } else {
                input_data[i] = static_cast<T>(0);
            }
        }

        // Process loaded data
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            thread_data[i] = input_data[i] * static_cast<T>(1.001);
        }

        // Warp-level cooperative storing
        long valid_items = min(WARP_SIZE * 4, int(total_gather_elements - warp_offset));
        WarpStore().Store(&output[warp_offset], thread_data, valid_items);
    }
}



// =============================================================================
// BENCHMARK FUNCTION
// =============================================================================

// Host function to benchmark matrix gathering
std::tuple<std::vector<float>, torch::Tensor> benchmark_matrix_gathering(
    torch::Tensor input,
    torch::Tensor indices,
    int method,
    int iterations = 100,
    bool return_output = false
) {
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto n_gather = indices.size(0);

    auto output = torch::zeros({n_gather, cols}, input.options());

    // Ensure tensors are on GPU and contiguous
    input = input.cuda().contiguous();
    indices = indices.cuda().contiguous();
    output = output.cuda().contiguous();

    // Calculate grid and block dimensions using common utilities
    dim3 block_size = calculate_optimal_block_size(method);
    dim3 grid_size = calculate_gather_grid_size(n_gather, cols, method, block_size);

    // Warmup runs
    for (int i = 0; i < 10; ++i) {
        switch(method) {
            case ELEMENTWISE:
                matrix_gather_elementwise<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case VECTORIZED_FLOAT2:
                matrix_gather_vectorized2<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case VECTORIZED_FLOAT4:
                matrix_gather_vectorized4<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case VECTORIZED_FLOAT8:
                matrix_gather_vectorized8<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case COALESCED_ROW:
                matrix_gather_coalesced_row<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case COALESCED_COLUMN:
                matrix_gather_coalesced_column<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case COALESCED_FLOAT4:
                matrix_gather_coalesced_float4<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case COALESCED_FLOAT8:
                matrix_gather_coalesced_float8<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case SHARED_MEMORY_TILED:
                matrix_gather_shared_memory<float, 32><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case CUB_DEVICE_LOAD:
                matrix_gather_cub_device<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case CUB_BLOCK_LOAD:
                matrix_gather_cub_block<float, 256><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case CUB_WARP_LOAD:
                matrix_gather_cub_warp<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            default:
                matrix_gather_elementwise<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
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
                matrix_gather_elementwise<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case VECTORIZED_FLOAT2:
                matrix_gather_vectorized2<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case VECTORIZED_FLOAT4:
                matrix_gather_vectorized4<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case VECTORIZED_FLOAT8:
                matrix_gather_vectorized8<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case COALESCED_ROW:
                matrix_gather_coalesced_row<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case COALESCED_COLUMN:
                matrix_gather_coalesced_column<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case COALESCED_FLOAT4:
                matrix_gather_coalesced_float4<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case COALESCED_FLOAT8:
                matrix_gather_coalesced_float8<<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case SHARED_MEMORY_TILED:
                matrix_gather_shared_memory<float, 32><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case CUB_DEVICE_LOAD:
                matrix_gather_cub_device<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case CUB_BLOCK_LOAD:
                matrix_gather_cub_block<float, 256><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            case CUB_WARP_LOAD:
                matrix_gather_cub_warp<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
                break;
            default:
                matrix_gather_elementwise<float><<<grid_size, block_size>>>(
                    input.data_ptr<float>(), indices.data_ptr<long>(), output.data_ptr<float>(), rows, cols, n_gather);
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

    if (return_output) {
        return std::make_tuple(times, output);
    } else {
        return std::make_tuple(times, torch::Tensor());
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("benchmark_matrix_gathering", &benchmark_matrix_gathering,
          "Benchmark matrix gathering operations",
          py::arg("input"), py::arg("indices"), py::arg("method"), py::arg("iterations") = 100, py::arg("return_output") = false);
}
