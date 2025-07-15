#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <type_traits>

namespace cuda_benchmark {
namespace memory {

// Loading method enumeration
enum LoadingMethod {
    ELEMENTWISE = 0,           // Basic element-wise loading
    VECTORIZED_FLOAT2 = 1,     // 64-bit vectorized loads
    VECTORIZED_FLOAT4 = 2,     // 128-bit vectorized loads
    VECTORIZED_FLOAT8 = 3,     // 256-bit vectorized loads (2x float4)
    COALESCED_ROW = 4,         // Row-wise coalesced access
    COALESCED_COLUMN = 5,      // Column-wise access (non-coalesced)
    COALESCED_FLOAT4 = 6,      // Coalesced float4 vectorized (1D grid)
    COALESCED_FLOAT8 = 7,      // Coalesced float8 vectorized (1D grid)
    SHARED_MEMORY_TILED = 8,   // Shared memory with tiling
    CUB_DEVICE_LOAD = 9,       // CUB device-wide load operations
    CUB_BLOCK_LOAD = 10,       // CUB block-level load operations
    CUB_WARP_LOAD = 11,        // CUB warp-level load operations
    TEXTURE_MEMORY = 12        // Texture memory access pattern
};

// Launch configuration utilities
__host__ inline dim3 calculate_optimal_block_size(int loading_method) {
    switch(loading_method) {
        case ELEMENTWISE:
            return dim3(16, 16);
        case VECTORIZED_FLOAT2:
            return dim3(16, 16);
        case VECTORIZED_FLOAT4:
            return dim3(16, 16);
        case VECTORIZED_FLOAT8:
            return dim3(16, 16);
        case COALESCED_ROW:
            return dim3(256);
        case COALESCED_COLUMN:
            return dim3(16, 16);
        case COALESCED_FLOAT4:
            return dim3(256);
        case COALESCED_FLOAT8:
            return dim3(256);
        case SHARED_MEMORY_TILED:
            return dim3(32, 32);
        case CUB_DEVICE_LOAD:
            return dim3(256);
        case CUB_BLOCK_LOAD:
            return dim3(256);
        case CUB_WARP_LOAD:
            return dim3(256);
        case TEXTURE_MEMORY:
            return dim3(16, 16);
        default:
            return dim3(16, 16);
    }
}

__host__ inline dim3 calculate_optimal_grid_size(size_t rows, size_t cols, int loading_method, dim3 block_size) {
    const int MAX_GRID_SIZE_X = 2147483647; // 2^31-1 CUDA maximum X grid dimension
    const int MAX_GRID_SIZE_Y = 65535; // CUDA maximum Y grid dimension

    switch(loading_method) {
        case ELEMENTWISE:
        case COALESCED_COLUMN:
        case SHARED_MEMORY_TILED:
        case TEXTURE_MEMORY: {
            int grid_x = min(MAX_GRID_SIZE_X, (int)((cols + block_size.x - 1) / block_size.x));
            int grid_y = min(MAX_GRID_SIZE_Y, (int)((rows + block_size.y - 1) / block_size.y));
            return dim3(grid_x, grid_y);
        }
        case VECTORIZED_FLOAT2: {
            // Each thread processes 2 elements in X direction
            int grid_x = max(1, min(MAX_GRID_SIZE_X, (int)((cols/2 + block_size.x - 1) / block_size.x)));
            int grid_y = max(1, min(MAX_GRID_SIZE_Y, (int)((rows + block_size.y - 1) / block_size.y)));
            return dim3(grid_x, grid_y);
        }
        case VECTORIZED_FLOAT4: {
            // Each thread processes 4 elements in X direction
            int grid_x = max(1, min(MAX_GRID_SIZE_X, (int)((cols/4 + block_size.x - 1) / block_size.x)));
            int grid_y = max(1, min(MAX_GRID_SIZE_Y, (int)((rows + block_size.y - 1) / block_size.y)));
            return dim3(grid_x, grid_y);
        }
        case VECTORIZED_FLOAT8: {
            // Each thread processes 8 elements in X direction
            int grid_x = max(1, min(MAX_GRID_SIZE_X, (int)((cols/8 + block_size.x - 1) / block_size.x)));
            int grid_y = max(1, min(MAX_GRID_SIZE_Y, (int)((rows + block_size.y - 1) / block_size.y)));
            return dim3(grid_x, grid_y);
        }
        case COALESCED_ROW:
        case CUB_DEVICE_LOAD:
        case CUB_BLOCK_LOAD:
        case CUB_WARP_LOAD: {
            int grid_x = min(MAX_GRID_SIZE_X, (int)((rows * cols + block_size.x - 1) / block_size.x));
            return dim3(grid_x);
        }
        case COALESCED_FLOAT4: {
            // Each thread processes 4 elements
            size_t elements_per_thread = (rows * cols + 3) / 4;  // Ceiling division by 4
            int grid_x = min(MAX_GRID_SIZE_X, (int)((elements_per_thread + block_size.x - 1) / block_size.x));
            return dim3(grid_x);
        }
        case COALESCED_FLOAT8: {
            // Each thread processes 8 elements
            size_t elements_per_thread = (rows * cols + 7) / 8;  // Ceiling division by 8
            int grid_x = min(MAX_GRID_SIZE_X, (int)((elements_per_thread + block_size.x - 1) / block_size.x));
            return dim3(grid_x);
        }
        default: {
            int grid_x = min(MAX_GRID_SIZE_Y, (int)((cols + block_size.x - 1) / block_size.x));
            int grid_y = min(MAX_GRID_SIZE_Y, (int)((rows + block_size.y - 1) / block_size.y));
            return dim3(grid_x, grid_y);
        }
    }
}

// Bandwidth calculation utilities
__host__ inline double calculate_bandwidth_gb_s(size_t rows, size_t cols, size_t data_type_size,
                                               double time_ms, int operations = 2) {
    size_t total_elements = rows * cols;
    size_t bytes_transferred = total_elements * data_type_size * operations;
    double time_s = time_ms / 1000.0;
    return (double)bytes_transferred / (1024.0 * 1024.0 * 1024.0) / time_s;
}

// Method name utilities
__host__ inline const char* get_method_name(int method) {
    switch(method) {
        case ELEMENTWISE: return "Element-wise";
        case VECTORIZED_FLOAT2: return "Float2 vectorized";
        case VECTORIZED_FLOAT4: return "Float4 vectorized";
        case VECTORIZED_FLOAT8: return "Float8 vectorized";
        case COALESCED_ROW: return "Coalesced row";
        case COALESCED_COLUMN: return "Coalesced column";
        case COALESCED_FLOAT4: return "Coalesced float4";
        case COALESCED_FLOAT8: return "Coalesced float8";
        case SHARED_MEMORY_TILED: return "Shared memory tiled";
        case CUB_DEVICE_LOAD: return "CUB device load";
        case CUB_BLOCK_LOAD: return "CUB block load";
        case CUB_WARP_LOAD: return "CUB warp load";
        case TEXTURE_MEMORY: return "Texture memory";
        default: return "Unknown";
    }
}

// Helper macros for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            return; \
        } \
    } while(0)

#define CUDA_CHECK_RETURN(call, retval) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            return retval; \
        } \
    } while(0)

} // namespace memory
} // namespace cuda_benchmark
