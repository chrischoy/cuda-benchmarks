#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <type_traits>

namespace cuda_benchmark {
namespace gather {

// Loading method enumeration for gather operations
enum LoadingMethod {
    ELEMENTWISE = 0,           // Basic element-wise gathering
    VECTORIZED_FLOAT2 = 1,     // 64-bit vectorized gathers
    VECTORIZED_FLOAT4 = 2,     // 128-bit vectorized gathers
    VECTORIZED_FLOAT8 = 3,     // 256-bit vectorized gathers (2x float4)
    COALESCED_ROW = 4,         // Row-wise coalesced gather
    COALESCED_COLUMN = 5,      // Column-wise gather (non-coalesced)
    COALESCED_FLOAT4 = 6,      // Coalesced float4 vectorized (1D grid)
    COALESCED_FLOAT8 = 7,      // Coalesced float8 vectorized (1D grid)
    SHARED_MEMORY_TILED = 8,   // Shared memory with tiling
    CUB_DEVICE_LOAD = 9,       // CUB device-wide gather operations
    CUB_BLOCK_LOAD = 10,       // CUB block-level gather operations
    CUB_WARP_LOAD = 11,        // CUB warp-level gather operations
    CUB_DEVICE_CACHE_MODIFIED = 12, // CUB device-wide gather with cache-modified iterators
    CUB_BLOCK_WARP_TRANSPOSE = 13,  // CUB block-level gather with warp transpose
    CUB_BLOCK_STRIPED_TRANSPOSE = 14, // CUB block-level gather with striped transpose
    CUB_WARP_STRIPED = 15,     // CUB warp-level gather with striped pattern
    TEXTURE_MEMORY = 16,       // Texture memory gather pattern
    PTX_FLOAT4 = 17,           // PTX ld.global.v4.f32 instruction for gather
    PTX_FLOAT4_NC = 18         // PTX ld.global.nc.v4.f32 (non-cached) instruction for gather
};

// Launch configuration utilities optimized for gather operations
__host__ inline dim3 calculate_optimal_block_size(int loading_method) {
    switch(loading_method) {
        case ELEMENTWISE:
            return dim3(32, 32);  // Use 32x32 to reduce grid Y dimension for large matrices
        case VECTORIZED_FLOAT2:
            return dim3(16, 16);  // Use 2D block to reduce grid Y dimension for large matrices
        case VECTORIZED_FLOAT4:
            return dim3(16, 16);  // Use 2D block to reduce grid Y dimension for large matrices
        case VECTORIZED_FLOAT8:
            return dim3(16, 16);  // Use 2D block to reduce grid Y dimension for large matrices
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
        case CUB_DEVICE_CACHE_MODIFIED:
            return dim3(256);
        case CUB_BLOCK_LOAD:
        case CUB_BLOCK_WARP_TRANSPOSE:
        case CUB_BLOCK_STRIPED_TRANSPOSE:
            return dim3(256);
        case CUB_WARP_LOAD:
        case CUB_WARP_STRIPED:
            return dim3(256);
        case TEXTURE_MEMORY:
            return dim3(16, 16);
        case PTX_FLOAT4:
        case PTX_FLOAT4_NC:
            return dim3(16, 16);
        default:
            return dim3(16, 16);
    }
}

// Grid size calculation specifically for gather operations
__host__ inline dim3 calculate_gather_grid_size(size_t n_gather, size_t cols, int loading_method, dim3 block_size) {
    const int MAX_GRID_SIZE_X = 2147483647; // 2^31-1 CUDA maximum X grid dimension
    const int MAX_GRID_SIZE_Y = 65535; // CUDA maximum Y grid dimension

    switch(loading_method) {
        case ELEMENTWISE:
        case COALESCED_COLUMN:
        case SHARED_MEMORY_TILED: {
            int grid_x = min(MAX_GRID_SIZE_X, (int)((cols + block_size.x - 1) / block_size.x));
            int grid_y = min(MAX_GRID_SIZE_Y, (int)((n_gather + block_size.y - 1) / block_size.y));
            return dim3(grid_x, grid_y);
        }
        case VECTORIZED_FLOAT2: {
            // Each thread processes 2 elements in X direction, use ceiling division
            int threads_needed = (cols + 1) / 2;
            int grid_x = max(1, min(MAX_GRID_SIZE_X, (int)((threads_needed + block_size.x - 1) / block_size.x)));
            int grid_y = max(1, min(MAX_GRID_SIZE_Y, (int)((n_gather + block_size.y - 1) / block_size.y)));
            return dim3(grid_x, grid_y);
        }
        case VECTORIZED_FLOAT4: {
            // Each thread processes 4 elements in X direction, use ceiling division
            int threads_needed = (cols + 3) / 4;
            int grid_x = max(1, min(MAX_GRID_SIZE_X, (int)((threads_needed + block_size.x - 1) / block_size.x)));
            int grid_y = max(1, min(MAX_GRID_SIZE_Y, (int)((n_gather + block_size.y - 1) / block_size.y)));
            return dim3(grid_x, grid_y);
        }
        case VECTORIZED_FLOAT8: {
            // Each thread processes 8 elements in X direction, use ceiling division
            int threads_needed = (cols + 7) / 8;
            int grid_x = max(1, min(MAX_GRID_SIZE_X, (int)((threads_needed + block_size.x - 1) / block_size.x)));
            int grid_y = max(1, min(MAX_GRID_SIZE_Y, (int)((n_gather + block_size.y - 1) / block_size.y)));
            return dim3(grid_x, grid_y);
        }
        case COALESCED_ROW:
        case CUB_DEVICE_LOAD:
        case CUB_BLOCK_LOAD:
        case CUB_WARP_LOAD: {
            int grid_x = min(MAX_GRID_SIZE_X, (int)((n_gather * cols + block_size.x - 1) / block_size.x));
            return dim3(grid_x);
        }
        case COALESCED_FLOAT4: {
            // Each thread processes 4 elements
            size_t elements_per_thread = (n_gather * cols + 3) / 4;  // Ceiling division by 4
            int grid_x = min(MAX_GRID_SIZE_X, (int)((elements_per_thread + block_size.x - 1) / block_size.x));
            return dim3(grid_x);
        }
        case COALESCED_FLOAT8: {
            // Each thread processes 8 elements
            size_t elements_per_thread = (n_gather * cols + 7) / 8;  // Ceiling division by 8
            int grid_x = min(MAX_GRID_SIZE_X, (int)((elements_per_thread + block_size.x - 1) / block_size.x));
            return dim3(grid_x);
        }
        default: {
            int grid_x = min(MAX_GRID_SIZE_Y, (int)((cols + block_size.x - 1) / block_size.x));
            int grid_y = min(MAX_GRID_SIZE_Y, (int)((n_gather + block_size.y - 1) / block_size.y));
            return dim3(grid_x, grid_y);
        }
    }
}

// Bandwidth calculation utilities for gather operations
__host__ inline double calculate_gather_bandwidth_gb_s(size_t n_gather, size_t cols, size_t data_type_size,
                                                      double time_ms, int operations = 2) {
    size_t total_elements = n_gather * cols;
    size_t bytes_transferred = total_elements * data_type_size * operations;  // read + write
    double time_s = time_ms / 1000.0;
    return (double)bytes_transferred / (1024.0 * 1024.0 * 1024.0) / time_s;
}

// Method name utilities for gather operations
__host__ inline const char* get_gather_method_name(int method) {
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
        case CUB_DEVICE_LOAD: return "CUB device gather";
        case CUB_DEVICE_CACHE_MODIFIED: return "CUB device cache-modified";
        case CUB_BLOCK_LOAD: return "CUB block gather";
        case CUB_BLOCK_WARP_TRANSPOSE: return "CUB block warp-transpose";
        case CUB_BLOCK_STRIPED_TRANSPOSE: return "CUB block striped-transpose";
        case CUB_WARP_LOAD: return "CUB warp gather";
        case CUB_WARP_STRIPED: return "CUB warp striped";
        case TEXTURE_MEMORY: return "Texture memory";
        case PTX_FLOAT4: return "PTX float4 ld.global";
        case PTX_FLOAT4_NC: return "PTX float4 ld.global.nc";
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

} // namespace gather
} // namespace cuda_benchmark
