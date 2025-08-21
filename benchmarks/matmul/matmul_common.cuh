#pragma once

#include <cuda_runtime.h>

namespace cuda_benchmark {
namespace matmul {

enum MatmulMethod {
    NAIVE_F32 = 0,
    WMMA_F16_ACC_F32 = 1,
    F32_PTX_V4 = 2,
    WMMA_F16_ACC_F32_DB_AMPERE = 3,
    WMMA_DB_AMPERE_GENERIC = 4,
    WMMA_DB_AMPERE_GENERIC_STORE = 5
};

__host__ inline dim3 calculate_block_size(int method) {
    switch (method) {
        case NAIVE_F32:
            return dim3(128, 1, 1);
        case WMMA_F16_ACC_F32:
            return dim3(32, 1, 1); // one warp per 16x16 tile
        case WMMA_F16_ACC_F32_DB_AMPERE:
        case WMMA_DB_AMPERE_GENERIC:
        case WMMA_DB_AMPERE_GENERIC_STORE:
            return dim3(32, 1, 1); // one warp per 16x16 tile
        case F32_PTX_V4:
            return dim3(128, 1, 1);
        default:
            return dim3(128, 1, 1);
    }
}

__host__ inline dim3 calculate_grid_size(int P, int N, int method, dim3 block) {
    switch (method) {
        case NAIVE_F32: {
            int grid_x = (N + block.x - 1) / block.x;
            int grid_y = P;
            if (grid_y > 65535) grid_y = 65535;
            return dim3(grid_x, grid_y, 1);
        }
        case WMMA_F16_ACC_F32: {
            int grid_x = (N + 16 - 1) / 16;
            int grid_y = (P + 16 - 1) / 16;
            // Cap grid_y moderately to avoid extremely large grids; kernel loops over tiles
            if (grid_y > 4096) grid_y = 4096;
            return dim3(grid_x, grid_y, 1);
        }
        case WMMA_F16_ACC_F32_DB_AMPERE:
        case WMMA_DB_AMPERE_GENERIC:
        case WMMA_DB_AMPERE_GENERIC_STORE: {
            int grid_x = (N + 16 - 1) / 16;
            int grid_y = (P + 16 - 1) / 16;
            if (grid_y > 4096) grid_y = 4096;
            return dim3(grid_x, grid_y, 1);
        }
        case F32_PTX_V4: {
            int elems_per_thread = 4; // each thread computes 4 columns
            int grid_x = (N + (block.x * elems_per_thread) - 1) / (block.x * elems_per_thread);
            int grid_y = P;
            if (grid_y > 65535) grid_y = 65535;
            return dim3(grid_x, grid_y, 1);
        }
        default: {
            int grid_x = (N + block.x - 1) / block.x;
            int grid_y = P;
            if (grid_y > 65535) grid_y = 65535;
            return dim3(grid_x, grid_y, 1);
        }
    }
}

} // namespace matmul
} // namespace cuda_benchmark
