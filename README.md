# CUDA Benchmarking For Primitive Operations

## Introduction

This project is a comprehensive benchmarking tool for CUDA primitive operations such as load, store, arithmetic, scan, reduce, gather-scatter, and more. The library provides systematic performance comparisons across different GPU architectures, input sizes, and data types (float16, float32, float64, etc.).

## Project Goals

- **Comprehensive Operation Coverage**: Benchmark all fundamental CUDA operations and primitives
- **Multi-Architecture Support**: Compare performance across different NVIDIA GPU architectures (Ampere, Ada Lovelace, Hopper, etc.)
- **Variable Input Sizes**: Test operations with different data sizes to understand scaling behavior
- **Multiple Data Types**: Support various floating-point precisions (FP16, FP32, FP64) and integer types
- **Reproducible Results**: Provide consistent and reliable benchmarking methodology

## Submodules

Located in 3rdparty/ directory.

### [CUDA Core Compute Libraries (CCCL)](https://github.com/NVIDIA/cccl)

CCCL is NVIDIA's collection of core CUDA libraries that provide fundamental parallel algorithms and data structures. It includes:

- **Thrust**: High-level parallel algorithms and data structures
- **CUB**: Low-level CUDA primitives for block and warp-level operations
- **libcudacxx**: CUDA C++ standard library implementation

This submodule provides the foundational primitives and algorithms that we benchmark, including reductions, scans, sorts, and other parallel operations.

### [CUTLASS](https://github.com/NVIDIA/cutlass)

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's collection of CUDA C++ template abstractions for implementing high-performance matrix operations. It includes:

- **GEMM Operations**: Optimized general matrix multiplication routines
- **Convolution Kernels**: High-performance convolution implementations
- **Template Abstractions**: Flexible building blocks for custom linear algebra operations
- **Multi-Precision Support**: FP16, FP32, FP64, and mixed-precision operations

This submodule enables benchmarking of matrix operations and serves as a reference for optimized CUDA implementations.

## Getting Started

### Prerequisites

- NVIDIA GPU with compute capability 7.0 or higher
- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- C++17 compatible compiler

### Building

```bash
# Clone the repository with submodules
git clone --recursive <repository-url>
cd cuda-benchmark

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)
```

## Benchmark Categories

The library organizes benchmarks into the following categories:

1. **Memory Operations**: Load, store, memory bandwidth tests
2. **Arithmetic Operations**: Addition, multiplication, division, transcendental functions
3. **Reduction Operations**: Sum, min/max, custom reductions
4. **Scan Operations**: Prefix sum, exclusive/inclusive scans
5. **Sort Operations**: Radix sort, merge sort comparisons
6. **Matrix Operations**: GEMM, convolutions, tensor operations
7. **Synchronization**: Warp-level and block-level synchronization primitives
