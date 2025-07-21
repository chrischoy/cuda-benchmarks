#!/usr/bin/env python3
"""
CuTe-based Memory Loading Benchmarks

This module implements various memory loading patterns using NVIDIA's CuTe DSL (Domain Specific Language).
CuTe provides a high-level abstraction for memory operations that can generate highly optimized CUDA kernels.

Features:
- Element-wise loading with vectorization
- Coalesced memory access patterns
- Tiled loading with shared memory staging
- Predicated loading for irregular shapes
- Performance comparison with traditional CUDA approaches

Requirements:
- CUTLASS with CuTe support
- PyTorch with CUDA
- Compute capability 7.0+ (Volta/Turing/Ampere/Ada)
"""

import argparse
import time
from typing import Type, List, Tuple
import torch
import numpy as np

try:
    import sys
    import os

    # Add nvidia_cutlass_dsl package path
    venv_path = os.path.join(
        os.path.dirname(sys.executable),
        "..",
        "lib",
        "python{}.{}".format(*sys.version_info[:2]),
        "site-packages",
        "nvidia_cutlass_dsl",
        "python_packages",
    )
    if os.path.exists(venv_path):
        sys.path.insert(0, venv_path)

    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False
    print("Install cutlass using `pip install nvidia-cutlass nvidia-cutlass-dsl`")


# Simple load kernel - element-wise loading
@cute.kernel
def cute_elementwise_load_kernel(
    gA: cute.Tensor,
    gOutput: cute.Tensor,
    cA: cute.Tensor,  # coordinate tensor for predication
    shape: cute.Shape,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
):
    """
    Element-wise loading kernel using CuTe.
    Loads data from global memory and stores the same values back (memory bandwidth test).
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Slice tensors for current thread block
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkOutput = gOutput[blk_coord]
    blkCrd = cA[blk_coord]

    # Create tiled copy operations
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gOutput.element_type
    )

    tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
    tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

    # Get per-thread slices
    thr_copy_load = tiled_copy_load.get_slice(tidx)
    thr_copy_store = tiled_copy_store.get_slice(tidx)

    thrA = thr_copy_load.partition_S(blkA)
    thrOutput = thr_copy_store.partition_S(blkOutput)
    thrCrd = thr_copy_load.partition_S(blkCrd)

    # Allocate register fragments
    frgA = cute.make_fragment_like(thrA)
    frgOutput = cute.make_fragment_like(thrOutput)
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)

    # Generate predicate mask for out-of-bounds protection
    for i in range(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    # Load data from global memory to registers
    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)

    # Simply copy loaded data back (measures load bandwidth)
    frgOutput.store(frgA.load())

    # Store result back to global memory
    cute.copy(copy_atom_store, frgOutput, thrOutput, pred=frgPred)


# Vectorized load kernel - optimized for coalesced access
@cute.kernel
def cute_vectorized_load_kernel(
    gA: cute.Tensor,
    gOutput: cute.Tensor,
    cA: cute.Tensor,
    shape: cute.Shape,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
):
    """
    Vectorized loading kernel optimized for memory bandwidth.
    Uses wider vector instructions (float4/double2) for better coalescing.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkOutput = gOutput[blk_coord]
    blkCrd = cA[blk_coord]

    # Use universal copy operations for better memory throughput
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gOutput.element_type
    )

    tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
    tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

    thr_copy_load = tiled_copy_load.get_slice(tidx)
    thr_copy_store = tiled_copy_store.get_slice(tidx)

    thrA = thr_copy_load.partition_S(blkA)
    thrOutput = thr_copy_store.partition_S(blkOutput)
    thrCrd = thr_copy_load.partition_S(blkCrd)

    frgA = cute.make_fragment_like(thrA)
    frgOutput = cute.make_fragment_like(thrOutput)
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)

    for i in range(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    # Vectorized load operation
    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    frgOutput.store(frgA.load())
    cute.copy(copy_atom_store, frgOutput, thrOutput, pred=frgPred)


# Tiled load kernel with shared memory staging
@cute.kernel
def cute_tiled_load_kernel(
    gA: cute.Tensor,
    gOutput: cute.Tensor,
    cA: cute.Tensor,
    shape: cute.Shape,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
):
    """
    Tiled loading kernel using shared memory for staging.
    Demonstrates cooperative loading within thread blocks.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Shared memory tile (adjust size based on thread block)
    smem_tile = cute.make_tensor(
        cute.make_ptr(cute.Float32, cute.AddressSpace.smem), (16, 16)
    )

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkOutput = gOutput[blk_coord]
    blkCrd = cA[blk_coord]

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gOutput.element_type
    )

    tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
    tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

    thr_copy_load = tiled_copy_load.get_slice(tidx)
    thr_copy_store = tiled_copy_store.get_slice(tidx)

    thrA = thr_copy_load.partition_S(blkA)
    thrOutput = thr_copy_store.partition_S(blkOutput)
    thrCrd = thr_copy_load.partition_S(blkCrd)

    # Partition shared memory tile
    thrSmem = thr_copy_load.partition_S(smem_tile)

    frgA = cute.make_fragment_like(thrA)
    frgSmem = cute.make_fragment_like(thrSmem)
    frgOutput = cute.make_fragment_like(thrOutput)
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)

    for i in range(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    # Load from global memory to registers
    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)

    # Stage through shared memory
    cute.copy(copy_atom_store, frgA, thrSmem)
    cute.arch.cp_async_wait_group(0)
    cute.copy(copy_atom_load, thrSmem, frgSmem)

    frgOutput.store(frgSmem.load())
    cute.copy(copy_atom_store, frgOutput, thrOutput, pred=frgPred)


# JIT compilation functions
@cute.jit
def cute_elementwise_load(mA, mOutput, copy_bits: cutlass.Constexpr = 128):
    """JIT-compiled element-wise load operation."""
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    # Thread and value layouts for optimal memory access
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # Tile input tensors
    gA = cute.zipped_divide(mA, tiler_mn)
    gOutput = cute.zipped_divide(mOutput, tiler_mn)

    # Create coordinate tensor for predication
    idA = cute.make_identity_tensor(mA.shape)
    cA = cute.zipped_divide(idA, tiler=tiler_mn)

    cute_elementwise_load_kernel(gA, gOutput, cA, mA.shape, tv_layout, tiler_mn).launch(
        grid=[cute.size(gA, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def cute_vectorized_load(mA, mOutput, copy_bits: cutlass.Constexpr = 128):
    """JIT-compiled vectorized load operation."""
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout(
        (8, 32), order=(1, 0)
    )  # More threads for vectorization
    val_layout = cute.make_ordered_layout((2, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gOutput = cute.zipped_divide(mOutput, tiler_mn)

    idA = cute.make_identity_tensor(mA.shape)
    cA = cute.zipped_divide(idA, tiler=tiler_mn)

    cute_vectorized_load_kernel(gA, gOutput, cA, mA.shape, tv_layout, tiler_mn).launch(
        grid=[cute.size(gA, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def cute_tiled_load(mA, mOutput, copy_bits: cutlass.Constexpr = 128):
    """JIT-compiled tiled load operation with shared memory."""
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gOutput = cute.zipped_divide(mOutput, tiler_mn)

    idA = cute.make_identity_tensor(mA.shape)
    cA = cute.zipped_divide(idA, tiler=tiler_mn)

    cute_tiled_load_kernel(gA, gOutput, cA, mA.shape, tv_layout, tiler_mn).launch(
        grid=[cute.size(gA, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def run_cute_load_benchmark(
    M: int,
    N: int,
    dtype: Type[cutlass.Numeric] = cutlass.Float32,
    method: str = "elementwise",
    iterations: int = 100,
    warmup_iterations: int = 10,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Run CuTe-based loading benchmark.

    Args:
        M, N: Matrix dimensions
        dtype: Data type (Float16, Float32, etc.)
        method: Loading method ("elementwise", "vectorized", "tiled")
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        verbose: Print detailed information

    Returns:
        Tuple of (average_time_ms, bandwidth_gb_s)
    """
    if not CUTE_AVAILABLE:
        raise RuntimeError("CuTe not available. Install CUTLASS with Python bindings.")

    if verbose:
        print(f"\nRunning CuTe {method} load benchmark:")
        print(f"Matrix dimensions: {M}x{N}")
        print(f"Data type: {dtype}")
        print(f"Method: {method}")

    torch_dtype = cutlass_torch.dtype(dtype)

    # Create input and output tensors
    if dtype.is_integer:
        input_tensor = torch.randint(
            0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype
        )
    else:
        input_tensor = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)

    output_tensor = torch.zeros_like(input_tensor)

    # Convert to CuTe tensors
    cute_input = from_dlpack(input_tensor).mark_layout_dynamic()
    cute_output = from_dlpack(output_tensor).mark_layout_dynamic()

    # Select and compile the appropriate kernel
    try:
        if method == "elementwise":
            compiled_func = cute.compile(cute_elementwise_load, cute_input, cute_output)
        elif method == "vectorized":
            compiled_func = cute.compile(cute_vectorized_load, cute_input, cute_output)
        elif method == "tiled":
            compiled_func = cute.compile(cute_tiled_load, cute_input, cute_output)
        else:
            raise ValueError(f"Unknown method: {method}")

        if verbose:
            print("Kernel compiled successfully")
    except Exception as e:
        import traceback

        error_msg = f"Kernel compilation failed for method '{method}'"
        error_msg += f"\nError type: {type(e).__name__}"
        error_msg += f"\nError message: {str(e)}"
        if hasattr(e, "__cause__") and e.__cause__:
            error_msg += (
                f"\nCaused by: {type(e.__cause__).__name__}: {str(e.__cause__)}"
            )
        if verbose:
            error_msg += "\nFull traceback:"
            print(error_msg)
            traceback.print_exc()
        else:
            print(error_msg)
        raise RuntimeError(error_msg) from e

    # Warmup runs
    for _ in range(warmup_iterations):
        compiled_func(cute_input, cute_output)

    torch.cuda.synchronize()

    # Benchmark runs
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        compiled_func(cute_input, cute_output)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate statistics
    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)

    # Calculate bandwidth (bytes read + bytes written)
    total_bytes = 2 * M * N * (dtype.width // 8)  # Read + write
    bandwidth_gb_s = (total_bytes / (avg_time_ms / 1000)) / 1e9

    if verbose:
        print(f"Average time: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms")
        print(f"Memory bandwidth: {bandwidth_gb_s:.2f} GB/s")
        print(f"Matrix size: {total_bytes / 1e6:.2f} MB")

    # Verify correctness (output should equal input for these load kernels)
    if method in ["elementwise", "vectorized", "tiled"]:
        try:
            torch.testing.assert_close(
                input_tensor, output_tensor, rtol=1e-5, atol=1e-6
            )
            if verbose:
                print("✓ Correctness verification passed")
        except Exception as e:
            if verbose:
                print(f"✗ Correctness verification failed: {e}")

    return avg_time_ms, bandwidth_gb_s


def run_cute_load_comparison(
    matrix_sizes: List[Tuple[int, int]] = None,
    dtype: Type[cutlass.Numeric] = cutlass.Float32,
    iterations: int = 100,
    verbose: bool = True,
):
    """
    Compare different CuTe loading methods across various matrix sizes.
    """
    if not CUTE_AVAILABLE:
        print("CuTe not available. Skipping comparison.")
        return

    if matrix_sizes is None:
        matrix_sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

    methods = ["elementwise", "vectorized", "tiled"]

    print(f"\n{'=' * 80}")
    print(f"CuTe Loading Methods Comparison - {dtype}")
    print(f"{'=' * 80}")
    print(f"{'Size':<12} {'Method':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<16}")
    print(f"{'-' * 80}")

    results = {}

    for M, N in matrix_sizes:
        results[(M, N)] = {}

        for method in methods:
            try:
                avg_time, bandwidth = run_cute_load_benchmark(
                    M, N, dtype, method, iterations, warmup_iterations=10, verbose=False
                )
                results[(M, N)][method] = (avg_time, bandwidth)

                print(f"{M}x{N:<8} {method:<12} {avg_time:<12.4f} {bandwidth:<16.2f}")

            except Exception as e:
                import traceback

                error_details = f"{type(e).__name__}: {str(e)}"
                if hasattr(e, "__cause__") and e.__cause__:
                    error_details += (
                        f"\nCaused by: {type(e.__cause__).__name__}: {str(e.__cause__)}"
                    )
                if verbose:
                    print(f"{M}x{N:<8} {method:<12} ERROR: {error_details}")
                    print("Full traceback:")
                    traceback.print_exc()
                else:
                    print(f"{M}x{N:<8} {method:<12} ERROR: {error_details}")
                results[(M, N)][method] = (float("inf"), 0.0)

    # Find best method for each size
    print(f"\n{'=' * 50}")
    print("Best Method per Matrix Size:")
    print(f"{'=' * 50}")

    for (M, N), method_results in results.items():
        best_method = max(
            method_results.keys(), key=lambda m: method_results[m][1]
        )  # Max bandwidth
        best_bandwidth = method_results[best_method][1]
        print(f"{M}x{N}: {best_method} ({best_bandwidth:.2f} GB/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CuTe-based memory loading benchmarks")
    parser.add_argument("--M", default=1024, type=int, help="Matrix rows")
    parser.add_argument("--N", default=1024, type=int, help="Matrix columns")
    parser.add_argument(
        "--method",
        default="comparison",
        choices=["elementwise", "vectorized", "tiled", "comparison"],
        help="Loading method to benchmark",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "int32"],
        help="Data type",
    )
    parser.add_argument(
        "--iterations", default=100, type=int, help="Benchmark iterations"
    )
    parser.add_argument("--warmup", default=10, type=int, help="Warmup iterations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not CUTE_AVAILABLE:
        print("Error: CuTe not available. Please install CUTLASS with Python bindings.")
        exit(1)

    if not torch.cuda.is_available():
        print("Error: CUDA not available.")
        exit(1)

    # Map string dtype to CuTe type
    dtype_map = {
        "float16": cutlass.Float16,
        "float32": cutlass.Float32,
        "int32": cutlass.Int32,
    }
    dtype = dtype_map[args.dtype]

    if args.method == "comparison":
        # Run comparison across multiple sizes
        matrix_sizes = [
            (args.M // 4, args.N // 4),
            (args.M // 2, args.N // 2),
            (args.M, args.N),
            (args.M * 2, args.N * 2),
        ]
        run_cute_load_comparison(matrix_sizes, dtype, args.iterations, args.verbose)
    else:
        # Run single method benchmark
        avg_time, bandwidth = run_cute_load_benchmark(
            args.M,
            args.N,
            dtype,
            args.method,
            args.iterations,
            args.warmup,
            args.verbose,
        )
        print("\nFinal Results:")
        print(f"Average time: {avg_time:.4f} ms")
        print(f"Memory bandwidth: {bandwidth:.2f} GB/s")

    print("\nCuTe loading benchmark completed!")
