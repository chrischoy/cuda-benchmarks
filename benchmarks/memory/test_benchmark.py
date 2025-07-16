#!/usr/bin/env python3

import torch
import numpy as np
import argparse
import matrix_load_benchmark
import matrix_store_benchmark


def test_basic_functionality(verbose=True):
    """Test basic matrix loading and storing functionality"""
    print("Testing basic matrix loading functionality...")

    # Create test matrix
    input_matrix = torch.randn(512, 512, device="cuda", dtype=torch.float32)

    try:
        # Test basic element-wise loading (method 0)
        times = matrix_load_benchmark.benchmark_matrix_loading(
            input_matrix, 0, iterations=10
        )

        # Check if we got reasonable results
        if len(times) == 10 and all(t > 0 for t in times):
            if verbose:
                print("✓ Matrix loading benchmark test passed!")
        else:
            if verbose:
                print("✗ Matrix loading benchmark test failed!")
            return False
    except Exception as e:
        if verbose:
            print(f"✗ Matrix loading benchmark test failed with error: {e}")
        return False

    if verbose:
        print("Testing basic matrix storing functionality...")

    # Create output matrix
    output_matrix = torch.zeros(512, 512, device="cuda", dtype=torch.float32)

    try:
        # Test basic element-wise storing (method 0)
        times = matrix_store_benchmark.benchmark_matrix_storing(
            output_matrix, 0, iterations=10
        )

        # Check if we got reasonable results
        if len(times) == 10 and all(t > 0 for t in times):
            if verbose:
                print("✓ Matrix storing benchmark test passed!")

            # Verify that the output matrix contains 1s
            expected_value = 1.0
            if torch.allclose(
                output_matrix,
                torch.ones_like(output_matrix) * expected_value,
                atol=1e-5,
            ):
                if verbose:
                    print("✓ Matrix storing correctness test passed!")
                return True
            else:
                if verbose:
                    print("✗ Matrix storing correctness test failed!")
                return False
        else:
            if verbose:
                print("✗ Matrix storing benchmark test failed!")
            return False
    except Exception as e:
        if verbose:
            print(f"✗ Matrix storing benchmark test failed with error: {e}")
        return False


def test_comprehensive_matrix_loading(verbose=True):
    """Test comprehensive matrix loading benchmark with all methods"""
    if verbose:
        print("\nTesting comprehensive matrix loading benchmarks...")

    # Test different matrix sizes
    Ns = [2**10, 2**14, 2**18, 2**20]
    Cs = [8, 32, 64, 128, 256, 512]
    sizes = [(N, C) for N in Ns for C in Cs]

    # Comprehensive method enumeration (matching the enum in matrix_loading_common.cuh)
    methods = {
        0: "Element-wise",
        1: "Float2 vectorized",
        2: "Float4 vectorized",
        3: "Float8 vectorized",
        4: "Coalesced row",
        5: "Coalesced column",
        6: "Coalesced float4",
        7: "Coalesced float8",
        8: "Shared memory tiled",
        9: "CUB device load",
        10: "CUB block load",
        11: "CUB warp load",
        # Note: Texture memory (12) not included in this test
    }

    results = {}

    for rows, cols in sizes:
        if verbose:
            print(f"\nMatrix size: {rows} x {cols}")

        # Create test matrix
        input_matrix = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        results[(rows, cols)] = {}

        for method_id, method_name in methods.items():
            if verbose:
                print(f"  Testing {method_name}...")

            try:
                # Run benchmark
                times = matrix_load_benchmark.benchmark_matrix_loading(
                    input_matrix, method_id, iterations=50
                )

                # Calculate statistics
                times_np = np.array(times)
                mean_time = np.mean(times_np)
                std_time = np.std(times_np)
                min_time = np.min(times_np)

                # Calculate bandwidth (GB/s)
                # Each operation reads and writes the matrix once
                bytes_transferred = (
                    2 * rows * cols * 4
                )  # 2 ops * elements * 4 bytes/float
                bandwidth_gb_s = (bytes_transferred / (1024**3)) / (mean_time / 1000)

                results[(rows, cols)][method_id] = {
                    "method": method_name,
                    "mean_time_ms": mean_time,
                    "std_time_ms": std_time,
                    "min_time_ms": min_time,
                    "bandwidth_gb_s": bandwidth_gb_s,
                }

                if verbose:
                    print(f"    Mean time: {mean_time:.3f} ± {std_time:.3f} ms")
                    print(f"    Min time:  {min_time:.3f} ms")
                    print(f"    Bandwidth: {bandwidth_gb_s:.2f} GB/s")

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                results[(rows, cols)][method_id] = None

    return results


def test_comprehensive_matrix_storing(verbose=True):
    """Test comprehensive matrix storing benchmark with all methods"""
    if verbose:
        print("\nTesting comprehensive matrix storing benchmarks...")

    # Test different matrix sizes
    Ns = [2**10, 2**14, 2**18, 2**20]
    Cs = [8, 32, 64, 128, 256, 512]
    sizes = [(N, C) for N in Ns for C in Cs]

    # Comprehensive method enumeration (matching the enum in matrix_loading_common.cuh)
    methods = {
        0: "Element-wise",
        1: "Float2 vectorized",
        2: "Float4 vectorized",
        3: "Float8 vectorized",
        4: "Coalesced row",
        5: "Coalesced column",
        6: "Coalesced float4",
        7: "Coalesced float8",
        8: "Shared memory tiled",
        9: "CUB device store",
        10: "CUB block store",
        11: "CUB warp store",
        # Note: Texture memory (12) not included in this test
    }

    results = {}

    for rows, cols in sizes:
        if verbose:
            print(f"\nMatrix size: {rows} x {cols}")

        # Create output matrix
        output_matrix = torch.zeros(rows, cols, device="cuda", dtype=torch.float32)

        results[(rows, cols)] = {}

        for method_id, method_name in methods.items():
            if verbose:
                print(f"  Testing {method_name}...")

            try:
                # Run benchmark
                times = matrix_store_benchmark.benchmark_matrix_storing(
                    output_matrix, method_id, iterations=50
                )

                # Calculate statistics
                times_np = np.array(times)
                mean_time = np.mean(times_np)
                std_time = np.std(times_np)
                min_time = np.min(times_np)

                # Calculate bandwidth (GB/s)
                # Each operation stores the matrix once
                bytes_transferred = (
                    rows * cols * 4
                )  # 1 write op * elements * 4 bytes/float
                bandwidth_gb_s = (bytes_transferred / (1024**3)) / (mean_time / 1000)

                results[(rows, cols)][method_id] = {
                    "method": method_name,
                    "mean_time_ms": mean_time,
                    "std_time_ms": std_time,
                    "min_time_ms": min_time,
                    "bandwidth_gb_s": bandwidth_gb_s,
                }

                if verbose:
                    print(f"    Mean time: {mean_time:.3f} ± {std_time:.3f} ms")
                    print(f"    Min time:  {min_time:.3f} ms")
                    print(f"    Bandwidth: {bandwidth_gb_s:.2f} GB/s")

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                results[(rows, cols)][method_id] = None

    return results


def analyze_comprehensive_results(results):
    """Analyze and visualize comprehensive benchmark results"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK ANALYSIS")
    print("=" * 80)

    # Create summary table
    print(
        f"{'Matrix Size':<15} {'Method':<25} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15}"
    )
    print("-" * 70)

    for size, methods in results.items():
        for method_id, data in methods.items():
            if data is not None:
                size_str = f"{size[0]}x{size[1]}"
                print(
                    f"{size_str:<15} {data['method']:<25} {data['mean_time_ms']:<12.3f} {data['bandwidth_gb_s']:<15.2f}"
                )

    # Find best and second best methods for each size
    print("\n## Best Performance Methods by Matrix Size")
    print(
        "\n| Matrix Size | Best Method | Time (ms) | BW (GB/s) | 2nd Best Method | 2nd Min (ms) |"
    )
    print(
        "|-------------|-------------|-----------|-----------|-----------------|--------------|"
    )

    for size, methods in results.items():
        # Sort methods by bandwidth to get best and second best
        valid_methods = [
            (data["bandwidth_gb_s"], data)
            for method_id, data in methods.items()
            if data is not None
        ]
        valid_methods.sort(
            reverse=True, key=lambda x: x[0]
        )  # Sort by bandwidth descending

        if len(valid_methods) >= 1:
            best_data = valid_methods[0][1]
            size_str = f"{size[0]}x{size[1]}"

            # Second best method info
            second_best_method = "N/A"
            second_best_min_time = "N/A"
            if len(valid_methods) >= 2:
                second_best_data = valid_methods[1][1]
                second_best_method = second_best_data["method"]
                second_best_min_time = f"{second_best_data['min_time_ms']:.4f}"

            print(
                f"| {size_str} | {best_data['method']} | {best_data['mean_time_ms']:.4f} | {best_data['bandwidth_gb_s']:.2f} | {second_best_method} | {second_best_min_time} |"
            )


def main():
    """Main benchmark execution"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="CUDA Matrix Memory Operations Benchmark Suite"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output showing intermediate results",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress intermediate output (opposite of --verbose)",
    )
    args = parser.parse_args()

    # Determine verbosity (--quiet overrides --verbose)
    verbose = args.verbose and not args.quiet

    print("CUDA Matrix Memory Operations Comprehensive Benchmark Suite")
    print("(Load + Store Benchmarks)")
    print("=" * 70)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch version: {torch.__version__}")
    if not verbose:
        print("Use --verbose flag to see detailed intermediate results")
    print("")

    # Run tests
    success = True

    # Test basic functionality
    if not test_basic_functionality(verbose=verbose):
        success = False
        return

    # Run comprehensive benchmarks
    if success:
        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE LOAD BENCHMARKS")
        print("=" * 60)
        load_results = test_comprehensive_matrix_loading(verbose=verbose)
        analyze_comprehensive_results(load_results)

        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE STORE BENCHMARKS")
        print("=" * 60)
        store_results = test_comprehensive_matrix_storing(verbose=verbose)
        analyze_comprehensive_results(store_results)


if __name__ == "__main__":
    main()
