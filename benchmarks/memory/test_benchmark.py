#!/usr/bin/env python3

import torch
import numpy as np
import argparse
import json
import datetime
import os
import matrix_load_benchmark
import matrix_store_benchmark

# Import CuTe load benchmarks
try:
    from cute_load import run_cute_load_benchmark, CUTE_AVAILABLE
    import cutlass

    CUTLASS_FLOAT32 = cutlass.Float32
except (ImportError, NameError):
    CUTE_AVAILABLE = False
    CUTLASS_FLOAT32 = None
    print("Warning: CuTe DSL not available. Skipping CuTe benchmarks.")

NS = [2**14, 2**18, 2**20]
CS = [8, 32, 64, 128, 256, 512]


def get_kernel_memory_ops(method_id, is_cute=False):
    if is_cute:
        return 1, 1

    # Load operations: (global_reads, global_writes)
    load_ops = {
        0: (1, 0),
        1: (1, 0),
        2: (1, 0),
        3: (1, 0),
        4: (1, 0),
        5: (1, 1),
        6: (1, 0),
        7: (1, 0),
        8: (1, 0),
        9: (1, 0),
        10: (1, 0),
        11: (1, 0),
        12: (1, 0),
        13: (1, 0),
        14: (1, 0),
        15: (1, 0),
        16: (1, 0),
        17: (1, 0),
        18: (1, 0),
    }

    # Store operations: (global_reads, global_writes)
    store_ops = {
        0: (0, 1),
        1: (0, 1),
        2: (0, 1),
        3: (0, 1),
        4: (0, 1),
        5: (0, 1),
        6: (0, 1),
        7: (0, 1),
        8: (0, 1),
        9: (0, 1),
        10: (0, 1),
        11: (0, 1),
        12: (0, 1),
        13: (0, 1),
        14: (0, 1),
        15: (0, 1),
        16: (0, 1),
        17: (0, 1),
        18: (0, 1),
    }

    # Check if this is a store operation (methods 0-18 for store)
    if method_id in store_ops:
        return store_ops[method_id]

    # Default to load operations
    return load_ops.get(method_id, (1, 0))


def test_basic_functionality(verbose=True):
    """Test basic matrix loading and storing functionality"""
    print("Testing basic matrix loading functionality...")

    # Create test matrix
    input_matrix = torch.randn(512, 512, device="cuda", dtype=torch.float32)

    try:
        # Test basic element-wise loading (method 0)
        times, _ = matrix_load_benchmark.benchmark_matrix_loading(
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

                # Test a few more store methods to ensure they work
                test_methods = [
                    2,
                    9,
                    13,
                    17,
                ]  # Float4, CUB device, CUB block warp-transpose, PTX float4
                for method in test_methods:
                    try:
                        # Reset output matrix
                        output_matrix.zero_()

                        # Test the method
                        test_times = matrix_store_benchmark.benchmark_matrix_storing(
                            output_matrix, method, iterations=5
                        )

                        if len(test_times) == 5 and all(t > 0 for t in test_times):
                            if verbose:
                                print(f"✓ Matrix storing method {method} test passed!")
                        else:
                            if verbose:
                                print(f"✗ Matrix storing method {method} test failed!")
                            return False

                    except Exception as e:
                        if verbose:
                            print(
                                f"✗ Matrix storing method {method} test failed with error: {e}"
                            )
                        return False

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


def test_load_correctness_large_matrix(verbose=True):
    """Test that all load methods produce correct results for large matrix size"""
    if verbose:
        print("\nTesting load correctness for large matrix (2^20 x 512)...")

    # Large matrix size for testing
    rows = 2**20  # 1,048,576
    cols = 512

    # Comprehensive method enumeration for loading operations
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
        12: "CUB device cache-modified",
        13: "CUB block warp-transpose",
        14: "CUB block striped-transpose",
        15: "CUB warp striped",
        # Note: Texture memory (16) not included in this test
        17: "PTX float4 ld.global",
        18: "PTX float4 ld.global.nc",
    }

    # Create input matrix with known values (use constant value for easier verification)
    input_value = 2.0
    input_matrix = torch.full(
        (rows, cols), input_value, device="cuda", dtype=torch.float32
    )

    # Expected output value after processing (input * 1.001f)
    expected_value = input_value * 1.001

    results = {}
    failed_methods = []

    for method_id, method_name in methods.items():
        if verbose:
            print(f"  Testing {method_name} (method {method_id})...")

        try:
            # Run the load operation with store=True
            times, output_matrix = matrix_load_benchmark.benchmark_matrix_loading(
                input_matrix, method_id, iterations=1, store=True
            )

            # Check if the operation completed successfully
            if len(times) == 1 and times[0] > 0:
                # Verify that the output matrix contains expected processed values
                if torch.allclose(
                    output_matrix,
                    torch.full_like(output_matrix, expected_value),
                    atol=1e-5,
                ):
                    if verbose:
                        print(f"    ✓ Correctness test passed!")
                    results[method_id] = {
                        "method": method_name,
                        "status": "PASS",
                        "time_ms": times[0],
                    }
                else:
                    # Debug: Check what values are actually in the matrix
                    unique_values = torch.unique(output_matrix)
                    min_val = torch.min(output_matrix).item()
                    max_val = torch.max(output_matrix).item()
                    mean_val = torch.mean(output_matrix).item()

                    # Additional debug info for vectorized methods
                    non_zero_count = torch.count_nonzero(output_matrix).item()
                    total_elements = output_matrix.numel()

                    if verbose:
                        print(
                            f"    ✗ Correctness test failed! Matrix does not contain expected processed values"
                        )
                        print(f"      Unique values: {unique_values.tolist()}")
                        print(
                            f"      Min: {min_val:.6f}, Max: {max_val:.6f}, Mean: {mean_val:.6f}"
                        )
                        print(f"      Expected: {expected_value:.6f}")
                        print(
                            f"      Non-zero elements: {non_zero_count}/{total_elements} ({100 * non_zero_count / total_elements:.1f}%)"
                        )
                        print(f"      Matrix shape: {output_matrix.shape}")

                    results[method_id] = {
                        "method": method_name,
                        "status": "FAIL",
                        "time_ms": times[0],
                        "error": f"Matrix does not contain expected processed values (min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, expected={expected_value:.6f}, non_zero={non_zero_count}/{total_elements})",
                    }
                    failed_methods.append(method_id)
            else:
                if verbose:
                    print(f"    ✗ Operation failed to complete!")
                results[method_id] = {
                    "method": method_name,
                    "status": "FAIL",
                    "time_ms": times[0] if times else 0,
                    "error": "Operation failed to complete",
                }
                failed_methods.append(method_id)

        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            results[method_id] = {
                "method": method_name,
                "status": "ERROR",
                "time_ms": 0,
                "error": str(e),
            }
            failed_methods.append(method_id)

    # Summary
    total_methods = len(methods)
    passed_methods = total_methods - len(failed_methods)

    if verbose:
        print(f"\n=== Load Correctness Test Summary ===")
        print(f"Matrix size: {rows} x {cols}")
        print(f"Input value: {input_value}")
        print(f"Expected output value: {expected_value:.6f}")
        print(f"Total methods tested: {total_methods}")
        print(f"Passed: {passed_methods}")
        print(f"Failed: {len(failed_methods)}")

        if failed_methods:
            print(f"\nFailed methods:")
            for method_id in failed_methods:
                print(
                    f"  - Method {method_id}: {results[method_id]['method']} - {results[method_id]['error']}"
                )
        else:
            print(f"\n✓ All load methods passed correctness test!")

    return results, failed_methods


def test_comprehensive_matrix_loading(verbose=True):
    """Test comprehensive matrix loading benchmark with all methods"""
    if verbose:
        print("\nTesting comprehensive matrix loading benchmarks...")

    # Test different matrix sizes
    sizes = [(N, C) for N in NS for C in CS]

    # Comprehensive method enumeration (matching the enum in matrix_loading_common.cuh)
    cuda_methods = {
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
        12: "CUB device cache-modified",
        13: "CUB block warp-transpose",
        14: "CUB block striped-transpose",
        15: "CUB warp striped",
        # Note: Texture memory (16) not included in this test
        17: "PTX float4 ld.global",
        18: "PTX float4 ld.global.nc",
    }

    # Add CuTe DSL methods if available
    if CUTE_AVAILABLE:
        cute_methods = {
            "cute_elementwise": "CuTe Elementwise",
            "cute_vectorized": "CuTe Vectorized",
            # "cute_tiled": "CuTe Tiled (Shared Memory)"  # Commented out due to memory alignment issues
        }
        methods = {**cuda_methods, **cute_methods}
    else:
        methods = cuda_methods

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
                # Check if this is a CuTe method
                if isinstance(method_id, str) and method_id.startswith("cute_"):
                    # Handle CuTe methods
                    if not CUTE_AVAILABLE or CUTLASS_FLOAT32 is None:
                        raise RuntimeError("CuTe DSL not available")

                    cute_method = method_id.replace("cute_", "")
                    mean_time, bandwidth_gb_s = run_cute_load_benchmark(
                        M=rows,
                        N=cols,
                        dtype=CUTLASS_FLOAT32,
                        method=cute_method,
                        iterations=50,
                        warmup_iterations=10,
                        verbose=False,
                    )

                    results[(rows, cols)][method_id] = {
                        "method": method_name,
                        "mean_time_ms": mean_time,
                        "std_time_ms": 0.0,  # CuTe benchmark doesn't return std
                        "min_time_ms": mean_time,  # Use mean as min for consistency
                        "bandwidth_gb_s": bandwidth_gb_s,
                    }

                    if verbose:
                        print(f"    Mean time: {mean_time:.3f} ms")
                        print(f"    Bandwidth: {bandwidth_gb_s:.2f} GB/s")

                else:
                    # Handle traditional CUDA methods
                    times, _ = matrix_load_benchmark.benchmark_matrix_loading(
                        input_matrix, method_id, iterations=50
                    )

                    # Calculate statistics
                    times_np = np.array(times)
                    mean_time = np.mean(times_np)
                    std_time = np.std(times_np)
                    min_time = np.min(times_np)

                    # Calculate bandwidth (GB/s)
                    # Each operation reads and writes the matrix once
                    is_cute = isinstance(method_id, str) and method_id.startswith(
                        "cute_"
                    )
                    global_reads, global_writes = get_kernel_memory_ops(
                        method_id, is_cute
                    )
                    bytes_transferred = (
                        (global_reads + global_writes) * rows * cols * 4
                    )  # 2 ops * elements * 4 bytes/float
                    bandwidth_gb_s = (bytes_transferred / (1024**3)) / (
                        mean_time / 1000
                    )

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


def test_store_correctness_large_matrix(verbose=True):
    """Test that all store methods produce correct results for large matrix size"""
    if verbose:
        print("\nTesting store correctness for large matrix (2^20 x 256)...")

    # Large matrix size for testing (reduced to fit in GPU memory)
    rows = 2**20  # 1,048,576
    cols = 256

    # Comprehensive method enumeration for storing operations
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
        12: "CUB device cache-modified",
        13: "CUB block warp-transpose",
        14: "CUB block striped-transpose",
        15: "CUB warp striped",
        # Note: Texture memory (16) not included in this test
        17: "PTX float4 st.global",
        18: "PTX float4 st.global.wb",
    }

    # Create output matrix
    output_matrix = torch.zeros(rows, cols, device="cuda", dtype=torch.float32)
    expected_value = 1.0

    results = {}
    failed_methods = []

    for method_id, method_name in methods.items():
        if verbose:
            print(f"  Testing {method_name} (method {method_id})...")

        try:
            # Reset output matrix to zeros
            output_matrix.zero_()

            # Run the store operation
            times = matrix_store_benchmark.benchmark_matrix_storing(
                output_matrix, method_id, iterations=1
            )

            # Check if the operation completed successfully
            if len(times) == 1 and times[0] > 0:
                # Verify that the output matrix contains 1s
                if torch.allclose(
                    output_matrix,
                    torch.ones_like(output_matrix) * expected_value,
                    atol=1e-5,
                ):
                    if verbose:
                        print(f"    ✓ Correctness test passed!")
                    results[method_id] = {
                        "method": method_name,
                        "status": "PASS",
                        "time_ms": times[0],
                    }
                else:
                    # Debug: Check what values are actually in the matrix
                    unique_values = torch.unique(output_matrix)
                    min_val = torch.min(output_matrix).item()
                    max_val = torch.max(output_matrix).item()
                    mean_val = torch.mean(output_matrix).item()

                    if verbose:
                        print(
                            f"    ✗ Correctness test failed! Matrix does not contain all 1s"
                        )
                        print(f"      Unique values: {unique_values.tolist()}")
                        print(
                            f"      Min: {min_val:.6f}, Max: {max_val:.6f}, Mean: {mean_val:.6f}"
                        )
                        print(f"      Expected: 1.0")

                    results[method_id] = {
                        "method": method_name,
                        "status": "FAIL",
                        "time_ms": times[0],
                        "error": f"Matrix does not contain all 1s (min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f})",
                    }
                    failed_methods.append(method_id)
            else:
                if verbose:
                    print(f"    ✗ Operation failed to complete!")
                results[method_id] = {
                    "method": method_name,
                    "status": "FAIL",
                    "time_ms": times[0] if times else 0,
                    "error": "Operation failed to complete",
                }
                failed_methods.append(method_id)

        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            results[method_id] = {
                "method": method_name,
                "status": "ERROR",
                "time_ms": 0,
                "error": str(e),
            }
            failed_methods.append(method_id)

    # Summary
    total_methods = len(methods)
    passed_methods = total_methods - len(failed_methods)

    if verbose:
        print(f"\n=== Store Correctness Test Summary ===")
        print(f"Matrix size: {rows} x {cols}")
        print(f"Total methods tested: {total_methods}")
        print(f"Passed: {passed_methods}")
        print(f"Failed: {len(failed_methods)}")

        if failed_methods:
            print(f"\nFailed methods:")
            for method_id in failed_methods:
                print(
                    f"  - Method {method_id}: {results[method_id]['method']} - {results[method_id]['error']}"
                )
        else:
            print(f"\n✓ All store methods passed correctness test!")

    return results, failed_methods


def test_comprehensive_matrix_storing(verbose=True):
    """Test comprehensive matrix storing benchmark with all methods"""
    if verbose:
        print("\nTesting comprehensive matrix storing benchmarks...")

    # Test different matrix sizes
    sizes = [(N, C) for N in NS for C in CS]

    # Comprehensive method enumeration for storing operations
    # Note: Store methods use the same method IDs as load methods but with different implementations
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
        12: "CUB device cache-modified",
        13: "CUB block warp-transpose",
        14: "CUB block striped-transpose",
        15: "CUB warp striped",
        # Note: Texture memory (16) not included in this test
        17: "PTX float4 st.global",
        18: "PTX float4 st.global.wb",
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
                global_reads, global_writes = get_kernel_memory_ops(
                    method_id, is_cute=False
                )
                bytes_transferred = (
                    (global_reads + global_writes) * rows * cols * 4
                )  # ops * elements * 4 bytes/float
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


def convert_results_for_json(all_results):
    """Convert results dictionary to JSON-serializable format"""
    json_results = all_results.copy()

    # Convert load results
    if json_results["load_results"]:
        load_results_json = {}
        for size_tuple, methods in json_results["load_results"].items():
            size_key = f"{size_tuple[0]}x{size_tuple[1]}"
            load_results_json[size_key] = methods
        json_results["load_results"] = load_results_json

    # Convert store results
    if json_results["store_results"]:
        store_results_json = {}
        for size_tuple, methods in json_results["store_results"].items():
            size_key = f"{size_tuple[0]}x{size_tuple[1]}"
            store_results_json[size_key] = methods
        json_results["store_results"] = store_results_json

    return json_results


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
                    f"{size_str:<15} {data['method']:<25} {data['min_time_ms']:<12.3f} {data['bandwidth_gb_s']:<15.2f}"
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
                f"| {size_str} | {best_data['method']} | {best_data['min_time_ms']:.4f} | {best_data['bandwidth_gb_s']:.2f} | {second_best_method} | {second_best_min_time} |"
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

    # Benchmark selection arguments
    benchmark_group = parser.add_argument_group("benchmark selection")
    benchmark_group.add_argument(
        "--load", action="store_true", help="Run load benchmarks only"
    )
    benchmark_group.add_argument(
        "--store", action="store_true", help="Run store benchmarks only"
    )
    benchmark_group.add_argument(
        "--cute", action="store_true", help="Run CuTe DSL benchmarks only"
    )
    benchmark_group.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark types (default if no specific type selected)",
    )

    # Output file arguments
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated with timestamp)",
    )
    output_group.add_argument(
        "--no-save", action="store_true", help="Don't save results to file"
    )

    args = parser.parse_args()

    # Determine verbosity (--quiet overrides --verbose)
    verbose = args.verbose and not args.quiet

    # Determine which benchmarks to run
    # CuTe is now integrated into load benchmarks
    run_load = args.load or args.cute or args.all
    run_store = args.store or args.all
    run_cute = args.cute or args.all

    # If no specific benchmark selected, run all
    if not (args.load or args.store or args.cute or args.all):
        run_load = run_store = run_cute = True

    print("CUDA Matrix Memory Operations Comprehensive Benchmark Suite")

    # Print which benchmarks will run
    benchmark_types = []
    if run_load:
        benchmark_types.append("Load")
    if run_store:
        benchmark_types.append("Store")
    if run_cute:
        benchmark_types.append("CuTe DSL")
    print(f"({' + '.join(benchmark_types)} Benchmarks)")
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

    # Test store correctness for large matrix
    if run_store:
        print("\n" + "=" * 60)
        print("TESTING STORE CORRECTNESS FOR LARGE MATRIX")
        print("=" * 60)
        store_correctness_results, failed_store_methods = (
            test_store_correctness_large_matrix(verbose=verbose)
        )

        if failed_store_methods:
            print(
                f"\n⚠️  Warning: {len(failed_store_methods)} store methods failed correctness test!"
            )
            success = False
        else:
            print(f"\n✓ All store methods passed correctness test for large matrix!")

    # Test load correctness for large matrix
    if run_load:
        print("\n" + "=" * 60)
        print("TESTING LOAD CORRECTNESS FOR LARGE MATRIX")
        print("=" * 60)
        load_correctness_results, failed_load_methods = (
            test_load_correctness_large_matrix(verbose=verbose)
        )

        if failed_load_methods:
            print(
                f"\n⚠️  Warning: {len(failed_load_methods)} load methods failed correctness test!"
            )
            success = False
        else:
            print(f"\n✓ All load methods passed correctness test for large matrix!")

    # Run comprehensive benchmarks based on selection
    all_results = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "gpu_name": gpu_name,
            "pytorch_version": torch.__version__,
            "cute_available": CUTE_AVAILABLE,
            "benchmarks_run": {"load": run_load, "store": run_store, "cute": run_cute},
        },
        "load_results": None,
        "store_results": None,
    }

    if success:
        if run_load:
            print("\n" + "=" * 60)
            print("RUNNING COMPREHENSIVE LOAD BENCHMARKS")
            print("=" * 60)
            load_results = test_comprehensive_matrix_loading(verbose=verbose)
            all_results["load_results"] = load_results
            analyze_comprehensive_results(load_results)

        if run_store:
            print("\n" + "=" * 60)
            print("RUNNING COMPREHENSIVE STORE BENCHMARKS")
            print("=" * 60)
            store_results = test_comprehensive_matrix_storing(verbose=verbose)
            all_results["store_results"] = store_results
            analyze_comprehensive_results(store_results)

        # Show message if CuTe was requested but not available
        if run_cute and not CUTE_AVAILABLE and not run_load:
            print("\n" + "=" * 60)
            print("CUTE DSL BENCHMARKS REQUESTED BUT NOT AVAILABLE")
            print("=" * 60)
            print(
                "Install CuTe DSL with: pip install nvidia-cutlass nvidia-cutlass-dsl"
            )

        # Save results to file
        if not args.no_save:
            if args.output:
                output_file = args.output
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                benchmark_types = []
                if run_load:
                    benchmark_types.append("load")
                if run_store:
                    benchmark_types.append("store")
                if run_cute and CUTE_AVAILABLE:
                    benchmark_types.append("cute")
                benchmark_str = "_".join(benchmark_types) if benchmark_types else "all"
                output_file = f"benchmark_results_{benchmark_str}_{timestamp}.json"

            try:
                # Ensure output directory exists
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Convert tuple keys to strings for JSON serialization
                json_results = convert_results_for_json(all_results)

                with open(output_file, "w") as f:
                    json.dump(json_results, f, indent=2)

                print(f"\n✓ Results saved to: {output_file}")

            except Exception as e:
                print(f"\n✗ Failed to save results: {e}")
        else:
            print("\n• Results not saved (--no-save flag used)")


if __name__ == "__main__":
    main()
