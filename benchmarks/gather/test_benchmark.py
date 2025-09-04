#!/usr/bin/env python3

import torch
import numpy as np
import argparse
import json
import datetime
import os
import matrix_gather_benchmark

# Optional CuTe-based gather implementation
try:
    import cute_gather  # when executed from benchmarks/gather/

    _CUTE_GATHER_AVAILABLE = True
except Exception:
    try:
        # fallback when executed from repo root
        from benchmarks.gather import cute_gather  # type: ignore

        _CUTE_GATHER_AVAILABLE = True
    except Exception:
        _CUTE_GATHER_AVAILABLE = False


def generate_gather_indices(n_total, n_gather, device="cuda"):
    """Generate sorted, unique gather indices for testing"""
    if n_gather >= n_total:
        # If we need all or more rows, just return all available indices
        return torch.arange(n_total, device=device, dtype=torch.long)

    # Generate random unique indices and sort them
    indices = torch.randperm(n_total, device=device, dtype=torch.long)[:n_gather]
    indices = torch.sort(indices)[0]  # Sort to maintain the requirement
    return indices


def test_basic_functionality(verbose=True):
    """Test basic matrix gathering functionality"""
    print("Testing basic matrix gathering functionality...")

    # Create test matrix
    input_matrix = torch.randn(512, 512, device="cuda", dtype=torch.float32)

    # Create test data for gathering
    n_gather = 256  # Gather half the rows
    gather_indices = generate_gather_indices(512, n_gather, device="cuda")

    try:
        # Test basic element-wise gathering (method 0)
        times, _ = matrix_gather_benchmark.benchmark_matrix_gathering(
            input_matrix, gather_indices, 0, iterations=10
        )

        # Check if we got reasonable results
        if len(times) == 10 and all(t > 0 for t in times):
            if verbose:
                print("✓ Matrix gathering benchmark test passed!")
            return True
        else:
            if verbose:
                print("✗ Matrix gathering benchmark test failed!")
            return False
    except Exception as e:
        if verbose:
            print(f"✗ Matrix gathering benchmark test failed with error: {e}")
        return False


def test_gather_correctness_large_matrix(verbose=True):
    """Test that all gather methods produce correct results for large matrix size"""
    if verbose:
        print("\nTesting gather correctness for large matrix (2^20 x 256)...")

    # Large matrix size for testing (reduced to fit in GPU memory)
    rows = 2**20  # 1,048,576
    cols = 256

    # Comprehensive method enumeration for gathering operations
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
        9: "CUB device gather",
        10: "CUB block gather",
        11: "CUB warp gather",
    }
    if _CUTE_GATHER_AVAILABLE:
        methods[12] = "CuTe TV gather"

    # Create input matrix with known values (use constant value for easier verification)
    input_value = 2.0
    input_matrix = torch.full(
        (rows, cols), input_value, device="cuda", dtype=torch.float32
    )

    # Expected output value after processing (input * 1.001f)
    expected_value = input_value * 1.001

    # Test with different gather ratios
    gather_ratios = [0.25, 0.5, 0.75]

    results = {}
    failed_methods = []

    for gather_ratio in gather_ratios:
        n_gather = max(1, int(rows * gather_ratio))
        gather_indices = generate_gather_indices(rows, n_gather, device="cuda")

        if verbose:
            print(f"  Gather ratio: {gather_ratio:.1%} ({n_gather} rows)")

        results[gather_ratio] = {}

        for method_id, method_name in methods.items():
            if verbose:
                print(f"    Testing {method_name} (method {method_id})...")

            try:
                # Run the gather operation with return_output=True
                if method_id == 12:
                    if not _CUTE_GATHER_AVAILABLE:
                        raise RuntimeError(
                            "CuTe gather not available (missing CUTLASS DSL)"
                        )
                    times, output_matrix = cute_gather.benchmark_matrix_gathering_cute(
                        input_matrix,
                        gather_indices,
                        iterations=1,
                        return_output=True,
                    )
                else:
                    times, output_matrix = (
                        matrix_gather_benchmark.benchmark_matrix_gathering(
                            input_matrix,
                            gather_indices,
                            method_id,
                            iterations=1,
                            return_output=True,
                        )
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
                            print(f"      ✓ Correctness test passed!")
                        results[gather_ratio][method_id] = {
                            "method": method_name,
                            "status": "PASS",
                            "time_ms": times[0],
                            "n_gather": n_gather,
                        }
                    else:
                        # Debug: Check what values are actually in the matrix
                        unique_values = torch.unique(output_matrix)
                        min_val = torch.min(output_matrix).item()
                        max_val = torch.max(output_matrix).item()
                        mean_val = torch.mean(output_matrix).item()

                        # Additional debug info
                        non_zero_count = torch.count_nonzero(output_matrix).item()
                        total_elements = output_matrix.numel()

                        if verbose:
                            print(
                                f"      ✗ Correctness test failed! Matrix does not contain expected processed values"
                            )
                            print(f"        Unique values: {unique_values.tolist()}")
                            print(
                                f"        Min: {min_val:.6f}, Max: {max_val:.6f}, Mean: {mean_val:.6f}"
                            )
                            print(f"        Expected: {expected_value:.6f}")
                            print(
                                f"        Non-zero elements: {non_zero_count}/{total_elements} ({100 * non_zero_count / total_elements:.1f}%)"
                            )
                            print(f"        Matrix shape: {output_matrix.shape}")

                        results[gather_ratio][method_id] = {
                            "method": method_name,
                            "status": "FAIL",
                            "time_ms": times[0],
                            "n_gather": n_gather,
                            "error": f"Matrix does not contain expected processed values (min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, expected={expected_value:.6f}, non_zero={non_zero_count}/{total_elements})",
                        }
                        failed_methods.append((gather_ratio, method_id))
                else:
                    if verbose:
                        print(f"      ✗ Operation failed to complete!")
                    results[gather_ratio][method_id] = {
                        "method": method_name,
                        "status": "FAIL",
                        "time_ms": times[0] if times else 0,
                        "n_gather": n_gather,
                        "error": "Operation failed to complete",
                    }
                    failed_methods.append((gather_ratio, method_id))

            except Exception as e:
                if verbose:
                    print(f"      ✗ Error: {e}")
                results[gather_ratio][method_id] = {
                    "method": method_name,
                    "status": "ERROR",
                    "time_ms": 0,
                    "n_gather": n_gather,
                    "error": str(e),
                }
                failed_methods.append((gather_ratio, method_id))

    # Summary
    total_tests = len(methods) * len(gather_ratios)
    passed_tests = total_tests - len(failed_methods)

    if verbose:
        print(f"\n=== Gather Correctness Test Summary ===")
        print(f"Matrix size: {rows} x {cols}")
        print(f"Input value: {input_value}")
        print(f"Expected output value: {expected_value:.6f}")
        print(f"Gather ratios tested: {gather_ratios}")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {len(failed_methods)}")

        if failed_methods:
            print(f"\nFailed tests:")
            for gather_ratio, method_id in failed_methods:
                print(
                    f"  - Gather {gather_ratio:.1%}, Method {method_id}: {results[gather_ratio][method_id]['method']} - {results[gather_ratio][method_id]['error']}"
                )
        else:
            print(f"\n✓ All gather methods passed correctness test!")

    return results, failed_methods


def test_comprehensive_matrix_gathering(verbose=True):
    """Test comprehensive matrix gathering benchmark with all methods"""
    if verbose:
        print("\nTesting comprehensive matrix gathering benchmarks...")

    # Test smaller matrix sizes for quick testing
    Ns = [2**10, 2**14, 2**18, 2**20]
    Cs = [8, 32, 64, 128, 256, 512]
    sizes = [(N, C) for N in Ns for C in Cs]

    # Gather ratios to test (percentage of rows to gather)
    gather_ratios = [0.25, 0.5, 0.75]

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
        9: "CUB device gather",
        10: "CUB block gather",
        11: "CUB warp gather",
    }
    if _CUTE_GATHER_AVAILABLE:
        methods[12] = "CuTe TV gather"

    results = {}

    for rows, cols in sizes:
        if verbose:
            print(f"\nMatrix size: {rows} x {cols}")

        # Create test matrix
        input_matrix = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        results[(rows, cols)] = {}

        for gather_ratio in gather_ratios:
            n_gather = max(1, int(rows * gather_ratio))
            gather_indices = generate_gather_indices(rows, n_gather, device="cuda")

            if verbose:
                print(f"  Gather ratio: {gather_ratio:.1%} ({n_gather} rows)")

            results[(rows, cols)][gather_ratio] = {}

            for method_id, method_name in methods.items():
                if verbose:
                    print(f"    Testing {method_name}...")

                try:
                    # Run benchmark with fewer iterations for quick test
                    if method_id == 12:
                        if not _CUTE_GATHER_AVAILABLE:
                            raise RuntimeError(
                                "CuTe gather not available (missing CUTLASS DSL)"
                            )
                        times, _ = cute_gather.benchmark_matrix_gathering_cute(
                            input_matrix,
                            gather_indices,
                            iterations=20,
                            return_output=False,
                        )
                    else:
                        times, _ = matrix_gather_benchmark.benchmark_matrix_gathering(
                            input_matrix, gather_indices, method_id, iterations=20
                        )

                    # Calculate statistics
                    times_np = np.array(times)
                    mean_time = np.mean(times_np)
                    std_time = np.std(times_np)
                    min_time = np.min(times_np)

                    # Calculate bandwidth (GB/s)
                    # Each operation reads from source matrix and writes to output matrix
                    bytes_transferred = (
                        2 * n_gather * cols * 4
                    )  # 2 ops (read + write) * gathered elements * 4 bytes/float
                    bandwidth_gb_s = (bytes_transferred / (1024**3)) / (
                        mean_time / 1000
                    )

                    results[(rows, cols)][gather_ratio][method_id] = {
                        "method": method_name,
                        "mean_time_ms": mean_time,
                        "std_time_ms": std_time,
                        "min_time_ms": min_time,
                        "bandwidth_gb_s": bandwidth_gb_s,
                        "n_gather": n_gather,
                        "gather_ratio": gather_ratio,
                    }

                    if verbose:
                        print(f"      Mean time: {mean_time:.3f} ± {std_time:.3f} ms")
                        print(f"      Min time:  {min_time:.3f} ms")
                        print(f"      Bandwidth: {bandwidth_gb_s:.2f} GB/s")

                except Exception as e:
                    if verbose:
                        print(f"      Error: {e}")
                    results[(rows, cols)][gather_ratio][method_id] = None

    return results


def analyze_gather_results(results):
    """Analyze and visualize comprehensive gather benchmark results"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE GATHER BENCHMARK ANALYSIS")
    print("=" * 80)

    # Create summary table
    print(
        f"{'Matrix Size':<15} {'Gather %':<10} {'Method':<25} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15}"
    )
    print("-" * 80)

    for size, gather_ratios in results.items():
        for gather_ratio, methods in gather_ratios.items():
            for method_id, data in methods.items():
                if data is not None:
                    size_str = f"{size[0]}x{size[1]}"
                    gather_str = f"{gather_ratio:.1%}"
                    print(
                        f"{size_str:<15} {gather_str:<10} {data['method']:<25} {data['min_time_ms']:<12.3f} {data['bandwidth_gb_s']:<15.2f}"
                    )

    # Find best methods for each matrix size and gather ratio
    print("\n## Best Performance Methods by Matrix Size and Gather Ratio")
    print(
        "\n| Matrix Size | Gather % | Best Method | Time (ms) | BW (GB/s) | 2nd Best Method | 2nd Min (ms) |"
    )
    print(
        "|-------------|----------|-------------|-----------|-----------|-----------------|--------------|"
    )

    for size, gather_ratios in results.items():
        for gather_ratio, methods in gather_ratios.items():
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
                gather_str = f"{gather_ratio:.1%}"

                # Second best method info
                second_best_method = "N/A"
                second_best_min_time = "N/A"
                if len(valid_methods) >= 2:
                    second_best_data = valid_methods[1][1]
                    second_best_method = second_best_data["method"]
                    second_best_min_time = f"{second_best_data['min_time_ms']:.4f}"

                print(
                    f"| {size_str} | {gather_str} | {best_data['method']} | {best_data['min_time_ms']:.4f} | {best_data['bandwidth_gb_s']:.2f} | {second_best_method} | {second_best_min_time} |"
                )


def convert_results_for_json(results):
    """Convert results dictionary to JSON-serializable format"""
    json_results = results.copy()

    # Convert tuple keys to strings for JSON serialization
    results_json = {}
    for size_tuple, gather_ratios in json_results.items():
        size_key = f"{size_tuple[0]}x{size_tuple[1]}"
        results_json[size_key] = {}

        for gather_ratio, methods in gather_ratios.items():
            results_json[size_key][str(gather_ratio)] = methods

    return results_json


def main():
    """Main benchmark execution"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="CUDA Matrix Gather Operations Benchmark Suite"
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
        "--comprehensive",
        action="store_true",
        help="Run comprehensive gather benchmarks across matrix sizes and gather ratios",
    )
    benchmark_group.add_argument(
        "--basic", action="store_true", help="Run basic functionality test only"
    )
    benchmark_group.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests for large matrix",
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
    run_comprehensive = args.comprehensive
    run_basic = args.basic
    run_correctness = args.correctness

    # If no specific benchmark selected, run all
    if not (args.comprehensive or args.basic or args.correctness):
        run_comprehensive = run_basic = run_correctness = True

    print("CUDA Matrix Gather Operations Benchmark Suite")
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
    gather_results = None
    correctness_results = None

    # Test basic functionality
    if run_basic:
        if not test_basic_functionality(verbose=verbose):
            success = False
            return

    # Test gather correctness for large matrix
    if success and run_correctness:
        print("\n" + "=" * 60)
        print("TESTING GATHER CORRECTNESS FOR LARGE MATRIX")
        print("=" * 60)
        correctness_results, failed_gather_methods = (
            test_gather_correctness_large_matrix(verbose=verbose)
        )

        if failed_gather_methods:
            print(
                f"\n⚠️  Warning: {len(failed_gather_methods)} gather tests failed correctness test!"
            )
            print(failed_gather_methods)
            success = False
        else:
            print(f"\n✓ All gather methods passed correctness test for large matrix!")

    # Run comprehensive benchmarks
    if success and run_comprehensive:
        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE GATHER BENCHMARKS")
        print("=" * 60)
        gather_results = test_comprehensive_matrix_gathering(verbose=verbose)
        analyze_gather_results(gather_results)

    # Save results to file
    if success and (gather_results or correctness_results) and not args.no_save:
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            benchmark_types = []
            if run_basic:
                benchmark_types.append("basic")
            if run_correctness:
                benchmark_types.append("correctness")
            if run_comprehensive:
                benchmark_types.append("comprehensive")
            benchmark_str = "_".join(benchmark_types) if benchmark_types else "all"
            output_file = f"gather_benchmark_results_{benchmark_str}_{timestamp}.json"

        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Prepare results for saving
            all_results = {
                "metadata": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "gpu_name": gpu_name,
                    "pytorch_version": torch.__version__,
                    "benchmark_type": "gather",
                    "benchmarks_run": {
                        "basic": run_basic,
                        "correctness": run_correctness,
                        "comprehensive": run_comprehensive,
                    },
                },
                "gather_results": gather_results,
                "correctness_results": correctness_results,
            }

            # Convert tuple keys to strings for JSON serialization
            json_results = convert_results_for_json(all_results)

            with open(output_file, "w") as f:
                json.dump(json_results, f, indent=2)

            print(f"\n✓ Results saved to: {output_file}")

        except Exception as e:
            print(f"\n✗ Failed to save results: {e}")
    elif args.no_save:
        print("\n• Results not saved (--no-save flag used)")


if __name__ == "__main__":
    main()
