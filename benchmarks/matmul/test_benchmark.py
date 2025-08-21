#!/usr/bin/env python3
"""
CLI for implicit GEMM (matmul) benchmarks and correctness checks.

Examples:
  - Small correctness (quick sanity across methods):
    python benchmarks/matmul/test_benchmark.py small --verbose

  - Large correctness (unique indices, fp16 inputs), method 3 or 4:
    python benchmarks/matmul/test_benchmark.py large --nrows 1048576 --ccols 512 --subset 8192 --method 3 --verbose

  - Benchmarks across sizes, aggregated JSON for analysis:
    python benchmarks/matmul/test_benchmark.py bench --ns 16384,262144,1048576 --cs 32,64,128,256,512 --iters 10 --out benchmarks/matmul/bench_results.json

Use analyze_results.py to visualize the aggregated JSON.
"""

import argparse
import datetime
import json
import os
import numpy as np
import torch
import implicit_gemm_benchmark as ig

METHODS = {
    0: "NAIVE_F32",
    1: "WMMA_F16_ACC_F32",
    2: "F32_PTX_V4",
    3: "WMMA_F16_ACC_F32_DB_AMPERE",
    4: "WMMA_DB_AMPERE_GENERIC_ATOMIC",
    5: "WMMA_DB_AMPERE_GENERIC_STORE",
    6: "CUB_F32_BLOCKLOAD",
}
NS = [2**14, 2**18, 2**20]
CS = [32, 64, 128, 256, 512]


def run_correctness_small(verbose=True):
    """Quick correctness check: compare against dense gather->gemm->add->scatter reference."""
    # Use memory-benchmark style: rows=N, cols=C for A, C, D
    Nrows, Ccols = 256, 128
    M, K, Nout = Nrows, Ccols, Ccols
    P, Q = 512, Nrows

    # Build inputs
    A32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B32 = torch.randn(K, Nout, device="cuda", dtype=torch.float32)  # [C, C]
    C32 = torch.randn(M, Nout, device="cuda", dtype=torch.float32)
    A16 = A32.half()
    B16 = B32.half()
    C16 = C32.half()
    Abf = A32.bfloat16()
    Bbf = B32.bfloat16()
    Cbf = C32.bfloat16()

    a_inds = torch.randperm(M, device="cuda", dtype=torch.long)[:P]
    d_inds = torch.randperm(Q, device="cuda", dtype=torch.long)[:P]

    # Reference
    D_ref = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
    A_g = A32[a_inds, :]  # [P, K]
    C_g = C32[a_inds, :]  # [P, Nout]
    D_ref.index_add_(0, d_inds, A_g @ B32 + C_g)

    # NAIVE_F32
    D0 = torch.zeros_like(D_ref)
    ig.benchmark_implicit_gemm(
        A32, B32, C32, a_inds, d_inds, D0, method=0, iterations=1
    )

    # WMMA_F16_ACC_F32
    D1 = torch.zeros_like(D_ref)
    ig.benchmark_implicit_gemm(
        A16, B16, C16, a_inds, d_inds, D1, method=1, iterations=1
    )

    # F32_PTX_V4
    D2 = torch.zeros_like(D_ref)
    ig.benchmark_implicit_gemm(
        A32, B32, C32, a_inds, d_inds, D2, method=2, iterations=1
    )

    # WMMA_F16_ACC_F32_DB_AMPERE
    D3 = torch.zeros_like(D_ref)
    ig.benchmark_implicit_gemm(
        A16, B16, C16, a_inds, d_inds, D3, method=3, iterations=1
    )

    # WMMA_DB_AMPERE_GENERIC (f16->f32)
    D4f = torch.zeros_like(D_ref)
    ig.benchmark_implicit_gemm(
        A16, B16, C16, a_inds, d_inds, D4f, method=4, iterations=1
    )

    # WMMA_DB_AMPERE_GENERIC (f16->f16)
    D4h = torch.zeros(Q, Nout, device="cuda", dtype=torch.float16)
    ig.benchmark_implicit_gemm(
        A16, B16, C16, a_inds, d_inds, D4h, method=4, iterations=1
    )

    # CUB_F32_BLOCKLOAD (f16->f32)
    D6 = torch.zeros_like(D_ref)
    ig.benchmark_implicit_gemm(
        A16, B16, C16, a_inds, d_inds, D6, method=6, iterations=1
    )

    # CUB_F32_BLOCKLOAD (bf16->f32)
    D6bf = torch.zeros_like(D_ref)
    ig.benchmark_implicit_gemm(
        Abf, Bbf, Cbf, a_inds, d_inds, D6bf, method=6, iterations=1
    )

    # CUB_F32_BLOCKLOAD (bf16->bf16)
    D6bf16 = torch.zeros(Q, Nout, device="cuda", dtype=torch.bfloat16)
    ig.benchmark_implicit_gemm(
        Abf, Bbf, Cbf, a_inds, d_inds, D6bf16, method=6, iterations=1
    )

    # BF16 path disabled in this build
    ok0 = torch.allclose(D0, D_ref, atol=1e-3, rtol=1e-3)
    ok1 = torch.allclose(D1, D_ref, atol=2e-2, rtol=2e-2)  # relaxed for fp16
    ok2 = torch.allclose(D2, D_ref, atol=1e-3, rtol=1e-3)
    ok3 = torch.allclose(D3, D_ref, atol=2e-2, rtol=2e-2)
    ok4f = torch.allclose(D4f, D_ref, atol=2e-2, rtol=2e-2)
    ok4h = torch.allclose(D4h.float(), D_ref, atol=3e-2, rtol=3e-2)
    ok6 = torch.allclose(D6, D_ref, atol=2e-2, rtol=2e-2)
    ok6bf = torch.allclose(D6bf, D_ref, atol=2e-2, rtol=2e-2)
    ok6bf16 = torch.allclose(D6bf16.float(), D_ref, atol=3e-2, rtol=3e-2)
    if verbose:
        print(f"NAIVE_F32 correctness: {'OK' if ok0 else 'FAIL'}")
        print(f"WMMA_F16_ACC_F32 correctness: {'OK' if ok1 else 'FAIL'}")
        print(f"F32_PTX_V4 correctness: {'OK' if ok2 else 'FAIL'}")
        print(f"WMMA_F16_ACC_F32_DB_AMPERE correctness: {'OK' if ok3 else 'FAIL'}")
        diff4f = (D4f - D_ref).abs()
        diff4h = (D4h.float() - D_ref).abs()
        print(
            f"WMMA_DB_AMPERE_GENERIC f16->f32 correctness: {'OK' if ok4f else 'FAIL'} | max_abs={diff4f.max().item():.4e}"
        )
        print(
            f"WMMA_DB_AMPERE_GENERIC f16->f16 correctness: {'OK' if ok4h else 'FAIL'} | max_abs={diff4h.max().item():.4e}"
        )
        print(f"WMMA_DB_AMPERE_GENERIC bf16 paths: SKIP")
        diff6 = (D6 - D_ref).abs()
        print(
            f"CUB_F32_BLOCKLOAD f16->f32 correctness: {'OK' if ok6 else 'FAIL'} | max_abs={diff6.max().item():.4e}"
        )
        diff6bf = (D6bf - D_ref).abs()
        print(
            f"CUB_F32_BLOCKLOAD bf16->f32 correctness: {'OK' if ok6bf else 'FAIL'} | max_abs={diff6bf.max().item():.4e}"
        )
        diff6bf16 = (D6bf16.float() - D_ref).abs()
        print(
            f"CUB_F32_BLOCKLOAD bf16->bf16 correctness: {'OK' if ok6bf16 else 'FAIL'} | max_abs={diff6bf16.max().item():.4e}"
        )
    # Only gate on established methods; method 4 is experimental and reported above
    return ok0 and ok1 and ok2 and ok3 and ok6 and ok6bf and ok6bf16


def run_correctness_large_unique(
    Nrows=2**20, Ccols=512, subset=8192, method=3, verbose=True
):
    """Large-scale correctness using unique a_inds/d_inds and subset reference.

    Allocates full-size inputs/outputs on GPU, runs kernel once, then validates
    a random subset of output rows against a dense reference computed only for
    those rows. Uses fp16 inputs for A/C and matches kernel input precision.
    """
    device = torch.device("cuda")
    M, K, Nout = Nrows, Ccols, Ccols
    P, Q = Nrows, Nrows

    # Inputs: keep A/C in half to reduce memory footprint; B in half as well to match kernel
    A16 = torch.randn(M, K, device=device, dtype=torch.float16)
    B16 = torch.randn(K, Nout, device=device, dtype=torch.float16)
    C16 = torch.randn(M, Nout, device=device, dtype=torch.float16)

    # Unique indices
    a_inds = torch.randperm(M, device=device, dtype=torch.long)[:P]
    d_inds = torch.randperm(Q, device=device, dtype=torch.long)[:P]

    # Output
    D = torch.zeros(Q, Nout, device=device, dtype=torch.float32)

    # Run kernel once
    _ = ig.benchmark_implicit_gemm(
        A16, B16, C16, a_inds, d_inds, D, method=method, iterations=1
    )

    # Build inverse mapping from d_inds -> position (tile row index)
    inv_d = torch.empty(Q, device=device, dtype=torch.long)
    inv_d[d_inds] = torch.arange(P, device=device, dtype=torch.long)

    # Subset rows to validate
    S = min(subset, Q)
    rows_subset = torch.randperm(Q, device=device)[:S]
    pos = inv_d[rows_subset]
    a_rows = a_inds[pos]

    # Dense reference only for subset rows (compute in fp32 for accumulation)
    A_sel = A16[a_rows, :].float()
    C_sel = C16[a_rows, :].float()
    Bf = B16.float()
    D_sub_ref = A_sel @ Bf + C_sel

    # Produced output rows
    D_sub = D[rows_subset, :]

    ok = torch.allclose(D_sub, D_sub_ref, atol=2e-2, rtol=2e-2)
    if verbose:
        diff = (D_sub - D_sub_ref).abs()
        print(
            f"Large unique correctness (Nrows={Nrows}, C={Ccols}, S={S}) => {'OK' if ok else 'FAIL'}",
            f"| max_abs={diff.max().item():.4e}",
        )
    return ok


def benchmark_suite(ns=NS, cs=CS, iterations=10, unique=True, verbose=False):
    results = {}
    for Nrows in ns:
        for Ccols in cs:
            M, K, Nout = Nrows, Ccols, Ccols
            S, Q = Nrows, Nrows

            # Build tensors
            A32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
            B32 = torch.randn(K, Nout, device="cuda", dtype=torch.float32)
            C32 = torch.randn(M, Nout, device="cuda", dtype=torch.float32)
            A16, B16, C16 = A32.half(), B32.half(), C32.half()

            # Indices
            if unique:
                a_inds = torch.randperm(M, device="cuda", dtype=torch.long)[:S]
                d_inds = torch.randperm(Q, device="cuda", dtype=torch.long)[:S]
            else:
                a_inds = torch.randint(0, M, (S,), device="cuda", dtype=torch.long)
                d_inds = torch.randint(0, Q, (S,), device="cuda", dtype=torch.long)

            per_size = {}

            # 0: NAIVE_F32
            D0 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            t0 = np.array(
                ig.benchmark_implicit_gemm(
                    A32, B32, C32, a_inds, d_inds, D0, method=0, iterations=iterations
                )
            )
            per_size[0] = {
                "method": METHODS[0],
                "mean_ms": float(t0.mean()),
                "std_ms": float(t0.std()),
                "min_ms": float(t0.min()),
            }

            # 1: WMMA_F16_ACC_F32
            D1 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            t1 = np.array(
                ig.benchmark_implicit_gemm(
                    A16, B16, C16, a_inds, d_inds, D1, method=1, iterations=iterations
                )
            )
            per_size[1] = {
                "method": METHODS[1],
                "mean_ms": float(t1.mean()),
                "std_ms": float(t1.std()),
                "min_ms": float(t1.min()),
            }

            # 2: F32_PTX_V4
            D2 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            t2 = np.array(
                ig.benchmark_implicit_gemm(
                    A32, B32, C32, a_inds, d_inds, D2, method=2, iterations=iterations
                )
            )
            per_size[2] = {
                "method": METHODS[2],
                "mean_ms": float(t2.mean()),
                "std_ms": float(t2.std()),
                "min_ms": float(t2.min()),
            }

            # 3: WMMA_F16_ACC_F32_DB_AMPERE
            D3 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            t3 = np.array(
                ig.benchmark_implicit_gemm(
                    A16, B16, C16, a_inds, d_inds, D3, method=3, iterations=iterations
                )
            )
            per_size[3] = {
                "method": METHODS[3],
                "mean_ms": float(t3.mean()),
                "std_ms": float(t3.std()),
                "min_ms": float(t3.min()),
            }

            # 4: WMMA_DB_AMPERE_GENERIC (f16->f32, atomic)
            D4 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            t4 = np.array(
                ig.benchmark_implicit_gemm(
                    A16, B16, C16, a_inds, d_inds, D4, method=4, iterations=iterations
                )
            )
            per_size[4] = {
                "method": METHODS[4],
                "mean_ms": float(t4.mean()),
                "std_ms": float(t4.std()),
                "min_ms": float(t4.min()),
            }

            # 5: WMMA_DB_AMPERE_GENERIC (f16->f32, store)
            D5 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            t5 = np.array(
                ig.benchmark_implicit_gemm(
                    A16, B16, C16, a_inds, d_inds, D5, method=5, iterations=iterations
                )
            )
            per_size[5] = {
                "method": METHODS[5],
                "mean_ms": float(t5.mean()),
                "std_ms": float(t5.std()),
                "min_ms": float(t5.min()),
            }

            # 6: CUB_F32_BLOCKLOAD (f16->f32)
            D6 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            t6 = np.array(
                ig.benchmark_implicit_gemm(
                    A16, B16, C16, a_inds, d_inds, D6, method=6, iterations=iterations
                )
            )
            per_size[6] = {
                "method": METHODS[6],
                "mean_ms": float(t6.mean()),
                "std_ms": float(t6.std()),
                "min_ms": float(t6.min()),
            }

            results[f"N{Nrows}_C{Ccols}"] = per_size

            if verbose:
                print(
                    f"Benchmark N={Nrows} C={Ccols} | "
                    + ", ".join(
                        f"{METHODS[m]}: {per_size[m]['mean_ms']:.3f} ms (min {per_size[m]['min_ms']:.3f})"
                        for m in sorted(per_size.keys())
                    )
                )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Implicit GEMM (matmul) benchmarks and correctness checks"
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    # Small correctness
    p_small = sub.add_parser("small", help="Run small correctness for all methods")
    p_small.add_argument("--verbose", action="store_true")

    # Large correctness
    p_large = sub.add_parser("large", help="Run large correctness (unique indices)")
    p_large.add_argument(
        "--nrows", type=int, default=int(os.environ.get("IG_NROWS", 2**20))
    )
    p_large.add_argument(
        "--ccols", type=int, default=int(os.environ.get("IG_CCOLS", 512))
    )
    p_large.add_argument(
        "--subset", type=int, default=int(os.environ.get("IG_SUBSET", 2**14))
    )
    p_large.add_argument("--method", type=int, choices=[3, 4, 5, 6], default=3)
    p_large.add_argument("--verbose", action="store_true")

    # Benchmarks
    p_bench = sub.add_parser("bench", help="Run benchmark suite and save JSON")
    p_bench.add_argument(
        "--ns",
        type=str,
        default=os.environ.get("IG_BENCH_NS", ""),
        help="Comma-separated Ns; default from script",
    )
    p_bench.add_argument(
        "--cs",
        type=str,
        default=os.environ.get("IG_BENCH_CS", ""),
        help="Comma-separated Cs; default from script",
    )
    p_bench.add_argument(
        "--iters", type=int, default=int(os.environ.get("IG_BENCH_ITERS", 10))
    )
    p_bench.add_argument(
        "--out", type=str, default=os.environ.get("IG_BENCH_OUT", "bench_results.json")
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    if args.cmd == "small":
        ok = run_correctness_small(verbose=args.verbose)
        print("Small correctness:", "OK" if ok else "FAIL")
        return

    if args.cmd == "large":
        ok = run_correctness_large_unique(
            Nrows=args.nrows,
            Ccols=args.ccols,
            subset=args.subset,
            method=args.method,
            verbose=args.verbose,
        )
        print("Large correctness:", "OK" if ok else "FAIL")
        return

    # Default: bench
    def _parse_list(s, default_list):
        return default_list if not s else [int(x) for x in s.split(",") if x]

    ns = _parse_list(getattr(args, "ns", ""), NS)
    cs = _parse_list(getattr(args, "cs", ""), CS)
    bench_iters = args.iters
    out = args.out

    aggregated = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "gpu_name": torch.cuda.get_device_name(0),
            "pytorch_version": torch.__version__,
            "iterations": bench_iters,
        },
        "methods": METHODS,
        "results": {},
    }
    if os.path.exists(out):
        try:
            with open(out, "r") as f:
                prev = json.load(f)
                if isinstance(prev, dict):
                    aggregated["results"].update(prev.get("results", {}))
        except Exception:
            pass

    # Run consolidated benchmark suite once for all sizes
    suite_results = benchmark_suite(
        ns=ns, cs=cs, iterations=bench_iters, unique=True, verbose=True
    )
    aggregated["results"].update(suite_results)

    out_dir = os.path.dirname(out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved results to: {out}")


if __name__ == "__main__":
    main()
