#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import numpy as np
import torch
import implicit_gemm_benchmark as ig

METHODS = {0: "NAIVE_F32", 1: "WMMA_F16_ACC_F32", 2: "F32_PTX_V4"}
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

    a_inds = torch.randint(0, M, (P,), device="cuda", dtype=torch.long)
    d_inds = torch.randint(0, Q, (P,), device="cuda", dtype=torch.long)

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

    ok0 = torch.allclose(D0, D_ref, atol=1e-3, rtol=1e-3)
    ok1 = torch.allclose(D1, D_ref, atol=2e-2, rtol=2e-2)  # relaxed for fp16
    ok2 = torch.allclose(D2, D_ref, atol=1e-3, rtol=1e-3)
    if verbose:
        print(f"NAIVE_F32 correctness: {'OK' if ok0 else 'FAIL'}")
        print(f"WMMA_F16_ACC_F32 correctness: {'OK' if ok1 else 'FAIL'}")
        print(f"F32_PTX_V4 correctness: {'OK' if ok2 else 'FAIL'}")
    return ok0 and ok1 and ok2


def benchmark_suite(verbose=False):
    results = {}
    for Nrows in NS:
        for Ccols in CS:
            M, K, Nout = Nrows, Ccols, Ccols
            P, Q = Nrows, Nrows  # large-scale: gather/scatter ~ rows
            if verbose:
                print(
                    f"\nBenchmarking rows={Nrows}, cols={Ccols} (A[{M},{K}], B[{K},{Nout}], C[{M},{Nout}], D[{Q},{Nout}])"
                )

            # Build tensors
            A32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
            B32 = torch.randn(K, Nout, device="cuda", dtype=torch.float32)
            C32 = torch.randn(M, Nout, device="cuda", dtype=torch.float32)
            A16 = A32.half()
            B16 = B32.half()
            C16 = C32.half()
            a_inds = torch.randint(0, M, (P,), device="cuda", dtype=torch.long)
            d_inds = torch.randint(0, Q, (P,), device="cuda", dtype=torch.long)

            per_size = {}

            # NAIVE_F32
            D0 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            times0 = ig.benchmark_implicit_gemm(
                A32, B32, C32, a_inds, d_inds, D0, method=0, iterations=10
            )
            t0 = np.array(times0)
            per_size[0] = {
                "method": METHODS[0],
                "mean_ms": float(t0.mean()),
                "std_ms": float(t0.std()),
                "min_ms": float(t0.min()),
            }
            if verbose:
                print(
                    f"  {METHODS[0]}: mean {per_size[0]['mean_ms']:.3f} ms, min {per_size[0]['min_ms']:.3f} ms"
                )

            # WMMA_F16_ACC_F32
            D1 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            times1 = ig.benchmark_implicit_gemm(
                A16, B16, C16, a_inds, d_inds, D1, method=1, iterations=10
            )
            t1 = np.array(times1)
            per_size[1] = {
                "method": METHODS[1],
                "mean_ms": float(t1.mean()),
                "std_ms": float(t1.std()),
                "min_ms": float(t1.min()),
            }
            if verbose:
                print(
                    f"  {METHODS[1]}: mean {per_size[1]['mean_ms']:.3f} ms, min {per_size[1]['min_ms']:.3f} ms"
                )

            # F32_PTX_V4
            D2 = torch.zeros(Q, Nout, device="cuda", dtype=torch.float32)
            times2 = ig.benchmark_implicit_gemm(
                A32, B32, C32, a_inds, d_inds, D2, method=2, iterations=10
            )
            t2 = np.array(times2)
            per_size[2] = {
                "method": METHODS[2],
                "mean_ms": float(t2.mean()),
                "std_ms": float(t2.std()),
                "min_ms": float(t2.min()),
            }
            if verbose:
                print(
                    f"  {METHODS[2]}: mean {per_size[2]['mean_ms']:.3f} ms, min {per_size[2]['min_ms']:.3f} ms"
                )

            results[f"N{Nrows}_C{Ccols}"] = per_size
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Implicit GEMM (gather @ B + bias -> scatter) benchmark"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output JSON file"
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("Implicit GEMM Benchmark (Tensor Cores)")
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}")

    ok = run_correctness_small(verbose=args.verbose)
    if not ok:
        print("⚠️  Correctness check failed (continuing to timings)")

    results = benchmark_suite(verbose=args.verbose)

    payload = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "gpu_name": torch.cuda.get_device_name(0),
            "pytorch_version": torch.__version__,
        },
        "methods": METHODS,
        "results": results,
    }

    out = (
        args.output
        or f"implicit_gemm_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    try:
        out_dir = os.path.dirname(out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {out}")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    main()
