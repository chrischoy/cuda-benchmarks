#!/usr/bin/env python3
"""
CuTe-based row gather benchmark: out = A[idx]

Implements a TV-layout store kernel that gathers rows from a 2D matrix using
an index vector, multiplies by 1.001 (to match C++ benchmarks), and writes to
the output. Exposes a runner compatible with test_benchmark expectations.

Requirements:
- CUTLASS with CuTe DSL Python bindings
- PyTorch with CUDA
"""

from typing import Tuple, List
import time

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _cute_row_gather_kernel(
    mA: cute.Tensor,  # (M, N)
    mIdx: cute.Tensor,  # (G)
    gOut: cute.Tensor,  # tiled out: ((TileG, TileN), (RestG, RestN))
    cOut: cute.Tensor,  # coordinate tensor for out
    out_shape: cute.Shape,  # (G, N)
    tv_layout: cute.Layout,  # (tid, vid) -> (TileG, TileN)
    tiler_gn: cute.Shape,  # (TileG, TileN)
    tiled_copy_out: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Tile selection for this CTA: pick the block tile of output and the
    # corresponding identity coordinates for predication and absolute addressing.
    blk_coord = ((None, None), bidx)
    blkOut = gOut[blk_coord]
    blkCrd = cOut[blk_coord]

    # Form per-thread partitioners using the provided tiled copy (constructed on host).
    # We partition destination (GMEM out tile) and source coords with the same TV layout
    # so predication aligns with the lanes we will store.
    thr_copy_out = tiled_copy_out.get_slice(tidx)
    thrOut = thr_copy_out.partition_D(blkOut)
    thrCrd = thr_copy_out.partition_S(blkCrd)

    # Build a predicate fragment matching the per-thread value shape.
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
    # Residual predication: guard out-of-bounds for partial tiles along (G,N).
    for i in cutlass.range(cute.size(frgPred), unroll_full=True):
        frgPred[i] = cute.elem_less(thrCrd[i], out_shape)

    # Scale constant computed in fp32, to match C++ benchmark convention.
    alpha = cutlass.Float32(1.001)

    # Register fragment for results and simple register double-buffering.
    # We compute out values in registers first, then emit a single predicated
    # vector store via the tiled copy.
    frgOut = cute.make_fragment_like(thrOut)
    frgOut.fill(0.0)
    # Initialize previous-value buffer. Using element 0 avoids SSA undef.
    val_prev = frgOut[0]
    if frgPred[0]:
        coord0 = thrCrd[0]
        g0 = coord0[0]
        n0 = coord0[1]
        idx0 = mIdx[(g0,)]
        idx0_i32 = idx0.to(cutlass.Int32)
        a0 = mA[(idx0_i32, n0)]
        val_prev = (a0.to(cutlass.Float32) * alpha).to(gOut.element_type)
    num_vals = cute.size(frgOut.shape)
    # Software pipeline: prefetch next element before storing current.
    for i in cutlass.range(num_vals, unroll_full=True):
        val_next = val_prev
        if (i + 1) < num_vals:
            if frgPred[i + 1]:
                coord1 = thrCrd[i + 1]
                g1 = coord1[0]
                n1 = coord1[1]
                idx1 = mIdx[(g1,)]
                idx1_i32 = idx1.to(cutlass.Int32)
                a1 = mA[(idx1_i32, n1)]
                val_next = (a1.to(cutlass.Float32) * alpha).to(gOut.element_type)
        if frgPred[i]:
            frgOut[i] = val_prev
        val_prev = val_next

    # Vectorized predicated store to GMEM using the tiled copy.
    cute.copy(tiled_copy_out, frgOut, thrOut, pred=frgPred)


@cute.jit
def _cute_row_gather(
    mA: cute.Tensor,  # (M, N)
    mIdx: cute.Tensor,  # (G)
    mOut: cute.Tensor,  # (G, N)
    vn: cutlass.Constexpr = 4,
    thr_n: cutlass.Constexpr = 32,
):
    """Host/JIT portion of gather.

    - Chooses a TV layout that aligns a thread's values along contiguous N for
      coalesced loads/stores: `thr_layout=(4,32)` threads, `val_layout=(1,VN)`.
    - Builds a tiled copy for vectorized, predicated stores.
    - Invokes the kernel with those layouts and the prebuilt tiled copy.

    Note: We do not use cp.async on Ampere for loads because gather addresses
    are indirect (A[idx[g], n]) and not affine per CTA tile.
    """
    # dtype = mA.element_type
    vector_size = vn

    # Adaptive layout is selected on host and passed as a constexpr (thr_n).
    thr_m = 128 // thr_n

    thr_layout = cute.make_ordered_layout((thr_m, thr_n), order=(1, 0))  # 128 threads
    val_layout = cute.make_ordered_layout(
        (1, vector_size), order=(1, 0)
    )  # widen along N
    tiler_gn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # Tile output and coordinates for predication and absolute addressing.
    gOut = cute.zipped_divide(mOut, tiler_gn)
    idOut = cute.make_identity_tensor(mOut.shape)
    cOut = cute.zipped_divide(idOut, tiler=tiler_gn)

    # Build store atom and tiled copy once on host side.
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gOut.element_type
    )
    tiled_copy_out = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

    _cute_row_gather_kernel(
        mA, mIdx, gOut, cOut, mOut.shape, tv_layout, tiler_gn, tiled_copy_out
    ).launch(
        grid=[cute.size(gOut, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def benchmark_matrix_gathering_cute(
    input_matrix: torch.Tensor,
    gather_indices: torch.Tensor,
    iterations: int = 100,
    warmup_iterations: int = 10,
    return_output: bool = False,
) -> Tuple[List[float], torch.Tensor | None]:
    """Run CuTe row-gather benchmark and return per-iteration times (ms) and optional output.

    Args:
        input_matrix: torch.Tensor [M, N], CUDA, contiguous
        gather_indices: torch.LongTensor [G], CUDA, contiguous
        iterations: measurement iterations
        warmup_iterations: warmup iterations
        return_output: whether to return the gathered output tensor
    """
    assert input_matrix.is_cuda and gather_indices.is_cuda, (
        "Inputs must be CUDA tensors"
    )

    M, N = input_matrix.shape
    G = gather_indices.shape[0]

    # Allocate output
    output = torch.zeros(G, N, device=input_matrix.device, dtype=input_matrix.dtype)

    # Convert to CuTe tensors
    cute_in = from_dlpack(input_matrix).mark_layout_dynamic()
    cute_idx = from_dlpack(gather_indices).mark_layout_dynamic()
    cute_out = from_dlpack(output).mark_layout_dynamic()

    # Select vn (elements per 128-bit vector) and thr_n on host for adaptive layout
    element_bits = input_matrix.element_size() * 8  # infer from dtype
    vector_size = max(1, 128 // element_bits)
    thr_n = 1
    if 32 * vector_size <= N:
        thr_n = 32
    elif 16 * vector_size <= N:
        thr_n = 16
    elif 8 * vector_size <= N:
        thr_n = 8
    elif 4 * vector_size <= N:
        thr_n = 4
    elif 2 * vector_size <= N:
        thr_n = 2

    # Compile once with constexpr (vn, thr_n) baked into the kernel
    compiled = cute.compile(
        _cute_row_gather, cute_in, cute_idx, cute_out, vector_size, thr_n
    )

    # Warmup
    for _ in range(warmup_iterations):
        compiled(cute_in, cute_idx, cute_out)
    torch.cuda.synchronize()

    # Timed runs
    times_ms: List[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compiled(cute_in, cute_idx, cute_out)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return times_ms, (output if return_output else None)


# -------------------------
# Simple sanity test helpers
# -------------------------
@cute.kernel
def _dump_coords_kernel(
    gOut: cute.Tensor,  # ((TileG, TileN),(RestG,RestN)) of dummy output
    cOut: cute.Tensor,  # ((TileG, TileN),(RestG,RestN)) identity coords
    gGAbs: cute.Tensor,  # ((TileG, TileN),(RestG,RestN)) int32
    gNAbs: cute.Tensor,  # ((TileG, TileN),(RestG,RestN)) int32
    out_shape: cute.Shape,
    tv_layout: cute.Layout,
    tiler_gn: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    # blkOut = gOut[blk_coord]
    blkCrd = cOut[blk_coord]
    blkG = gGAbs[blk_coord]
    blkN = gNAbs[blk_coord]

    tidfrgCrd = cute.composition(blkCrd, tv_layout)
    tidfrgG = cute.composition(blkG, tv_layout)
    tidfrgN = cute.composition(blkN, tv_layout)

    thr_coord = (tidx, None)
    thrCrd = tidfrgCrd[thr_coord]
    thrG = tidfrgG[thr_coord]
    thrN = tidfrgN[thr_coord]

    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
    for i in cutlass.range(cute.size(frgPred), unroll_full=True):
        frgPred[i] = cute.elem_less(thrCrd[i], out_shape)

    for i in cutlass.range(cute.size(thrCrd.shape), unroll_full=True):
        if frgPred[i]:
            coord = thrCrd[i]
            thrG[i] = coord[0].to(cutlass.Int32)
            thrN[i] = coord[1].to(cutlass.Int32)


@cute.jit
def dump_coords(mOut: cute.Tensor, gGAbs: cute.Tensor, gNAbs: cute.Tensor):
    # Use same TV layout as gather
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 1), order=(1, 0))
    tiler_gn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gOut = cute.zipped_divide(mOut, tiler_gn)
    idOut = cute.make_identity_tensor(mOut.shape)
    cOut = cute.zipped_divide(idOut, tiler=tiler_gn)
    gG = cute.zipped_divide(gGAbs, tiler_gn)
    gN = cute.zipped_divide(gNAbs, tiler_gn)

    _dump_coords_kernel(gOut, cOut, gG, gN, mOut.shape, tv_layout, tiler_gn).launch(
        grid=[cute.size(gOut, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def sanity_check_cute_gather(verbose: bool = True) -> bool:
    torch.manual_seed(0)
    M, N, G = 16, 32, 7
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    # Ensure unique, sorted indices for testing
    idx = torch.unique(torch.randint(0, M, (G,), device="cuda"), sorted=True)
    G = idx.shape[0]
    out_times, out = benchmark_matrix_gathering_cute(
        a, idx, iterations=1, warmup_iterations=1, return_output=True
    )
    ref = a[idx, :] * 1.001

    ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-6)
    if verbose:
        print(f"[sanity] gather correctness: {ok}")
        if not ok:
            max_diff = (out - ref).abs().max().item()
            print(f"[sanity] max abs diff: {max_diff:.6e}")
            print("[sanity] out[0,:8]   =", out[0, :8].tolist())
            print("[sanity] ref[0,:8]   =", ref[0, :8].tolist())
    return ok


def sanity_check_coords(verbose: bool = True) -> bool:
    G, N = 5, 12
    dummy_out = torch.zeros(G, N, device="cuda", dtype=torch.float32)
    g_abs = torch.full((G, N), -1, device="cuda", dtype=torch.int32)
    n_abs = torch.full((G, N), -1, device="cuda", dtype=torch.int32)

    # Run coordinate dump
    compiled = cute.compile(
        dump_coords,
        from_dlpack(dummy_out).mark_layout_dynamic(),
        from_dlpack(g_abs),
        from_dlpack(n_abs),
    )
    compiled(
        from_dlpack(dummy_out).mark_layout_dynamic(),
        from_dlpack(g_abs),
        from_dlpack(n_abs),
    )

    # Expected
    exp_g = torch.arange(G, device="cuda", dtype=torch.int32).unsqueeze(1).expand(G, N)
    exp_n = torch.arange(N, device="cuda", dtype=torch.int32).unsqueeze(0).expand(G, N)

    ok = torch.equal(g_abs, exp_g) and torch.equal(n_abs, exp_n)
    if verbose:
        print(f"[sanity] coords correctness: {ok}")
        if not ok:
            print("[sanity] g_abs (got):\n", g_abs[:G, : min(N, 8)].cpu())
            print("[sanity] exp_g (exp):\n", exp_g[:G, : min(N, 8)].cpu())
            print("[sanity] n_abs (got):\n", n_abs[:G, : min(N, 8)].cpu())
            print("[sanity] exp_n (exp):\n", exp_n[:G, : min(N, 8)].cpu())
    return ok


if __name__ == "__main__":
    ok_coords = sanity_check_coords(verbose=True)
    ok_gather = sanity_check_cute_gather(verbose=True)
    print("sanity check: ", "PASS" if (ok_coords and ok_gather) else "FAIL")
