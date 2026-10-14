"""TileLang thread-block-cluster R tiling with the full PR #79 contract (H100, sm_90a).

Algorithm (M5): a cluster of k = D/S CTAs per (batch tile, head). CTA s keeps the
four gate slices R[g, head, s*S:(s+1)*S, :] in shared memory and its c/n/m slices
in fragments. Per step, WGMMA computes R_slice[4S x D] @ h^T[D x BM] (weights as
operand A); each CTA writes its rounded h slice into its own next-step h^T buffer
and copies it into every peer's buffer with cp.async.bulk.shared::cluster, which
completes on the receiver's mbarrier (TileLang's copy_cluster arms that barrier
from the sender side). Double buffering plus the data dependency order the step.

Contract (same as triton_rtiled): FP16/BF16 inputs; FP32 compute carry; full
([S,B,T,H,D] states), hidden ([B,T,H,D]) or last ([B,1,H,D]) output in the input
dtype or FP32; chunked continuation through the FP32 carry with the sLSTM
normalizer predicate applied only at the logical call's first step.

TileLang is optional: it is imported only when this module is used.
"""

_KERNELS = {}
SLICE_WIDTHS = (16, 32, 64)
BATCH_TILES = (16, 32, 64, 128)  # WGMMA N: rows of the batch handled by one CTA per step
MODES = {"full": 0, "hidden": 1, "last": 2}
SMEM_LIMIT = 232448  # H100 opt-in dynamic shared memory per block


def cluster_smem(width, slice_width, bm=16):
    """R slice + double-buffered h^T + FP32 gate stage."""
    return 4 * slice_width * width * 2 + 2 * width * bm * 2 + 4 * slice_width * bm * 4


CLUSTER_BATCH_TILES = (16, 32, 64)


def supported(width, slice_width, bm=16):
    """Cluster size 2..16 and the shared-memory bound (no BM = 16 shape lies between the M5 bound of
    200 KiB and the hardware limit, so the qualified BM = 16 set is unchanged).

    BM <= 64: h^T is the WGMMA B operand, whose swizzled shared layout stores 64-column atoms one after
    another. Only within one atom are a CTA's S rows contiguous, which the single bulk copy per peer needs;
    TileLang 0.1.15 splits a narrower region into per-row copies at swizzled addresses (job 47306925, BM = 128:
    nondeterministic, wrong values), so BM = 128 is excluded here.
    """
    if slice_width not in SLICE_WIDTHS or width % slice_width or bm not in CLUSTER_BATCH_TILES:
        return False
    k = width // slice_width
    return 2 <= k <= 16 and cluster_smem(width, slice_width, bm) <= SMEM_LIMIT


def build(batch, total_steps, heads, width, slice_width, *, cell="lstm", dtype="float16", mode="full",
          snapshot="float32", threads=128, bm=16):
    import tilelang.language as T

    if not supported(width, slice_width, bm):
        raise ValueError("unsupported (D, S, BM) for the cluster path")
    k = width // slice_width
    S, D, H, B, BM, TT = slice_width, width, heads, batch, bm, total_steps
    groups = (B + BM - 1) // BM * H
    slstm = cell == "slstm"
    states = 4 if slstm else 2
    acc = "float32"
    out_mode = MODES[mode]
    out_shape = {0: (states, B, TT, H, D), 1: (B, TT, H, D), 2: (B, 1, H, D)}[out_mode]

    @T.prim_func
    def main(
        WX: T.Tensor((B, TT, 4, H, D), dtype),
        R: T.Tensor((4, H, D, D), dtype),
        BIAS: T.Tensor((4, H, D), dtype),
        INIT: T.Tensor((states, B, 1, H, D), acc),
        ZERO: T.Tensor((1,), "int32"),
        OUT: T.Tensor(out_shape, snapshot),
        FINAL: T.Tensor((states, B, 1, H, D), acc),
        NSTEPS: T.int32,
        BASE: T.int32,
        START: T.int32,
    ):
        with T.ClusterKernel(k, groups, cluster_dims=(k, 1, 1), threads=threads) as (s, g):
            bt = g // H
            head = g % H
            rs = T.alloc_shared((4 * S, D), dtype)
            h0 = T.alloc_shared((D, BM), dtype)
            h1 = T.alloc_shared((D, BM), dtype)
            gates = T.alloc_fragment((4 * S, BM), acc)
            stage = T.alloc_shared((4 * S, BM), acc)
            ready = T.alloc_cluster_barrier([k - 1, k - 1])
            c = T.alloc_fragment((BM, S), acc)
            n = T.alloc_fragment((BM, S), acc)
            m = T.alloc_fragment((BM, S), acc)
            hid = T.alloc_fragment((BM, S), acc)

            for q, kk in T.Parallel(4 * S, D):
                rs[q, kk] = R[q // S, head, s * S + q % S, kk]
            for kk, b in T.Parallel(D, BM):
                h0[kk, b] = T.if_then_else(bt * BM + b < B, T.cast(INIT[0, bt * BM + b, 0, head, kk], dtype),
                                           T.cast(0, dtype))
            for b, j in T.Parallel(BM, S):
                valid = bt * BM + b < B
                c[b, j] = T.if_then_else(valid, INIT[1, bt * BM + b, 0, head, s * S + j], 0.0)
                hid[b, j] = T.if_then_else(valid, INIT[0, bt * BM + b, 0, head, s * S + j], 0.0)
                if slstm:
                    n[b, j] = T.if_then_else(valid, INIT[2, bt * BM + b, 0, head, s * S + j], 1.0)
                    m[b, j] = T.if_then_else(valid, INIT[3, bt * BM + b, 0, head, s * S + j], 0.0)
            tx = T.get_thread_binding()
            T.fence_proxy_async()
            T.cluster_sync()

            for t in T.serial(NSTEPS):
                if t > 0:
                    T.mbarrier_wait_parity(ready[t % 2], ((t - 1) // 2) % 2)
                if t % 2 == 0:
                    T.gemm(rs, h0, gates, clear_accum=True)
                else:
                    T.gemm(rs, h1, gates, clear_accum=True)
                T.copy(gates, stage)
                for b, j in T.Parallel(BM, S):
                    row = bt * BM + b
                    col = s * S + j
                    valid = row < B
                    step = BASE + t
                    gi = stage[j, b] + T.if_then_else(valid, T.cast(WX[row, step, 0, head, col], acc), 0.0) \
                        + T.cast(BIAS[0, head, col], acc)
                    gf = stage[S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 1, head, col], acc), 0.0) \
                        + T.cast(BIAS[1, head, col], acc)
                    gz = stage[2 * S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 2, head, col], acc), 0.0) \
                        + T.cast(BIAS[2, head, col], acc)
                    go = stage[3 * S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 3, head, col], acc), 0.0) \
                        + T.cast(BIAS[3, head, col], acc)
                    zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                    og = 1.0 / (1.0 + T.exp(-go))
                    if slstm:
                        logf = T.min(gf, 0.0) - T.log(1.0 + T.exp(-T.abs(gf))) + m[b, j]
                        nm = T.if_then_else((START != 0) & (t == 0) & (ZERO[0] != 0), gi, T.max(gi, logf))
                        ig = T.exp(gi - nm)
                        fg = T.exp(logf - nm)
                        c[b, j] = fg * c[b, j] + ig * zv
                        n[b, j] = T.max(fg * n[b, j] + ig, 1.0)
                        hid[b, j] = og * c[b, j] / n[b, j]
                        m[b, j] = nm
                    else:
                        c[b, j] = 1.0 / (1.0 + T.exp(-gf)) * c[b, j] + 1.0 / (1.0 + T.exp(-gi)) * zv
                        hid[b, j] = og * (2.0 / (1.0 + T.exp(-2.0 * c[b, j])) - 1.0)
                    if valid:
                        if out_mode == 0:
                            OUT[0, row, step, head, col] = T.cast(hid[b, j], snapshot)
                            OUT[1, row, step, head, col] = T.cast(c[b, j], snapshot)
                            if slstm:
                                OUT[2, row, step, head, col] = T.cast(n[b, j], snapshot)
                                OUT[3, row, step, head, col] = T.cast(m[b, j], snapshot)
                        elif out_mode == 1:
                            OUT[row, step, head, col] = T.cast(hid[b, j], snapshot)
                        else:
                            if step == TT - 1:
                                OUT[row, 0, head, col] = T.cast(hid[b, j], snapshot)
                    if t + 1 < NSTEPS:
                        if t % 2 == 0:
                            h1[col, b] = T.cast(hid[b, j], dtype)
                        else:
                            h0[col, b] = T.cast(hid[b, j], dtype)
                if t + 1 < NSTEPS:
                    # Own slice written by all threads -> async-proxy fence -> block barrier -> thread 0 sends.
                    T.fence_proxy_async()
                    T.sync_threads()
                    if tx == 0:
                        for r in T.serial(k):
                            if r != s:
                                if t % 2 == 0:
                                    T.copy_cluster(h1[s * S:(s + 1) * S, 0:BM], h1[s * S:(s + 1) * S, 0:BM],
                                                   dst_block=r, remote_barrier=ready[1])
                                else:
                                    T.copy_cluster(h0[s * S:(s + 1) * S, 0:BM], h0[s * S:(s + 1) * S, 0:BM],
                                                   dst_block=r, remote_barrier=ready[0])

            for b, j in T.Parallel(BM, S):
                row = bt * BM + b
                col = s * S + j
                if row < B:
                    FINAL[0, row, 0, head, col] = hid[b, j]
                    FINAL[1, row, 0, head, col] = c[b, j]
                    if slstm:
                        FINAL[2, row, 0, head, col] = n[b, j]
                        FINAL[3, row, 0, head, col] = m[b, j]
            T.cluster_sync()

    return main


GRID_SLICE_WIDTHS = (16, 32)
GRID_SMEM_LIMIT = SMEM_LIMIT


def grid_smem(width, slice_width, bm=16):
    return 4 * slice_width * width * 2 + width * bm * 2 + 4 * slice_width * bm * 4


def grid_plan(batch, heads, width, slice_width, sms, bm=16):
    """(k, co-resident groups, slots): every CTA of a cooperative grid must be resident, one per SM."""
    k = width // slice_width
    groups = (batch + bm - 1) // bm * heads
    resident = max(1, min(groups, sms // k))
    return k, resident, (groups + resident - 1) // resident


def supported_grid(width, slice_width, sms=132, bm=16):
    """Grid-synchronized path: any k = D/S up to the SM count, R slice + h^T + stage in shared memory."""
    if slice_width not in GRID_SLICE_WIDTHS or width % slice_width or bm not in BATCH_TILES:
        return False
    return 2 <= width // slice_width <= sms and grid_smem(width, slice_width, bm) <= GRID_SMEM_LIMIT


def build_grid(batch, total_steps, heads, width, slice_width, *, sms, cell="lstm", dtype="float16", mode="full",
               snapshot="float32", threads=128, bm=16):
    """Same per-CTA work as build(), for k beyond the 16-CTA cluster limit (D = 768).

    The h exchange goes through a double-buffered global buffer XBUF[2, groups, D, BM]:
    each CTA writes its rounded h slice into XBUF[t % 2], one cooperative grid barrier
    (T.sync_grid) orders the step, and every CTA loads the full h^T of its group for the
    next step. A CTA can only overwrite XBUF[t % 2] again at step t + 2, after the barrier
    of step t + 1 that every reader of step t has reached. Groups beyond the co-resident
    count run in slots; every CTA executes the same barriers, idle or not.
    """
    import tilelang.language as T

    if not supported_grid(width, slice_width, sms, bm):
        raise ValueError("unsupported (D, S, BM) for the grid path")
    S, D, H, B, BM, TT = slice_width, width, heads, batch, bm, total_steps
    k, resident, slots = grid_plan(B, H, D, S, sms, BM)
    tiles = (B + BM - 1) // BM
    groups = tiles * H
    slstm = cell == "slstm"
    states = 4 if slstm else 2
    acc = "float32"
    out_mode = MODES[mode]
    out_shape = {0: (states, B, TT, H, D), 1: (B, TT, H, D), 2: (B, 1, H, D)}[out_mode]

    @T.prim_func
    def main(
        WX: T.Tensor((B, TT, 4, H, D), dtype),
        R: T.Tensor((4, H, D, D), dtype),
        BIAS: T.Tensor((4, H, D), dtype),
        INIT: T.Tensor((states, B, 1, H, D), acc),
        ZERO: T.Tensor((1,), "int32"),
        OUT: T.Tensor(out_shape, snapshot),
        FINAL: T.Tensor((states, B, 1, H, D), acc),
        XBUF: T.Tensor((2, groups, D, BM), dtype),
        NSTEPS: T.int32,
        BASE: T.int32,
        START: T.int32,
    ):
        with T.Kernel(k, resident, threads=threads) as (s, gy):
            rs = T.alloc_shared((4 * S, D), dtype)
            hs = T.alloc_shared((D, BM), dtype)
            gates = T.alloc_fragment((4 * S, BM), acc)
            stage = T.alloc_shared((4 * S, BM), acc)
            c = T.alloc_fragment((BM, S), acc)
            n = T.alloc_fragment((BM, S), acc)
            m = T.alloc_fragment((BM, S), acc)
            hid = T.alloc_fragment((BM, S), acc)

            for slot in T.serial(slots):
                g = slot * resident + gy
                active = g < groups
                head = T.min(g, groups - 1) // tiles
                bt = T.min(g, groups - 1) % tiles
                if active:
                    for q, kk in T.Parallel(4 * S, D):
                        rs[q, kk] = R[q // S, head, s * S + q % S, kk]
                    for kk, b in T.Parallel(D, BM):
                        hs[kk, b] = T.if_then_else(bt * BM + b < B,
                                                   T.cast(INIT[0, bt * BM + b, 0, head, kk], dtype),
                                                   T.cast(0, dtype))
                    for b, j in T.Parallel(BM, S):
                        valid = bt * BM + b < B
                        c[b, j] = T.if_then_else(valid, INIT[1, bt * BM + b, 0, head, s * S + j], 0.0)
                        hid[b, j] = T.if_then_else(valid, INIT[0, bt * BM + b, 0, head, s * S + j], 0.0)
                        if slstm:
                            n[b, j] = T.if_then_else(valid, INIT[2, bt * BM + b, 0, head, s * S + j], 1.0)
                            m[b, j] = T.if_then_else(valid, INIT[3, bt * BM + b, 0, head, s * S + j], 0.0)
                T.fence_proxy_async()
                T.sync_threads()

                for t in T.serial(NSTEPS):
                    if active:
                        if t > 0:
                            T.copy(XBUF[(t - 1) % 2, g, 0:D, 0:BM], hs)
                            T.fence_proxy_async()
                            T.sync_threads()
                        T.gemm(rs, hs, gates, clear_accum=True)
                        T.copy(gates, stage)
                        for b, j in T.Parallel(BM, S):
                            row = bt * BM + b
                            col = s * S + j
                            valid = row < B
                            step = BASE + t
                            gi = stage[j, b] + T.if_then_else(valid, T.cast(WX[row, step, 0, head, col], acc), 0.0) \
                                + T.cast(BIAS[0, head, col], acc)
                            gf = stage[S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 1, head, col], acc),
                                                                  0.0) + T.cast(BIAS[1, head, col], acc)
                            gz = stage[2 * S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 2, head, col], acc),
                                                                      0.0) + T.cast(BIAS[2, head, col], acc)
                            go = stage[3 * S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 3, head, col], acc),
                                                                      0.0) + T.cast(BIAS[3, head, col], acc)
                            zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                            og = 1.0 / (1.0 + T.exp(-go))
                            if slstm:
                                logf = T.min(gf, 0.0) - T.log(1.0 + T.exp(-T.abs(gf))) + m[b, j]
                                nm = T.if_then_else((START != 0) & (t == 0) & (ZERO[0] != 0), gi, T.max(gi, logf))
                                ig = T.exp(gi - nm)
                                fg = T.exp(logf - nm)
                                c[b, j] = fg * c[b, j] + ig * zv
                                n[b, j] = T.max(fg * n[b, j] + ig, 1.0)
                                hid[b, j] = og * c[b, j] / n[b, j]
                                m[b, j] = nm
                            else:
                                c[b, j] = 1.0 / (1.0 + T.exp(-gf)) * c[b, j] + 1.0 / (1.0 + T.exp(-gi)) * zv
                                hid[b, j] = og * (2.0 / (1.0 + T.exp(-2.0 * c[b, j])) - 1.0)
                            if valid:
                                if out_mode == 0:
                                    OUT[0, row, step, head, col] = T.cast(hid[b, j], snapshot)
                                    OUT[1, row, step, head, col] = T.cast(c[b, j], snapshot)
                                    if slstm:
                                        OUT[2, row, step, head, col] = T.cast(n[b, j], snapshot)
                                        OUT[3, row, step, head, col] = T.cast(m[b, j], snapshot)
                                elif out_mode == 1:
                                    OUT[row, step, head, col] = T.cast(hid[b, j], snapshot)
                                else:
                                    if step == TT - 1:
                                        OUT[row, 0, head, col] = T.cast(hid[b, j], snapshot)
                            if t + 1 < NSTEPS:
                                XBUF[t % 2, g, col, b] = T.cast(hid[b, j], dtype)
                    if t + 1 < NSTEPS:
                        T.sync_grid()

                if active:
                    for b, j in T.Parallel(BM, S):
                        row = bt * BM + b
                        col = s * S + j
                        if row < B:
                            FINAL[0, row, 0, head, col] = hid[b, j]
                            FINAL[1, row, 0, head, col] = c[b, j]
                            if slstm:
                                FINAL[2, row, 0, head, col] = n[b, j]
                                FINAL[3, row, 0, head, col] = m[b, j]

    return main


def sm_count():
    import torch

    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def kernel(batch, total_steps, heads, width, slice_width, *, cell, dtype, mode, snapshot, sync="cluster", sms=None,
           bm=16, threads=128):
    """Compiled kernel for one logical-call shape; chunk length, offset and start are runtime arguments."""
    import tilelang

    if sync == "grid":
        sms = sms or sm_count()
        key = ("grid", sms, batch, total_steps, heads, width, slice_width, cell, dtype, mode, snapshot)
        func = lambda: build_grid(batch, total_steps, heads, width, slice_width, sms=sms, cell=cell,  # noqa: E731
                                  dtype=dtype, mode=mode, snapshot=snapshot, bm=bm, threads=threads)
    else:
        key = (batch, total_steps, heads, width, slice_width, cell, dtype, mode, snapshot)
        func = lambda: build(batch, total_steps, heads, width, slice_width, cell=cell, dtype=dtype,  # noqa: E731
                             mode=mode, snapshot=snapshot, bm=bm, threads=threads)
    if bm != 16:
        key = key + (("bm", bm),)  # BM = 16 keys stay as they were
    if threads != 128:
        key = key + (("threads", threads),)
    if key not in _KERNELS:
        _KERNELS[key] = tilelang.compile(func(), target={"kind": "cuda", "arch": "sm_90a"})
    return _KERNELS[key]


def version():
    import tilelang

    return tilelang.__version__


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, sequence_context, slice_width=32,
               output_mode="full", snapshot_dtype=None, sync="cluster", batch_tile=16, threads=128):
    """Public per-call API with the triton_rtiled contract; returns (output, FP32 compute carry).

    sync="cluster" exchanges h inside a thread-block cluster (k <= 16); sync="grid" exchanges it
    through global memory under one cooperative grid barrier per step (k up to the SM count).
    batch_tile is the number of batch rows one CTA carries per step (WGMMA N); threads is the CTA size
    (128 = one warpgroup, 256 = two).
    """
    import torch

    from .state_contract import state_contract

    if output_mode not in MODES:
        raise ValueError("output mode must be full, hidden or last")
    if sync not in ("cluster", "grid"):
        raise ValueError("sync must be cluster or grid")
    if cell not in ("lstm", "slstm"):
        raise ValueError("cluster R tiling supports LSTM and sLSTM")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("cluster R tiling requires CUDA FP16/BF16 inputs")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (wx, recurrent, bias, initial)):
        raise ValueError("cluster R tiling has no backward implementation")
    contract = state_contract(cell, wx.dtype)
    if sequence_context is None or sequence_context.contract != contract:
        raise ValueError("cluster R tiling requires a matching full logical-call sequence context")
    if snapshot_dtype not in (None, wx.dtype, torch.float32):
        raise ValueError("snapshot dtype must be input dtype or FP32")
    batch, steps, gates, heads, width = wx.shape
    states = len(contract.components)
    ok = (supported(width, slice_width, batch_tile) if sync == "cluster" else
          supported_grid(width, slice_width, sm_count(), batch_tile))
    if gates != 4 or steps < 1 or batch < 1 or heads < 1 or not ok:
        raise ValueError(f"unsupported shape for the {sync} path")
    if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
        raise ValueError("expected public weight and bias layouts")
    if initial.shape != (states, batch, 1, heads, width) or initial.dtype not in (wx.dtype, torch.float32):
        raise ValueError("expected public initial state layout in input dtype or FP32 carry")
    if sequence_context.step_offset > 0 and initial.dtype != torch.float32:
        raise ValueError("continuation requires compute carry, not a rounded snapshot")
    if recurrent.dtype != wx.dtype or bias.dtype != wx.dtype:
        raise ValueError("weights must match the input dtype")
    names = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
    snap = wx.dtype if snapshot_dtype is None else snapshot_dtype
    fn = kernel(batch, steps, heads, width, slice_width, cell=cell, dtype=names[wx.dtype], mode=output_mode,
                snapshot=names[snap], sync=sync, bm=batch_tile, threads=threads)
    out_shape = {"full": (states, batch, steps, heads, width), "hidden": (batch, steps, heads, width),
                 "last": (batch, 1, heads, width)}[output_mode]
    output = torch.empty(out_shape, device=wx.device, dtype=snap)
    final = torch.empty(initial.shape, device=wx.device, dtype=torch.float32)
    zero = sequence_context.zero_normalizer.to(torch.int32).reshape(1)
    args = [wx.contiguous(), recurrent.contiguous(), bias.contiguous(), initial.float().contiguous(), zero, output, final]
    if sync == "grid":
        tiles = (batch + batch_tile - 1) // batch_tile
        args.append(torch.empty((2, tiles * heads, width, batch_tile), device=wx.device, dtype=wx.dtype))
    fn(*args, steps, 0, int(sequence_context.step_offset == 0))
    return output, final
