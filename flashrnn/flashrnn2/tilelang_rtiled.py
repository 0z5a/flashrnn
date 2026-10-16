"""TileLang R tiling across a thread-block cluster (H100, sm_90a): M3 spike.

Same split as ``triton_rtiled``: CTA ``s`` of a cluster keeps the four gate
slices R[g, head, s*S:(s+1)*S, :] in shared memory and its c/n/m slices in
fragments. The hidden operand is exchanged through distributed shared memory:
every CTA writes its rounded h slice into the next-step buffer of every CTA in
the cluster (``copy_cluster``), then one cluster barrier (release/acquire)
orders the step. Double-buffered h needs one barrier per step.

Forward only, FlashRNN generic LSTM/sLSTM semantics, FP16/BF16, k <= 16.
"""
import tilelang
import tilelang.language as T


def build(batch, steps, heads, width, slice_width, *, cell="lstm", dtype="float16", threads=128, bm=16):
    if width % slice_width:
        raise ValueError("spike requires D divisible by the slice width")
    k = width // slice_width
    if not 1 <= k <= 16:
        raise ValueError("cluster path needs 1 <= k <= 16")
    S, D, H, B, BM = slice_width, width, heads, batch, bm
    groups = (B + BM - 1) // BM * H
    slstm = cell == "slstm"
    states = 4 if slstm else 2
    acc = "float32"

    @T.prim_func
    def main(
        WX: T.Tensor((B, steps, 4, H, D), dtype),
        R: T.Tensor((4, H, D, D), dtype),
        BIAS: T.Tensor((4, H, D), dtype),
        INIT: T.Tensor((states, B, 1, H, D), acc),
        ZERO: T.Tensor((1,), "int32"),
        OUT: T.Tensor((B, steps, H, D), dtype),
        FINAL: T.Tensor((states, B, 1, H, D), acc),
    ):
        with T.ClusterKernel(k, groups, cluster_dims=(k, 1, 1), threads=threads) as (s, g):
            bt = g // H
            head = g % H
            ri = T.alloc_shared((D, S), dtype)
            rf = T.alloc_shared((D, S), dtype)
            rz = T.alloc_shared((D, S), dtype)
            ro = T.alloc_shared((D, S), dtype)
            h0 = T.alloc_shared((BM, D), dtype)
            h1 = T.alloc_shared((BM, D), dtype)
            mine = T.alloc_shared((BM, S), dtype)
            ai = T.alloc_fragment((BM, S), acc)
            af = T.alloc_fragment((BM, S), acc)
            az = T.alloc_fragment((BM, S), acc)
            ao = T.alloc_fragment((BM, S), acc)
            c = T.alloc_fragment((BM, S), acc)
            n = T.alloc_fragment((BM, S), acc)
            m = T.alloc_fragment((BM, S), acc)
            hid = T.alloc_fragment((BM, S), acc)

            for kk, j in T.Parallel(D, S):
                ri[kk, j] = R[0, head, s * S + j, kk]
                rf[kk, j] = R[1, head, s * S + j, kk]
                rz[kk, j] = R[2, head, s * S + j, kk]
                ro[kk, j] = R[3, head, s * S + j, kk]
            for i, kk in T.Parallel(BM, D):
                h0[i, kk] = T.if_then_else(bt * BM + i < B, T.cast(INIT[0, bt * BM + i, 0, head, kk], dtype),
                                           T.cast(0, dtype))
            for i, j in T.Parallel(BM, S):
                valid = bt * BM + i < B
                c[i, j] = T.if_then_else(valid, INIT[1, bt * BM + i, 0, head, s * S + j], 0.0)
                hid[i, j] = 0.0
                if slstm:
                    n[i, j] = T.if_then_else(valid, INIT[2, bt * BM + i, 0, head, s * S + j], 0.0)
                    m[i, j] = T.if_then_else(valid, INIT[3, bt * BM + i, 0, head, s * S + j], 0.0)
            T.cluster_sync()

            for t in T.serial(steps):
                if t % 2 == 0:
                    T.gemm(h0, ri, ai, clear_accum=True)
                    T.gemm(h0, rf, af, clear_accum=True)
                    T.gemm(h0, rz, az, clear_accum=True)
                    T.gemm(h0, ro, ao, clear_accum=True)
                else:
                    T.gemm(h1, ri, ai, clear_accum=True)
                    T.gemm(h1, rf, af, clear_accum=True)
                    T.gemm(h1, rz, az, clear_accum=True)
                    T.gemm(h1, ro, ao, clear_accum=True)
                for i, j in T.Parallel(BM, S):
                    row = bt * BM + i
                    col = s * S + j
                    valid = row < B
                    xi = T.if_then_else(valid, T.cast(WX[row, t, 0, head, col], acc), 0.0)
                    xf = T.if_then_else(valid, T.cast(WX[row, t, 1, head, col], acc), 0.0)
                    xz = T.if_then_else(valid, T.cast(WX[row, t, 2, head, col], acc), 0.0)
                    xo = T.if_then_else(valid, T.cast(WX[row, t, 3, head, col], acc), 0.0)
                    gi = ai[i, j] + xi + T.cast(BIAS[0, head, col], acc)
                    gf = af[i, j] + xf + T.cast(BIAS[1, head, col], acc)
                    gz = az[i, j] + xz + T.cast(BIAS[2, head, col], acc)
                    go = ao[i, j] + xo + T.cast(BIAS[3, head, col], acc)
                    zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                    og = 1.0 / (1.0 + T.exp(-go))
                    if slstm:
                        logf = T.min(gf, 0.0) - T.log(1.0 + T.exp(-T.abs(gf))) + m[i, j]
                        nm = T.if_then_else((t == 0) & (ZERO[0] != 0), gi, T.max(gi, logf))
                        ig = T.exp(gi - nm)
                        fg = T.exp(logf - nm)
                        c[i, j] = fg * c[i, j] + ig * zv
                        n[i, j] = T.max(fg * n[i, j] + ig, 1.0)
                        hid[i, j] = og * c[i, j] / n[i, j]
                        m[i, j] = nm
                    else:
                        c[i, j] = 1.0 / (1.0 + T.exp(-gf)) * c[i, j] + 1.0 / (1.0 + T.exp(-gi)) * zv
                        hid[i, j] = og * (2.0 / (1.0 + T.exp(-2.0 * c[i, j])) - 1.0)
                    mine[i, j] = T.cast(hid[i, j], dtype)
                    if valid:
                        OUT[row, t, head, col] = T.cast(hid[i, j], dtype)
                if t + 1 < steps:
                    for r in T.serial(k):
                        if t % 2 == 0:
                            T.copy_cluster(mine, h1[0:BM, s * S:(s + 1) * S], dst_block=r)
                        else:
                            T.copy_cluster(mine, h0[0:BM, s * S:(s + 1) * S], dst_block=r)
                    T.cluster_sync()

            for i, j in T.Parallel(BM, S):
                row = bt * BM + i
                col = s * S + j
                if row < B:
                    FINAL[0, row, 0, head, col] = hid[i, j]
                    FINAL[1, row, 0, head, col] = c[i, j]
                    if slstm:
                        FINAL[2, row, 0, head, col] = n[i, j]
                        FINAL[3, row, 0, head, col] = m[i, j]

    return main


def compile_kernel(batch, steps, heads, width, slice_width, *, cell="lstm", dtype="float16", threads=128,
                   target=None):
    target = target or {"kind": "cuda", "arch": "sm_90a"}
    return tilelang.compile(build(batch, steps, heads, width, slice_width, cell=cell, dtype=dtype,
                                  threads=threads), target=target)


_KERNELS = {}


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, slice_width=64, threads=128):
    """Spike wrapper: one full logical call from step 0.

    Returns hidden history [B,T,H,D] in the input dtype and the FP32 compute
    carry [S,B,1,H,D]. No chunk continuation or full-state output yet.
    """
    import torch

    batch, steps, gates, heads, width = wx.shape
    if gates != 4 or wx.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("spike supports FP16/BF16 LSTM/sLSTM inputs")
    dtype = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[wx.dtype]
    key = (batch, steps, heads, width, slice_width, cell, dtype, threads)
    if key not in _KERNELS:
        _KERNELS[key] = compile_kernel(batch, steps, heads, width, slice_width, cell=cell, dtype=dtype,
                                       threads=threads)
    states = 4 if cell == "slstm" else 2
    if cell == "slstm":
        zero = torch.all(initial[2] == 0).to(torch.int32).reshape(1)
    else:
        zero = torch.zeros(1, dtype=torch.int32, device=wx.device)
    output = torch.empty((batch, steps, heads, width), dtype=wx.dtype, device=wx.device)
    final = torch.empty((states, batch, 1, heads, width), dtype=torch.float32, device=wx.device)
    _KERNELS[key](wx.contiguous(), recurrent.contiguous(), bias.contiguous(), initial.float().contiguous(), zero,
                  output, final)
    return output, final


def build_v2(batch, steps, heads, width, slice_width, *, cell="lstm", dtype="float16", threads=128, bm=16,
             exchange="bulk"):
    """Same algorithm as tk_rtiled.cu: gates = R_slice[4S x D] @ h^T[D x 16] (weights as operand A), h slices
    exchanged by asynchronous SM-to-SM copies that complete on the receiver's mbarrier (no per-step cluster
    barrier). Double-buffered h^T; one mbarrier per buffer. Barrier arming follows TileLang's copy_cluster
    convention (each sender arrives and announces its bytes); the ThunderKittens kernel arms on the receiver."""
    if width % slice_width:
        raise ValueError("D must be divisible by the slice width")
    k = width // slice_width
    if not 2 <= k <= 16:
        raise ValueError("cluster path needs 2 <= k <= 16")
    S, D, H, B, BM = slice_width, width, heads, batch, bm
    groups = (B + BM - 1) // BM * H
    slstm = cell == "slstm"
    states = 4 if slstm else 2
    acc = "float32"

    @T.prim_func
    def main(
        WX: T.Tensor((B, steps, 4, H, D), dtype),
        R: T.Tensor((4, H, D, D), dtype),
        BIAS: T.Tensor((4, H, D), dtype),
        INIT: T.Tensor((states, B, 1, H, D), acc),
        ZERO: T.Tensor((1,), "int32"),
        OUT: T.Tensor((B, steps, H, D), dtype),
        FINAL: T.Tensor((states, B, 1, H, D), acc),
    ):
        with T.ClusterKernel(k, groups, cluster_dims=(k, 1, 1), threads=threads) as (s, g):
            bt = g // H
            head = g % H
            rs = T.alloc_shared((4 * S, D), dtype)
            h0 = T.alloc_shared((D, BM), dtype)
            h1 = T.alloc_shared((D, BM), dtype)
            gates = T.alloc_fragment((4 * S, BM), acc)
            stage = T.alloc_shared((4 * S, BM), acc)
            # tma_store_cluster arrives on the receiver's barrier and arms its byte count itself (one remote
            # mbarrier.arrive.expect_tx per copy), so each phase completes after the k-1 senders' copies land.
            ready = T.alloc_cluster_barrier([k - 1, k - 1])
            # (batch, unit) fragments: consecutive threads take consecutive units, so Wx/OUT accesses coalesce.
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
                hid[b, j] = 0.0
                if slstm:
                    n[b, j] = T.if_then_else(valid, INIT[2, bt * BM + b, 0, head, s * S + j], 1.0)
                    m[b, j] = T.if_then_else(valid, INIT[3, bt * BM + b, 0, head, s * S + j], 0.0)
            tx = T.get_thread_binding()
            # One arrival per phase: the barriers were initialized with an arrive count of 1.
            T.fence_proxy_async()
            T.cluster_sync()

            for t in T.serial(steps):
                if exchange == "bulk":
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
                    gi = stage[j, b] + T.if_then_else(valid, T.cast(WX[row, t, 0, head, col], acc), 0.0) \
                        + T.cast(BIAS[0, head, col], acc)
                    gf = stage[S + j, b] + T.if_then_else(valid, T.cast(WX[row, t, 1, head, col], acc), 0.0) \
                        + T.cast(BIAS[1, head, col], acc)
                    gz = stage[2 * S + j, b] + T.if_then_else(valid, T.cast(WX[row, t, 2, head, col], acc), 0.0) \
                        + T.cast(BIAS[2, head, col], acc)
                    go = stage[3 * S + j, b] + T.if_then_else(valid, T.cast(WX[row, t, 3, head, col], acc), 0.0) \
                        + T.cast(BIAS[3, head, col], acc)
                    zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                    og = 1.0 / (1.0 + T.exp(-go))
                    if slstm:
                        logf = T.min(gf, 0.0) - T.log(1.0 + T.exp(-T.abs(gf))) + m[b, j]
                        nm = T.if_then_else((t == 0) & (ZERO[0] != 0), gi, T.max(gi, logf))
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
                        OUT[row, t, head, col] = T.cast(hid[b, j], dtype)
                    if t + 1 < steps:
                        if t % 2 == 0:
                            h1[col, b] = T.cast(hid[b, j], dtype)
                        else:
                            h0[col, b] = T.cast(hid[b, j], dtype)
                if exchange == "sync" and t + 1 < steps:
                    # Debug variant: synchronous SM-to-SM stores and one cluster barrier per step.
                    for r in T.serial(k):
                        if r != s:
                            if t % 2 == 0:
                                T.copy_cluster(h1[s * S:(s + 1) * S, 0:BM], h1[s * S:(s + 1) * S, 0:BM], dst_block=r)
                            else:
                                T.copy_cluster(h0[s * S:(s + 1) * S, 0:BM], h0[s * S:(s + 1) * S, 0:BM], dst_block=r)
                    T.cluster_sync()
                if exchange == "bulk" and t + 1 < steps:
                    # Own slice written by all threads -> async-proxy fence -> one block barrier -> thread 0
                    # issues the k-1 bulk copies (same order as the ThunderKittens kernel).
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


_KERNELS_V2 = {}


def recurrence_v2(wx, recurrent, bias, initial, cell="lstm", *, slice_width=32, threads=128, exchange="bulk"):
    """Bulk-DSMEM TileLang path (same algorithm as the ThunderKittens kernel); same contract as recurrence()."""
    import torch

    batch, steps, gates, heads, width = wx.shape
    dtype = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[wx.dtype]
    key = (batch, steps, heads, width, slice_width, cell, dtype, threads, exchange)
    if key not in _KERNELS_V2:
        _KERNELS_V2[key] = tilelang.compile(build_v2(batch, steps, heads, width, slice_width, cell=cell, dtype=dtype,
                                                     threads=threads, exchange=exchange),
                                            target={"kind": "cuda", "arch": "sm_90a"})
    states = 4 if cell == "slstm" else 2
    zero = (torch.all(initial[2] == 0) if cell == "slstm"
            else torch.zeros((), dtype=torch.bool, device=wx.device)).to(torch.int32).reshape(1)
    output = torch.empty((batch, steps, heads, width), dtype=wx.dtype, device=wx.device)
    final = torch.empty((states, batch, 1, heads, width), dtype=torch.float32, device=wx.device)
    _KERNELS_V2[key](wx.contiguous(), recurrent.contiguous(), bias.contiguous(), initial.float().contiguous(), zero,
                     output, final)
    return output, final
