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


def _verify_opts(verify):
    """Normalized race-verification request (None when off); see recurrence()."""
    if not verify:
        return None
    v = dict(verify)
    if v.get("broken") not in (None, "single", "no_wait", "no_sync", "no_barrier", "arrive_first", "threads256"):
        raise ValueError("unknown broken variant")
    return dict(check=v.get("check", True), seed=v.get("delay_seed"), broken=v.get("broken"))


def _delay_ns(T, seed, a, b, c, point):
    # Race verification: 0-65 us pseudo-random sleep from the runtime seed (ZERO[3]).
    x = (a * 7919) ^ (b * 104729) ^ (c * 1299709) ^ (seed * 40503 + point * 977)
    return T.cast(((x ^ (x >> 13)) * 48271) & 65535, "uint32")


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
          snapshot="float32", threads=128, bm=16, prefetch=False, _ablate=frozenset(), _verify=None):
    """Cluster kernel (see the module docstring).
    _ablate (timing only, never served): a set of components to remove for a ceiling estimate -- "exchange"
    (h exchange and its synchronization), "gemm" (cleared gates instead of the WGMMA), "stage" (no FP32 gate
    round trip through shared memory), "wx" (no input-projection loads), "epilogue" (cheap arithmetic instead
    of the gate nonlinearities), "out" (no output stores). Results of an ablated kernel are wrong by design.
    _verify (race verification, never served): see recurrence(); counters go to ZERO[1] (received operand vs the
    senders' stored output, after the barrier wait) and ZERO[2] (the same after the GEMM).
    """
    import tilelang.language as T

    ab = frozenset(_ablate)
    if ab and prefetch:
        raise ValueError("ablations apply to the kernel without prefetch")
    v = _verify_opts(_verify)
    if v and (ab or prefetch or mode != "hidden" or snapshot != "float32"):
        raise ValueError("race verification uses the plain kernel with an FP32 hidden output")
    positive_control = bool(v) and v["broken"] == "threads256"
    single = bool(v) and v["broken"] == "single"
    check = bool(v) and v["check"]
    delay = bool(v) and v["seed"] is not None
    if not supported(width, slice_width, bm) or (threads != 128 and not (positive_control and threads == 256)):
        raise ValueError("unsupported (D, S, BM, threads) for the cluster path")
    zshape = (4,) if v else (1,)
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
        ZERO: T.Tensor(zshape, "int32"),
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
            # prefetch: this thread's input projections for the next step, loaded right after the current step's
            # epilogue so their latency overlaps the exchange wait and the GEMM instead of the critical path.
            xi = T.alloc_fragment((BM, S), acc)
            xf = T.alloc_fragment((BM, S), acc)
            xz = T.alloc_fragment((BM, S), acc)
            xo = T.alloc_fragment((BM, S), acc)

            for q, kk in T.Parallel(4 * S, D):
                rs[q, kk] = R[q // S, head, s * S + q % S, kk]
            for kk, b in T.Parallel(D, BM):
                h0[kk, b] = T.if_then_else(bt * BM + b < B, T.cast(INIT[0, bt * BM + b, 0, head, kk], dtype),
                                           T.cast(0, dtype))
            if prefetch:
                for b, j in T.Parallel(BM, S):
                    valid = bt * BM + b < B
                    xi[b, j] = T.if_then_else(valid, T.cast(WX[bt * BM + b, BASE, 0, head, s * S + j], acc), 0.0)
                    xf[b, j] = T.if_then_else(valid, T.cast(WX[bt * BM + b, BASE, 1, head, s * S + j], acc), 0.0)
                    xz[b, j] = T.if_then_else(valid, T.cast(WX[bt * BM + b, BASE, 2, head, s * S + j], acc), 0.0)
                    xo[b, j] = T.if_then_else(valid, T.cast(WX[bt * BM + b, BASE, 3, head, s * S + j], acc), 0.0)
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
                if "exchange" not in ab:
                    if t > 0:
                        if v and v["broken"] == "no_wait":
                            # Negative control: skip the wait, except at the last step so that no copy is still in
                            # flight into this CTA's shared memory when it exits.
                            if t == NSTEPS - 1:
                                T.mbarrier_wait_parity(ready[t % 2], ((t - 1) // 2) % 2)
                        else:
                            T.mbarrier_wait_parity(ready[t % 2], ((t - 1) // 2) % 2)
                if delay:
                    # Consumer side, after the wait: the received buffer stays in use longer (single-buffer window).
                    T.call_extern("handle", "__nanosleep", _delay_ns(T, ZERO[3], s, g, t, 1))
                if check:
                    if t > 0:
                        for kk, b in T.Parallel(D, BM):
                            if bt * BM + b < B:
                                if single:
                                    if h0[kk, b] != T.cast(OUT[bt * BM + b, BASE + t - 1, head, kk], dtype):
                                        T.atomic_add(ZERO[1], 1)
                                elif t % 2 == 0:
                                    if h0[kk, b] != T.cast(OUT[bt * BM + b, BASE + t - 1, head, kk], dtype):
                                        T.atomic_add(ZERO[1], 1)
                                else:
                                    if h1[kk, b] != T.cast(OUT[bt * BM + b, BASE + t - 1, head, kk], dtype):
                                        T.atomic_add(ZERO[1], 1)
                if "gemm" in ab:
                    T.clear(gates)
                elif single:
                    T.gemm(rs, h0, gates, clear_accum=True)
                else:
                    if t % 2 == 0:
                        T.gemm(rs, h0, gates, clear_accum=True)
                    else:
                        T.gemm(rs, h1, gates, clear_accum=True)
                if check:
                    if t > 0:
                        for kk, b in T.Parallel(D, BM):
                            if bt * BM + b < B:
                                if single:
                                    if h0[kk, b] != T.cast(OUT[bt * BM + b, BASE + t - 1, head, kk], dtype):
                                        T.atomic_add(ZERO[2], 1)
                                elif t % 2 == 0:
                                    if h0[kk, b] != T.cast(OUT[bt * BM + b, BASE + t - 1, head, kk], dtype):
                                        T.atomic_add(ZERO[2], 1)
                                else:
                                    if h1[kk, b] != T.cast(OUT[bt * BM + b, BASE + t - 1, head, kk], dtype):
                                        T.atomic_add(ZERO[2], 1)
                if "stage" not in ab:
                    T.copy(gates, stage)
                if delay:
                    # Producer side, skewed per warp: the epilogue's h-slice writes finish at different times.
                    T.call_extern("handle", "__nanosleep", _delay_ns(T, ZERO[3], s * 64 + tx // 32, g, t, 0))
                for b, j in T.Parallel(BM, S):
                    row = bt * BM + b
                    col = s * S + j
                    valid = row < B
                    step = BASE + t
                    if prefetch:
                        gi = stage[j, b] + xi[b, j] + T.cast(BIAS[0, head, col], acc)
                        gf = stage[S + j, b] + xf[b, j] + T.cast(BIAS[1, head, col], acc)
                        gz = stage[2 * S + j, b] + xz[b, j] + T.cast(BIAS[2, head, col], acc)
                        go = stage[3 * S + j, b] + xo[b, j] + T.cast(BIAS[3, head, col], acc)
                        nxt = valid & (t + 1 < NSTEPS)
                        xi[b, j] = T.if_then_else(nxt, T.cast(WX[row, step + 1, 0, head, col], acc), 0.0)
                        xf[b, j] = T.if_then_else(nxt, T.cast(WX[row, step + 1, 1, head, col], acc), 0.0)
                        xz[b, j] = T.if_then_else(nxt, T.cast(WX[row, step + 1, 2, head, col], acc), 0.0)
                        xo[b, j] = T.if_then_else(nxt, T.cast(WX[row, step + 1, 3, head, col], acc), 0.0)
                    elif "wx" in ab:
                        gi = stage[j, b] + T.cast(BIAS[0, head, col], acc)
                        gf = stage[S + j, b] + T.cast(BIAS[1, head, col], acc)
                        gz = stage[2 * S + j, b] + T.cast(BIAS[2, head, col], acc)
                        go = stage[3 * S + j, b] + T.cast(BIAS[3, head, col], acc)
                    else:
                        gi = stage[j, b] + T.if_then_else(valid, T.cast(WX[row, step, 0, head, col], acc), 0.0) \
                            + T.cast(BIAS[0, head, col], acc)
                        gf = stage[S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 1, head, col], acc), 0.0) \
                            + T.cast(BIAS[1, head, col], acc)
                        gz = stage[2 * S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 2, head, col], acc),
                                                                  0.0) + T.cast(BIAS[2, head, col], acc)
                        go = stage[3 * S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 3, head, col], acc),
                                                                  0.0) + T.cast(BIAS[3, head, col], acc)
                    if "epilogue" in ab:
                        c[b, j] = c[b, j] * 0.5 + gi * gf
                        hid[b, j] = c[b, j] + go * gz
                    elif slstm:
                        zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                        og = 1.0 / (1.0 + T.exp(-go))
                        logf = T.min(gf, 0.0) - T.log(1.0 + T.exp(-T.abs(gf))) + m[b, j]
                        nm = T.if_then_else((START != 0) & (t == 0) & (ZERO[0] != 0), gi, T.max(gi, logf))
                        ig = T.exp(gi - nm)
                        fg = T.exp(logf - nm)
                        c[b, j] = fg * c[b, j] + ig * zv
                        n[b, j] = T.max(fg * n[b, j] + ig, 1.0)
                        hid[b, j] = og * c[b, j] / n[b, j]
                        m[b, j] = nm
                    else:
                        zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                        og = 1.0 / (1.0 + T.exp(-go))
                        c[b, j] = 1.0 / (1.0 + T.exp(-gf)) * c[b, j] + 1.0 / (1.0 + T.exp(-gi)) * zv
                        hid[b, j] = og * (2.0 / (1.0 + T.exp(-2.0 * c[b, j])) - 1.0)
                    if "out" not in ab:
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
                        if single:
                            h0[col, b] = T.cast(hid[b, j], dtype)
                        elif t % 2 == 0:
                            h1[col, b] = T.cast(hid[b, j], dtype)
                        else:
                            h0[col, b] = T.cast(hid[b, j], dtype)
                if "exchange" not in ab:
                    if t + 1 < NSTEPS:
                        # Own slice written by all threads -> async-proxy fence -> block barrier -> thread 0 sends.
                        if check:
                            # Verification: make this step's output stores visible before peers check against them.
                            T.call_extern("handle", "__threadfence")
                        T.fence_proxy_async()
                        if not (v and v["broken"] == "no_sync"):
                            T.sync_threads()
                        if tx == 0:
                            for r in T.serial(k):
                                if r != s:
                                    if single:
                                        if t % 2 == 0:
                                            T.copy_cluster(h0[s * S:(s + 1) * S, 0:BM], h0[s * S:(s + 1) * S, 0:BM],
                                                           dst_block=r, remote_barrier=ready[1])
                                        else:
                                            T.copy_cluster(h0[s * S:(s + 1) * S, 0:BM], h0[s * S:(s + 1) * S, 0:BM],
                                                           dst_block=r, remote_barrier=ready[0])
                                    elif t % 2 == 0:
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
               snapshot="float32", threads=128, bm=16, prefetch=False, _ablate=frozenset(), _verify=None):
    """Same per-CTA work as build(), for k beyond the 16-CTA cluster limit (D = 768).

    The h exchange goes through a double-buffered global buffer XBUF[2, groups, D, BM]:
    each CTA writes its rounded h slice into XBUF[t % 2], one cooperative grid barrier
    (T.sync_grid) orders the step, and every CTA loads the full h^T of its group for the
    next step. A CTA can only overwrite XBUF[t % 2] again at step t + 2, after the barrier
    of step t + 1 that every reader of step t has reached. Groups beyond the co-resident
    count run in slots; every CTA executes the same barriers, idle or not.
    """
    import tilelang.language as T

    ab = frozenset(_ablate)
    if ab and prefetch:
        raise ValueError("ablations apply to the kernel without prefetch")
    if not supported_grid(width, slice_width, sms, bm):
        raise ValueError("unsupported (D, S, BM) for the grid path")
    v = _verify_opts(_verify)
    if v and (ab or prefetch or mode != "hidden" or snapshot != "float32" or v["broken"] in ("no_wait", "no_sync",
                                                                                              "threads256")):
        raise ValueError("race verification uses the plain kernel with an FP32 hidden output")
    single = bool(v) and v["broken"] == "single"
    check = bool(v) and v["check"]
    delay = bool(v) and v["seed"] is not None
    no_barrier = bool(v) and v["broken"] == "no_barrier"
    arrive_first = bool(v) and v["broken"] == "arrive_first"
    zshape = (4,) if v else (1,)
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
        ZERO: T.Tensor(zshape, "int32"),
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
            xi = T.alloc_fragment((BM, S), acc)  # prefetched input projections (see build)
            xf = T.alloc_fragment((BM, S), acc)
            xz = T.alloc_fragment((BM, S), acc)
            xo = T.alloc_fragment((BM, S), acc)

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
                        if prefetch:
                            xi[b, j] = T.if_then_else(valid, T.cast(WX[bt * BM + b, BASE, 0, head, s * S + j], acc), 0.0)
                            xf[b, j] = T.if_then_else(valid, T.cast(WX[bt * BM + b, BASE, 1, head, s * S + j], acc), 0.0)
                            xz[b, j] = T.if_then_else(valid, T.cast(WX[bt * BM + b, BASE, 2, head, s * S + j], acc), 0.0)
                            xo[b, j] = T.if_then_else(valid, T.cast(WX[bt * BM + b, BASE, 3, head, s * S + j], acc), 0.0)
                T.fence_proxy_async()
                T.sync_threads()

                for t in T.serial(NSTEPS):
                    if active:
                        if delay:
                            T.call_extern("handle", "__nanosleep", _delay_ns(T, ZERO[3], s, g, t, 1))
                        if "exchange" not in ab:
                            if t > 0:
                                if single:
                                    T.copy(XBUF[0, g, 0:D, 0:BM], hs)
                                else:
                                    T.copy(XBUF[(t - 1) % 2, g, 0:D, 0:BM], hs)
                                T.fence_proxy_async()
                                T.sync_threads()
                        if check:
                            if t > 0:
                                for kk, b in T.Parallel(D, BM):
                                    if bt * BM + b < B:
                                        if hs[kk, b] != T.cast(OUT[bt * BM + b, BASE + t - 1, head, kk], dtype):
                                            T.atomic_add(ZERO[1], 1)
                        if "gemm" in ab:
                            T.clear(gates)
                        else:
                            T.gemm(rs, hs, gates, clear_accum=True)
                        if check:
                            if t > 0:
                                # The exchange buffer must still hold step t - 1 after the GEMM (no early overwrite).
                                for kk, b in T.Parallel(D, BM):
                                    if bt * BM + b < B:
                                        if single:
                                            if XBUF[0, g, kk, b] != T.cast(OUT[bt * BM + b, BASE + t - 1, head, kk],
                                                                           dtype):
                                                T.atomic_add(ZERO[2], 1)
                                        else:
                                            if XBUF[(t - 1) % 2, g, kk, b] != T.cast(
                                                    OUT[bt * BM + b, BASE + t - 1, head, kk], dtype):
                                                T.atomic_add(ZERO[2], 1)
                        if "stage" not in ab:
                            T.copy(gates, stage)
                        if delay:
                            T.call_extern("handle", "__nanosleep",
                                          _delay_ns(T, ZERO[3], s * 64 + T.get_thread_binding() // 32, g, t, 0))
                        for b, j in T.Parallel(BM, S):
                            row = bt * BM + b
                            col = s * S + j
                            valid = row < B
                            step = BASE + t
                            if prefetch:
                                gi = stage[j, b] + xi[b, j] + T.cast(BIAS[0, head, col], acc)
                                gf = stage[S + j, b] + xf[b, j] + T.cast(BIAS[1, head, col], acc)
                                gz = stage[2 * S + j, b] + xz[b, j] + T.cast(BIAS[2, head, col], acc)
                                go = stage[3 * S + j, b] + xo[b, j] + T.cast(BIAS[3, head, col], acc)
                                nxt = valid & (t + 1 < NSTEPS)
                                xi[b, j] = T.if_then_else(nxt, T.cast(WX[row, step + 1, 0, head, col], acc), 0.0)
                                xf[b, j] = T.if_then_else(nxt, T.cast(WX[row, step + 1, 1, head, col], acc), 0.0)
                                xz[b, j] = T.if_then_else(nxt, T.cast(WX[row, step + 1, 2, head, col], acc), 0.0)
                                xo[b, j] = T.if_then_else(nxt, T.cast(WX[row, step + 1, 3, head, col], acc), 0.0)
                            elif "wx" in ab:
                                gi = stage[j, b] + T.cast(BIAS[0, head, col], acc)
                                gf = stage[S + j, b] + T.cast(BIAS[1, head, col], acc)
                                gz = stage[2 * S + j, b] + T.cast(BIAS[2, head, col], acc)
                                go = stage[3 * S + j, b] + T.cast(BIAS[3, head, col], acc)
                            else:
                                gi = stage[j, b] + T.if_then_else(valid, T.cast(WX[row, step, 0, head, col], acc), 0.0) \
                                    + T.cast(BIAS[0, head, col], acc)
                                gf = stage[S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 1, head, col], acc),
                                                                      0.0) + T.cast(BIAS[1, head, col], acc)
                                gz = stage[2 * S + j, b] + T.if_then_else(
                                    valid, T.cast(WX[row, step, 2, head, col], acc), 0.0) + T.cast(BIAS[2, head, col], acc)
                                go = stage[3 * S + j, b] + T.if_then_else(
                                    valid, T.cast(WX[row, step, 3, head, col], acc), 0.0) + T.cast(BIAS[3, head, col], acc)
                            if "epilogue" in ab:
                                c[b, j] = c[b, j] * 0.5 + gi * gf
                                hid[b, j] = c[b, j] + go * gz
                            elif slstm:
                                zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                                og = 1.0 / (1.0 + T.exp(-go))
                                logf = T.min(gf, 0.0) - T.log(1.0 + T.exp(-T.abs(gf))) + m[b, j]
                                nm = T.if_then_else((START != 0) & (t == 0) & (ZERO[0] != 0), gi, T.max(gi, logf))
                                ig = T.exp(gi - nm)
                                fg = T.exp(logf - nm)
                                c[b, j] = fg * c[b, j] + ig * zv
                                n[b, j] = T.max(fg * n[b, j] + ig, 1.0)
                                hid[b, j] = og * c[b, j] / n[b, j]
                                m[b, j] = nm
                            else:
                                zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                                og = 1.0 / (1.0 + T.exp(-go))
                                c[b, j] = 1.0 / (1.0 + T.exp(-gf)) * c[b, j] + 1.0 / (1.0 + T.exp(-gi)) * zv
                                hid[b, j] = og * (2.0 / (1.0 + T.exp(-2.0 * c[b, j])) - 1.0)
                            if "out" not in ab:
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
                            if "exchange" not in ab:
                                if t + 1 < NSTEPS:
                                    if single:
                                        XBUF[0, g, col, b] = T.cast(hid[b, j], dtype)
                                    elif not arrive_first:
                                        XBUF[t % 2, g, col, b] = T.cast(hid[b, j], dtype)
                    if "exchange" not in ab:
                        if t + 1 < NSTEPS:
                            if not no_barrier:
                                T.sync_grid()
                            if arrive_first:
                                if active:
                                    for b, j in T.Parallel(BM, S):
                                        XBUF[t % 2, g, s * S + j, b] = T.cast(hid[b, j], dtype)

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


def hybrid_smem(width, reg_k, slice_width=16, bm=16, threads=128):
    """Shared memory of the hybrid grid kernel: the R rows outside registers, h^T and one gate stage per warpgroup."""
    return 4 * slice_width * (width - reg_k) * 2 + width * bm * 2 + (threads // 128) * 4 * slice_width * bm * 4


HYBRID_SLICE, HYBRID_THREADS, HYBRID_REG_K_STEP, HYBRID_REG_K_MAX = 16, 128, 64, 768  # max: rows per warpgroup


def supported_hybrid(width, reg_k, sms=132, bm=16, threads=128):
    """Hybrid residency: reg_k rows of every CTA's R slice in registers (WGMMA A from registers), the rest in shared
    memory; S = 16, BM = 16, every CTA of the cooperative grid resident. threads = 128: one warpgroup holds rows
    [0, reg_k). threads = 256: K is split in halves between two warpgroups, each holding reg_k / 2 rows of its half in
    registers and the rest in shared memory; the two partial gate accumulators are summed in shared memory."""
    if bm != 16 or threads not in (128, 256):
        return False
    wgs = threads // 128
    if width % (HYBRID_REG_K_STEP * wgs) or reg_k % (HYBRID_REG_K_STEP * wgs):
        return False
    per = reg_k // wgs
    if not (HYBRID_REG_K_STEP <= per <= HYBRID_REG_K_MAX and per < width // wgs):
        return False
    return 2 <= width // HYBRID_SLICE <= sms and hybrid_smem(width, reg_k, HYBRID_SLICE, bm, threads) <= GRID_SMEM_LIMIT


def build_grid_hybrid(batch, total_steps, heads, width, reg_k, *, sms, cell="lstm", dtype="float16", mode="full",
                      snapshot="float32"):
    """The grid kernel of build_grid with R split along K between registers and shared memory (large D).

    CTA s keeps rows [0, reg_k) of its R slice [4S, D] in a register fragment and rows [reg_k, D) in shared memory;
    h^T is staged in two shared tiles of the same split, and each step runs two WGMMAs into one accumulator
    (A from registers, then A from shared memory). The exchange (double-buffered XBUF, one T.sync_grid per step),
    the cell epilogue and the outputs are those of build_grid. No race-verification or ablation variants.
    """
    import tilelang.language as T

    S, BM = HYBRID_SLICE, 16
    if not supported_hybrid(width, reg_k, sms, BM):
        raise ValueError("unsupported (D, reg_k) for the hybrid grid path")
    D, H, B, TT, KR = width, heads, batch, total_steps, reg_k
    KS = D - KR
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
        with T.Kernel(k, resident, threads=HYBRID_THREADS) as (s, gy):
            rr = T.alloc_fragment((4 * S, KR), dtype)
            rs = T.alloc_shared((4 * S, KS), dtype)
            hr = T.alloc_shared((KR, BM), dtype)
            hs = T.alloc_shared((KS, BM), dtype)
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
                    for q, kk in T.Parallel(4 * S, KR):
                        rr[q, kk] = R[q // S, head, s * S + q % S, kk]
                    for q, kk in T.Parallel(4 * S, KS):
                        rs[q, kk] = R[q // S, head, s * S + q % S, KR + kk]
                    for kk, b in T.Parallel(KR, BM):
                        hr[kk, b] = T.if_then_else(bt * BM + b < B, T.cast(INIT[0, bt * BM + b, 0, head, kk], dtype),
                                                   T.cast(0, dtype))
                    for kk, b in T.Parallel(KS, BM):
                        hs[kk, b] = T.if_then_else(bt * BM + b < B,
                                                   T.cast(INIT[0, bt * BM + b, 0, head, KR + kk], dtype),
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
                            T.copy(XBUF[(t - 1) % 2, g, 0:KR, 0:BM], hr)
                            T.copy(XBUF[(t - 1) % 2, g, KR:D, 0:BM], hs)
                            T.fence_proxy_async()
                            T.sync_threads()
                        T.gemm(rr, hr, gates, clear_accum=True)
                        T.gemm(rs, hs, gates, clear_accum=False)
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
                            gz = stage[2 * S + j, b] + T.if_then_else(
                                valid, T.cast(WX[row, step, 2, head, col], acc), 0.0) + T.cast(BIAS[2, head, col], acc)
                            go = stage[3 * S + j, b] + T.if_then_else(
                                valid, T.cast(WX[row, step, 3, head, col], acc), 0.0) + T.cast(BIAS[3, head, col], acc)
                            if slstm:
                                zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                                og = 1.0 / (1.0 + T.exp(-go))
                                logf = T.min(gf, 0.0) - T.log(1.0 + T.exp(-T.abs(gf))) + m[b, j]
                                nm = T.if_then_else((START != 0) & (t == 0) & (ZERO[0] != 0), gi, T.max(gi, logf))
                                ig = T.exp(gi - nm)
                                fg = T.exp(logf - nm)
                                c[b, j] = fg * c[b, j] + ig * zv
                                n[b, j] = T.max(fg * n[b, j] + ig, 1.0)
                                hid[b, j] = og * c[b, j] / n[b, j]
                                m[b, j] = nm
                            else:
                                zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                                og = 1.0 / (1.0 + T.exp(-go))
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


def build_grid_hybrid2(batch, total_steps, heads, width, reg_k, *, sms, cell="lstm", dtype="float16", mode="full",
                       snapshot="float32"):
    """build_grid_hybrid with two warpgroups splitting K (D up to 2048 at S = 16).

    Warpgroup w (threads 128w .. 128w + 127) owns K rows [w D/2, (w + 1) D/2) of the CTA's R slice: the first
    reg_k / 2 of them in its register fragment, the rest in shared memory, and runs two WGMMAs into its own
    accumulator. Each warpgroup writes its partial gates to its own shared stage; after a block barrier all 256
    threads sum the two stages in the epilogue (one output element per thread). Exchange and epilogue as in
    build_grid.
    """
    import tilelang.language as T

    S, BM = HYBRID_SLICE, 16
    if not supported_hybrid(width, reg_k, sms, BM, 256):
        raise ValueError("unsupported (D, reg_k) for the two-warpgroup hybrid grid path")
    D, H, B, TT = width, heads, batch, total_steps
    KH, KR = D // 2, reg_k // 2
    KS = KH - KR
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
        with T.Kernel(k, resident, threads=256) as (s, gy):
            rr0 = T.alloc_fragment((4 * S, KR), dtype)
            rr1 = T.alloc_fragment((4 * S, KR), dtype)
            rs0 = T.alloc_shared((4 * S, KS), dtype)
            rs1 = T.alloc_shared((4 * S, KS), dtype)
            hr0 = T.alloc_shared((KR, BM), dtype)
            hs0 = T.alloc_shared((KS, BM), dtype)
            hr1 = T.alloc_shared((KR, BM), dtype)
            hs1 = T.alloc_shared((KS, BM), dtype)
            g0 = T.alloc_fragment((4 * S, BM), acc)
            g1 = T.alloc_fragment((4 * S, BM), acc)
            stage = T.alloc_shared((2, 4 * S, BM), acc)
            c = T.alloc_fragment((BM, S), acc)
            n = T.alloc_fragment((BM, S), acc)
            m = T.alloc_fragment((BM, S), acc)
            hid = T.alloc_fragment((BM, S), acc)
            tx = T.get_thread_binding()

            for slot in T.serial(slots):
                g = slot * resident + gy
                active = g < groups
                head = T.min(g, groups - 1) // tiles
                bt = T.min(g, groups - 1) % tiles
                if active:
                    for q, kk in T.Parallel(4 * S, KS):
                        rs0[q, kk] = R[q // S, head, s * S + q % S, KR + kk]
                    for q, kk in T.Parallel(4 * S, KS):
                        rs1[q, kk] = R[q // S, head, s * S + q % S, KH + KR + kk]
                    for kk, b in T.Parallel(KR, BM):
                        hr0[kk, b] = T.if_then_else(bt * BM + b < B, T.cast(INIT[0, bt * BM + b, 0, head, kk], dtype),
                                                    T.cast(0, dtype))
                    for kk, b in T.Parallel(KS, BM):
                        hs0[kk, b] = T.if_then_else(bt * BM + b < B,
                                                    T.cast(INIT[0, bt * BM + b, 0, head, KR + kk], dtype),
                                                    T.cast(0, dtype))
                    for kk, b in T.Parallel(KR, BM):
                        hr1[kk, b] = T.if_then_else(bt * BM + b < B,
                                                    T.cast(INIT[0, bt * BM + b, 0, head, KH + kk], dtype),
                                                    T.cast(0, dtype))
                    for kk, b in T.Parallel(KS, BM):
                        hs1[kk, b] = T.if_then_else(bt * BM + b < B,
                                                    T.cast(INIT[0, bt * BM + b, 0, head, KH + KR + kk], dtype),
                                                    T.cast(0, dtype))
                    if tx < 128:
                        for q, kk in T.Parallel(4 * S, KR):
                            rr0[q, kk] = R[q // S, head, s * S + q % S, kk]
                    else:
                        for q, kk in T.Parallel(4 * S, KR):
                            rr1[q, kk] = R[q // S, head, s * S + q % S, KH + kk]
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
                            T.copy(XBUF[(t - 1) % 2, g, 0:KR, 0:BM], hr0)
                            T.copy(XBUF[(t - 1) % 2, g, KR:KH, 0:BM], hs0)
                            T.copy(XBUF[(t - 1) % 2, g, KH:KH + KR, 0:BM], hr1)
                            T.copy(XBUF[(t - 1) % 2, g, KH + KR:D, 0:BM], hs1)
                            T.fence_proxy_async()
                            T.sync_threads()
                        if tx < 128:
                            T.gemm(rr0, hr0, g0, clear_accum=True)
                            T.gemm(rs0, hs0, g0, clear_accum=False)
                            T.copy(g0, stage[0, 0:4 * S, 0:BM])
                        else:
                            T.gemm(rr1, hr1, g1, clear_accum=True)
                            T.gemm(rs1, hs1, g1, clear_accum=False)
                            T.copy(g1, stage[1, 0:4 * S, 0:BM])
                        T.sync_threads()
                        for b, j in T.Parallel(BM, S):
                            row = bt * BM + b
                            col = s * S + j
                            valid = row < B
                            step = BASE + t
                            gi = stage[0, j, b] + stage[1, j, b] + T.if_then_else(
                                valid, T.cast(WX[row, step, 0, head, col], acc), 0.0) + T.cast(BIAS[0, head, col], acc)
                            gf = stage[0, S + j, b] + stage[1, S + j, b] + T.if_then_else(
                                valid, T.cast(WX[row, step, 1, head, col], acc), 0.0) + T.cast(BIAS[1, head, col], acc)
                            gz = stage[0, 2 * S + j, b] + stage[1, 2 * S + j, b] + T.if_then_else(
                                valid, T.cast(WX[row, step, 2, head, col], acc), 0.0) + T.cast(BIAS[2, head, col], acc)
                            go = stage[0, 3 * S + j, b] + stage[1, 3 * S + j, b] + T.if_then_else(
                                valid, T.cast(WX[row, step, 3, head, col], acc), 0.0) + T.cast(BIAS[3, head, col], acc)
                            if slstm:
                                zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                                og = 1.0 / (1.0 + T.exp(-go))
                                logf = T.min(gf, 0.0) - T.log(1.0 + T.exp(-T.abs(gf))) + m[b, j]
                                nm = T.if_then_else((START != 0) & (t == 0) & (ZERO[0] != 0), gi, T.max(gi, logf))
                                ig = T.exp(gi - nm)
                                fg = T.exp(logf - nm)
                                c[b, j] = fg * c[b, j] + ig * zv
                                n[b, j] = T.max(fg * n[b, j] + ig, 1.0)
                                hid[b, j] = og * c[b, j] / n[b, j]
                                m[b, j] = nm
                            else:
                                zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                                og = 1.0 / (1.0 + T.exp(-go))
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
           bm=16, threads=128, prefetch=False, reg_k=0, _ablate=frozenset(), _verify=None):
    """Compiled kernel for one logical-call shape; chunk length, offset and start are runtime arguments."""
    import tilelang

    if reg_k:
        if sync != "grid" or slice_width != HYBRID_SLICE or threads not in (128, 256) or prefetch or _ablate or _verify:
            raise ValueError("the hybrid path is the grid kernel with S = 16, one or two warpgroups and no variants")
        sms = sms or sm_count()
        key = ("hybrid", sms, batch, total_steps, heads, width, reg_k, cell, dtype, mode, snapshot, threads)
        if key not in _KERNELS:
            build_fn = build_grid_hybrid if threads == 128 else build_grid_hybrid2
            func = build_fn(batch, total_steps, heads, width, reg_k, sms=sms, cell=cell, dtype=dtype,
                            mode=mode, snapshot=snapshot)
            _KERNELS[key] = tilelang.compile(func, target={"kind": "cuda", "arch": "sm_90a"})
        return _KERNELS[key]
    if sync == "grid":
        sms = sms or sm_count()
        key = ("grid", sms, batch, total_steps, heads, width, slice_width, cell, dtype, mode, snapshot)
        func = lambda: build_grid(batch, total_steps, heads, width, slice_width, sms=sms, cell=cell,  # noqa: E731
                                  dtype=dtype, mode=mode, snapshot=snapshot, bm=bm, threads=threads,
                                  prefetch=prefetch, _ablate=_ablate, _verify=_verify)
    else:
        key = (batch, total_steps, heads, width, slice_width, cell, dtype, mode, snapshot)
        func = lambda: build(batch, total_steps, heads, width, slice_width, cell=cell, dtype=dtype,  # noqa: E731
                             mode=mode, snapshot=snapshot, bm=bm, threads=threads, prefetch=prefetch,
                             _ablate=_ablate, _verify=_verify)
    if bm != 16:
        key = key + (("bm", bm),)  # BM = 16 keys stay as they were
    if threads != 128:
        key = key + (("threads", threads),)
    if prefetch:
        key = key + (("prefetch", True),)
    if _ablate:
        key = key + (("ablate", tuple(sorted(_ablate))),)
    if _verify:
        key = key + (("verify", _verify.get("check", True), _verify.get("delay_seed") is not None, _verify.get("broken")),)
    if key not in _KERNELS:
        _KERNELS[key] = tilelang.compile(func(), target={"kind": "cuda", "arch": "sm_90a"})
    return _KERNELS[key]


def version():
    import tilelang

    return tilelang.__version__


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, sequence_context, slice_width=32,
               output_mode="full", snapshot_dtype=None, sync="cluster", batch_tile=16, threads=128, prefetch=False,
               reg_k=0, stream_h=False, _ablate=frozenset(), _verify=None):
    """Public per-call API with the triton_rtiled contract; returns (output, FP32 compute carry).

    sync="cluster" exchanges h inside a thread-block cluster (k <= 16); sync="grid" exchanges it
    through global memory under one cooperative grid barrier per step (k up to the SM count).
    batch_tile is the number of batch rows one CTA carries per step (WGMMA N); threads is the CTA size
    (128 = one warpgroup, 256 = two).

    _verify (race verification only, never served): {"check": bool, "delay_seed": int | None, "broken": None |
    "single" | "no_wait" | "no_sync" | "threads256" (cluster) | "no_barrier" | "arrive_first" (grid)}. Needs
    output_mode "hidden" with an FP32 snapshot; the violation counters are written back into the dict.
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
    if stream_h:
        from . import tilelang_stream

        ok = (sync == "grid" and slice_width == tilelang_stream.S and threads == tilelang_stream.THREADS
              and tilelang_stream.supported_stream(width, reg_k, batch_tile, sm_count()))
    elif reg_k:
        ok = sync == "grid" and slice_width == HYBRID_SLICE and supported_hybrid(width, reg_k, sm_count(), batch_tile,
                                                                                 threads)
    else:
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
    if stream_h:
        fn = tilelang_stream.kernel(batch, steps, heads, width, reg_k, batch_tile, cell=cell, dtype=names[wx.dtype],
                                    mode=output_mode, snapshot=names[snap], sms=sm_count())
    else:
        fn = kernel(batch, steps, heads, width, slice_width, cell=cell, dtype=names[wx.dtype], mode=output_mode,
                    snapshot=names[snap], sync=sync, bm=batch_tile, threads=threads, prefetch=prefetch, reg_k=reg_k,
                    _ablate=_ablate, _verify=_verify)
    out_shape = {"full": (states, batch, steps, heads, width), "hidden": (batch, steps, heads, width),
                 "last": (batch, 1, heads, width)}[output_mode]
    output = torch.empty(out_shape, device=wx.device, dtype=snap)
    final = torch.empty(initial.shape, device=wx.device, dtype=torch.float32)
    zero = sequence_context.zero_normalizer.to(torch.int32).reshape(1)
    if _verify:
        seed = int(_verify.get("delay_seed") or 0)
        zero = torch.cat((zero, torch.tensor([0, 0, seed], device=wx.device, dtype=torch.int32)))
    args = [wx.contiguous(), recurrent.contiguous(), bias.contiguous(), initial.float().contiguous(), zero, output, final]
    if sync == "grid":
        tiles = (batch + batch_tile - 1) // batch_tile
        args.append(torch.empty((2, tiles * heads, width, batch_tile), device=wx.device, dtype=wx.dtype))
    fn(*args, steps, 0, int(sequence_context.step_offset == 0))
    if _verify:
        counts = zero.tolist()
        _verify.update(after_barrier=counts[1], after_gemm=counts[2])
    return output, final
