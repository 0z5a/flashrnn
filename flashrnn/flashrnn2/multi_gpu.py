"""Forward recurrence with R split by output columns across several GPUs of one node (one process, peer memory).

GPU g owns D/P output columns and runs the same persistent R-tiled step as ``triton_rtiled``; after every step it
publishes its slice of round(h_t) into the exchange buffer of every GPU and waits until the whole h_t has arrived.
Peer buffers are reached through CUDA unified addressing: the kernel takes the buffer addresses as integers and casts
them to pointers; flags are system-scope release/acquire atomics.

Schedules (``SCHED``): 0 direct (every destination at once, visited in stride order, ``STRIDE`` coprime to P), 1 ring
(P-1 store-and-forward rounds), 2 recursive doubling (log2 P rounds, P a power of two). ``RMODE`` 0 keeps the R slice
in shared memory for the whole call (loaded before the step loop); 1 re-reads it every step from global memory (L2 or
HBM, the "stream" baselines). The h operand is read in K chunks of ``KC`` so that D up to 4096 fits. ``PROF`` times the
step phases (compute, publish, wait) with the global timer behind block barriers, as the backward kernel does.

``alloc``: "uva" (default) allocates the exchange buffer and the arrival counters with the PyTorch allocator and passes
their unified addresses (peer access enabled); "nccl_symm" takes them from NCCL 2.32 symmetric windows (symm_mem.py:
ncclMemAlloc, cuMem mappings, peer addresses from the LSA mapping). The kernel is the same for both.

Only whole logical calls (sequence start), output modes full/hidden, LSTM/sLSTM. Not served by dispatch; a
measurement path (docs/flashrnn2/scale-out-plan.md).
"""
import math

import torch
import triton
import triton.language as tl

from .state_contract import begin_sequence
from .triton_rtiled import SPIN_LIMIT  # noqa: F401  (single-GPU kernels' limit, kept for reference)

# Spin limit of the cross-GPU waits: 64x the single-GPU kernels' limit (seconds, not milliseconds), so that the launch
# skew between GPUs of one call cannot end a wait early (scale-out-plan.md section 10).
SPIN_MG = 1 << 26
from .triton_rtiled_bw import _now


@triton.jit
def _peer(PEERS, index, dtype: tl.constexpr):
    return tl.load(PEERS + index).to(tl.pointer_type(dtype))


@triton.jit
def _signal(SYNCS, dest, offset):
    cnt = _peer(SYNCS, dest, tl.int32)
    tl.atomic_add(cnt + offset, 1, sem="release", scope="sys")


@triton.jit
def _fence_sys():
    """System-scope acquire-release fence (one per thread): orders this CTA's earlier stores, behind a block
    barrier, before the relaxed arrival atomics that follow."""
    return tl.inline_asm_elementwise("fence.acq_rel.sys;\n mov.u32 $0, 0;", "=r", [], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _signal_relaxed(SYNCS, dest, offset):
    cnt = _peer(SYNCS, dest, tl.int32)
    tl.atomic_add(cnt + offset, 1, sem="relaxed", scope="sys")


@triton.jit
def _wait(SYNCS, rank, offset, target, ERR, SPIN: tl.constexpr):
    cnt = _peer(SYNCS, rank, tl.int32)
    seen = tl.atomic_add(cnt + offset, 0, sem="acquire", scope="sys")
    spins = seen * 0
    failed = seen * 0
    while (seen < target) & (spins < SPIN) & (failed == 0):
        seen = tl.atomic_add(cnt + offset, 0, sem="acquire", scope="sys")
        failed = tl.load(ERR, volatile=True)
        spins += 1
    if seen < target:
        tl.atomic_max(ERR, 1)
    tl.debug_barrier()


@triton.jit
def _forward_block(PEERS, rank, dest, xrow, col0, split, S: tl.constexpr, dtype: tl.constexpr):
    """Store-and-forward: this CTA's S-wide sub-slice of the block starting at column col0, from the local exchange
    buffer into ``dest``'s. The caller signals once per round after all of the round's blocks."""
    cols = col0 + split * S + tl.arange(0, S)
    src = _peer(PEERS, rank, dtype)
    dst = _peer(PEERS, dest, dtype)
    v = tl.load(src + xrow + cols[None, :], cache_modifier=".cg")
    tl.store(dst + xrow + cols[None, :], v)


@triton.jit(do_not_specialize=["B", "T", "GROUPS", "SLOTS", "RANK", "COL0", "SPLITS_PER", "TOTAL_SPLITS"])
def _rtiled_mg(
    WX, R, BIAS, INITIAL, OUTPUT, FINAL, PEERS, SYNCS, ERR, ZERO_NORMALIZER,
    B, T, GROUPS, SLOTS, RANK, COL0, SPLITS_PER, TOTAL_SPLITS,
    H: tl.constexpr, D: tl.constexpr, SLSTM: tl.constexpr, BM: tl.constexpr, S: tl.constexpr,
    KC: tl.constexpr, NCH: tl.constexpr, OUTPUT_MODE: tl.constexpr, SPIN: tl.constexpr,
    NP: tl.constexpr, SCHED: tl.constexpr, STRIDE: tl.constexpr, ROUNDS: tl.constexpr, DG: tl.constexpr,
    RMODE: tl.constexpr, PROF: tl.constexpr,
):
    split = tl.program_id(0)
    slot = tl.program_id(1)
    local = tl.arange(0, BM)
    o = COL0 + split * S + tl.arange(0, S)
    o_valid = o < D
    dtype = WX.dtype.element_ty
    stride = B * H * D
    output_stride = B * T * H * D
    bars = split * 0
    if PROF:
        p_comp = tl.zeros((), tl.int64)
        p_pub = tl.zeros((), tl.int64)
        p_wait = tl.zeros((), tl.int64)
        p_n = tl.zeros((), tl.int64)
    for group in range(slot, GROUPS, SLOTS):
        rows = (group // H) * BM + local
        head = group % H
        row_valid = rows < B
        valid = row_valid[:, None] & o_valid[None, :]
        state_offset = (rows[:, None] * H + head) * D + o[None, :]
        c = tl.load(INITIAL + stride + state_offset, valid, 0).to(tl.float32)
        n = tl.full((BM, S), 1.0, tl.float32)
        m = tl.zeros((BM, S), tl.float32)
        if SLSTM:
            n = tl.load(INITIAL + 2 * stride + state_offset, valid, 0).to(tl.float32)
            m = tl.load(INITIAL + 3 * stride + state_offset, valid, 0).to(tl.float32)
            zero_normalizer = tl.load(ZERO_NORMALIZER)
        # Tile [k, o] = R[g, head, o, k] (operand b of the MMA). Resident mode keeps one or two K chunks of it for
        # the whole call (D <= 2 * KC); stream mode re-reads the chunks every step.
        rbase = R + (head * D + o[None, :]) * D
        if RMODE == 0:
            kk0 = tl.arange(0, KC)[:, None]
            m0 = o_valid[None, :] & (kk0 < D)
            ri0 = tl.load(rbase + kk0, m0, 0)
            rf0 = tl.load(rbase + H * D * D + kk0, m0, 0)
            rz0 = tl.load(rbase + 2 * H * D * D + kk0, m0, 0)
            ro0 = tl.load(rbase + 3 * H * D * D + kk0, m0, 0)
            ri1, rf1, rz1, ro1 = ri0, rf0, rz0, ro0
            if NCH > 1:
                kk1 = KC + tl.arange(0, KC)[:, None]
                m1 = o_valid[None, :] & (kk1 < D)
                ri1 = tl.load(rbase + kk1, m1, 0)
                rf1 = tl.load(rbase + H * D * D + kk1, m1, 0)
                rz1 = tl.load(rbase + 2 * H * D * D + kk1, m1, 0)
                ro1 = tl.load(rbase + 3 * H * D * D + kk1, m1, 0)
        bi = tl.load(BIAS + head * D + o, o_valid, 0).to(tl.float32)
        bf = tl.load(BIAS + H * D + head * D + o, o_valid, 0).to(tl.float32)
        bz = tl.load(BIAS + 2 * H * D + head * D + o, o_valid, 0).to(tl.float32)
        bo = tl.load(BIAS + 3 * H * D + head * D + o, o_valid, 0).to(tl.float32)
        h_row = (rows[:, None] * H + head) * D
        hidden = tl.zeros((BM, S), tl.float32)
        for step in range(T):
            if PROF:
                t_a = _now()
            ai = tl.zeros((BM, S), tl.float32)
            af = tl.zeros((BM, S), tl.float32)
            az = tl.zeros((BM, S), tl.float32)
            ao = tl.zeros((BM, S), tl.float32)
            parity = bars % 2
            xrow = ((slot * 2 + parity) * BM + local)[:, None] * D
            xb = _peer(PEERS, RANK, dtype)
            if RMODE == 0:
                # Resident: one or two K chunks held for the whole call (static).
                for ch in tl.static_range(NCH):
                    kk = ch * KC + tl.arange(0, KC)
                    if step > 0:
                        hk = tl.load(xb + xrow + kk[None, :], kk[None, :] < D, 0, cache_modifier=".cg")
                    else:
                        hk = tl.load(INITIAL + h_row + kk[None, :], row_valid[:, None] & (kk[None, :] < D), 0)
                        hk = hk.to(tl.float32).to(dtype)
                    if ch == 0:
                        ai = tl.dot(hk, ri0, ai)
                        af = tl.dot(hk, rf0, af)
                        az = tl.dot(hk, rz0, az)
                        ao = tl.dot(hk, ro0, ao)
                    else:
                        ai = tl.dot(hk, ri1, ai)
                        af = tl.dot(hk, rf1, af)
                        az = tl.dot(hk, rz1, az)
                        ao = tl.dot(hk, ro1, ao)
            else:
                # Stream: re-read the R chunks every step (runtime loop keeps one chunk of code live).
                for ch in range(NCH):
                    kk = ch * KC + tl.arange(0, KC)
                    if step > 0:
                        hk = tl.load(xb + xrow + kk[None, :], kk[None, :] < D, 0, cache_modifier=".cg")
                    else:
                        hk = tl.load(INITIAL + h_row + kk[None, :], row_valid[:, None] & (kk[None, :] < D), 0)
                        hk = hk.to(tl.float32).to(dtype)
                    kk2 = kk[:, None]
                    msk = o_valid[None, :] & (kk2 < D)
                    ai = tl.dot(hk, tl.load(rbase + kk2, msk, 0), ai)
                    af = tl.dot(hk, tl.load(rbase + H * D * D + kk2, msk, 0), af)
                    az = tl.dot(hk, tl.load(rbase + 2 * H * D * D + kk2, msk, 0), az)
                    ao = tl.dot(hk, tl.load(rbase + 3 * H * D * D + kk2, msk, 0), ao)
            x_offset = (((rows[:, None] * T + step) * 4) * H + head) * D + o[None, :]
            i = ai + tl.load(WX + x_offset, valid, 0).to(tl.float32) + bi[None, :]
            f = af + tl.load(WX + x_offset + H * D, valid, 0).to(tl.float32) + bf[None, :]
            z = az + tl.load(WX + x_offset + 2 * H * D, valid, 0).to(tl.float32) + bz[None, :]
            o_gate = ao + tl.load(WX + x_offset + 3 * H * D, valid, 0).to(tl.float32) + bo[None, :]
            z_value = 2.0 * tl.sigmoid(2.0 * z) - 1.0
            if SLSTM:
                logf = tl.minimum(f, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(f))) + m
                next_m = tl.where((step == 0) & zero_normalizer, i, tl.maximum(i, logf))
                igate = tl.exp(i - next_m)
                fgate = tl.exp(logf - next_m)
                c = fgate * c + igate * z_value
                n = tl.maximum(fgate * n + igate, 1.0)
                hidden = tl.sigmoid(o_gate) * c / n
                m = next_m
            else:
                c = tl.sigmoid(f) * c + tl.sigmoid(i) * z_value
                hidden = tl.sigmoid(o_gate) * (2.0 * tl.sigmoid(2.0 * c) - 1.0)
            output_offset = ((rows[:, None] * T + step) * H + head) * D + o[None, :]
            tl.store(OUTPUT + output_offset, hidden, valid)
            if OUTPUT_MODE == 0:
                tl.store(OUTPUT + output_stride + output_offset, c, valid)
                if SLSTM:
                    tl.store(OUTPUT + 2 * output_stride + output_offset, n, valid)
                    tl.store(OUTPUT + 3 * output_stride + output_offset, m, valid)
            if PROF:
                t_b = _now()
                t_c = t_b
            if (step < T - 1) | (group + SLOTS < GROUPS):
                parity = (bars + 1) % 2
                wrow = ((slot * 2 + parity) * BM + local)[:, None] * D
                hv = hidden.to(dtype)
                bars += 1
                if SCHED == 0:
                    # Direct: own slice into every GPU's buffer (stride order), one arrival on every counter.
                    for r in tl.static_range(NP):
                        dest = (RANK + r * STRIDE) % NP
                        tl.store(_peer(PEERS, dest, dtype) + wrow + o[None, :], hv, o_valid[None, :])
                    tl.debug_barrier()
                    # One system-scope fence, then relaxed arrivals: P release atomics in a row each waited for the
                    # stores to become visible again, which serialized the publish at P = 4 (section 20).
                    fenced = _fence_sys()  # the offset below depends on it, so the fence cannot be dropped or moved
                    for r in tl.static_range(NP):
                        dest = (RANK + r * STRIDE) % NP
                        _signal_relaxed(SYNCS, dest, slot + fenced)
                    if PROF:
                        t_c = _now()
                    _wait(SYNCS, RANK, slot, bars * TOTAL_SPLITS, ERR, SPIN)
                else:
                    # Ring and doubling wait for their round partners only, so the CTAs of this GPU also meet at a
                    # per-GPU counter after the rounds (sections 10, 14): each CTA signals it once its slice of the
                    # own block is stored.
                    tl.store(_peer(PEERS, RANK, dtype) + wrow + o[None, :], hv, o_valid[None, :])
                    tl.debug_barrier()
                    _signal(SYNCS, RANK, SLOTS * ROUNDS + slot)  # this CTA's slice of the own block is stored
                    for rnd in tl.static_range(ROUNDS):
                        # Round 0 sends this GPU's own block from registers. Reading it back from the local
                        # buffer (L2-only loads after a CTA-scope barrier) could see the L2 copy before this
                        # CTA's store reached it (section 12). Later rounds forward blocks that arrived from
                        # peers, ordered by the system-scope acquire of the previous round's wait.
                        if SCHED == 1:
                            # Ring: forward the block received last round (own block first) to the next GPU.
                            dest = (RANK + 1) % NP
                            origin = (RANK - rnd + NP) % NP
                            if rnd == 0:
                                tl.store(_peer(PEERS, dest, dtype) + wrow + o[None, :], hv, o_valid[None, :])
                            else:
                                _forward_block(PEERS, RANK, dest, wrow, origin * DG, split, S, dtype)
                        else:
                            # Recursive doubling: forward every block owned so far to the partner of this round.
                            dest = RANK ^ (1 << rnd)
                            low = RANK - (RANK % (1 << rnd))
                            if rnd == 0:
                                tl.store(_peer(PEERS, dest, dtype) + wrow + o[None, :], hv, o_valid[None, :])
                            else:
                                for b in tl.static_range(1 << rnd):
                                    if low + b == RANK:
                                        tl.store(_peer(PEERS, dest, dtype) + wrow + o[None, :], hv, o_valid[None, :])
                                    else:
                                        _forward_block(PEERS, RANK, dest, wrow, (low + b) * DG, split, S, dtype)
                        tl.debug_barrier()
                        _signal(SYNCS, dest, slot * ROUNDS + rnd)  # one arrival per CTA per round, after its blocks
                        if PROF:
                            if rnd == 0:
                                t_c = _now()
                        _wait(SYNCS, RANK, slot * ROUNDS + rnd, bars * SPLITS_PER, ERR, SPIN)
                    # Every CTA of this GPU has stored its slice of this step's own block: the next step may read
                    # the whole block, and (each CTA signalled after its compute) nobody still reads the buffer
                    # half that the next publish overwrites.
                    _wait(SYNCS, RANK, SLOTS * ROUNDS + slot, bars * SPLITS_PER, ERR, SPIN)
            if PROF:
                t_d = _now()
                p_comp += t_b - t_a
                p_pub += t_c - t_b
                p_wait += t_d - t_c
                p_n += 1
        tl.store(FINAL + state_offset, hidden, valid)
        tl.store(FINAL + stride + state_offset, c, valid)
        if SLSTM:
            tl.store(FINAL + 2 * stride + state_offset, n, valid)
            tl.store(FINAL + 3 * stride + state_offset, m, valid)
    if PROF:
        prof = ERR + 32 + (split + SPLITS_PER * slot) * 8
        tl.atomic_add(prof, p_comp.to(tl.int32))
        tl.atomic_add(prof + 1, p_pub.to(tl.int32))
        tl.atomic_add(prof + 2, p_wait.to(tl.int32))
        tl.atomic_add(prof + 3, p_n.to(tl.int32))


SCHEDULES = {"direct": 0, "ring": 1, "doubling": 2}


def rounds_of(schedule, peers):
    if schedule == "direct":
        return 1
    if schedule == "ring":
        return max(1, peers - 1)
    if peers & (peers - 1):
        raise ValueError("recursive doubling needs a power-of-two GPU count")
    return max(1, int(math.log2(peers)))


def enable_peer_access(devices):
    """Enable P2P between every pair (PyTorch enables it on the first cross-device copy) and prove it with a kernel
    store from each device into a buffer of every other device."""
    for a in devices:
        for b in devices:
            if a != b and not torch.cuda.can_device_access_peer(a, b):
                raise RuntimeError(f"no peer access between cuda:{a} and cuda:{b}")
    probes = {d: torch.zeros((len(devices),), device=f"cuda:{d}", dtype=torch.int32) for d in devices}
    for a in devices:
        for b in devices:
            if a != b:
                torch.ones((1,), device=f"cuda:{a}").to(f"cuda:{b}")  # first copy enables peer access
    for i, a in enumerate(devices):
        torch.cuda.set_device(a)
        addrs = torch.tensor([probes[b].data_ptr() for b in devices], device=f"cuda:{a}", dtype=torch.int64)
        _probe_store[(len(devices),)](addrs, i + 1, i)
    for d in devices:
        torch.cuda.synchronize(d)
    seen = {d: probes[d].tolist() for d in devices}
    expected = [i + 1 for i in range(len(devices))]
    if any(v != expected for v in seen.values()):
        raise RuntimeError(f"peer store probe failed: {seen}")
    return seen


@triton.jit
def _probe_store(ADDRS, value, slot):
    dest = tl.program_id(0)
    base = tl.load(ADDRS + dest).to(tl.pointer_type(tl.int32))
    tl.store(base + slot, value)


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, devices, **kw):
    """One logical call on ``devices`` (list of CUDA device indices). Inputs may live anywhere; each GPU gets a copy
    of Wx, R, bias and the initial state and computes its own D/P output columns. Returns the gathered hidden output
    ([B,T,H,D] or [S,B,T,H,D] for ``full``) on the first device, the FP32 final carry gathered likewise, and timing.
    Keyword options: see ``_recurrence``. The caller's current CUDA device is restored: Triton and the CUDA runtime
    launch on the current device, and a later launch on another GPU's tensors would otherwise run on the last GPU of
    this call (section 14)."""
    previous = torch.cuda.current_device()
    try:
        return _recurrence(wx, recurrent, bias, initial, cell, devices=devices, **kw)
    finally:
        torch.cuda.set_device(previous)


def _recurrence(wx, recurrent, bias, initial, cell="lstm", *, devices, slice_width=16, schedule="direct", stride=1,
                rmode="smem", num_warps=4, output_mode="hidden", k_chunk=512, profile=False, diagnostics=None,
                alloc="uva"):
    if cell not in ("lstm", "slstm"):
        raise ValueError("LSTM/sLSTM only")
    if output_mode not in ("full", "hidden"):
        raise ValueError("output mode full or hidden")
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule in {sorted(SCHEDULES)}")
    if rmode not in ("smem", "stream"):
        raise ValueError("rmode smem or stream")
    if alloc not in ("uva", "nccl_symm"):
        raise ValueError("alloc uva or nccl_symm")
    peers = len(devices)
    batch, steps, gates, heads, width = wx.shape
    if width % (peers * slice_width):
        raise ValueError("D must be a multiple of P * slice width")
    if schedule == "direct" and math.gcd(stride, peers) != 1:
        raise ValueError("stride must be coprime to the GPU count")
    if k_chunk & (k_chunk - 1) or width % k_chunk and width > k_chunk:
        raise ValueError("k_chunk must be a power of two dividing D (or >= D)")
    kc = min(k_chunk, triton.next_power_of_2(width))
    nch = triton.cdiv(width, kc)
    if rmode == "smem" and nch > 2:
        raise ValueError("the resident mode holds at most two K chunks: raise k_chunk (D <= 2 * k_chunk)")
    dg = width // peers
    splits_per = dg // slice_width
    rounds = rounds_of(schedule, peers)
    states = 4 if cell == "slstm" else 2
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("expected public initial state layout")
    ctx = begin_sequence(initial, cell, wx.dtype)
    groups = triton.cdiv(batch, 16) * heads
    # Slots: the same on every GPU (the step barrier spans the whole group across GPUs).
    sms = min(torch.cuda.get_device_properties(d).multi_processor_count for d in devices)
    if splits_per > sms:
        raise ValueError("per-GPU R split exceeds the SM count")
    slots = min(groups, sms // splits_per)
    per = {}
    for d in devices:
        dev = f"cuda:{d}"
        per[d] = dict(
            wx=wx.to(dev, non_blocking=True).contiguous(), r=recurrent.to(dev, non_blocking=True).contiguous(),
            bias=bias.to(dev, non_blocking=True).contiguous(), init=initial.to(dev, non_blocking=True).contiguous(),
            zero=ctx.zero_normalizer.to(dev),
            out=torch.empty((states, batch, steps, heads, width) if output_mode == "full"
                            else (batch, steps, heads, width), device=dev, dtype=torch.float32),
            final=torch.empty((states, batch, 1, heads, width), device=dev, dtype=torch.float32),
            xbuf=torch.zeros((slots, 2, 16, width), device=dev, dtype=wx.dtype) if alloc == "uva" else None,
            sync=torch.zeros((slots * (rounds + 1) + 8,), device=dev, dtype=torch.int32) if alloc == "uva" else None,
            err=torch.zeros((32 + 8 * splits_per * slots if profile else 1,), device=dev, dtype=torch.int32),
        )
    if alloc == "nccl_symm":
        from . import symm_mem

        sess = symm_mem.session(devices)
        xbytes = slots * 2 * 16 * width * wx.element_size()
        sbytes = (slots * (rounds + 1) + 8) * 4
        xid, sid = sess.buffer("xbuf", xbytes), sess.buffer("sync", sbytes)
        for rank, d in enumerate(devices):
            torch.cuda.set_device(d)
            sess.zero(xid, rank, xbytes)
            sess.zero(sid, rank, sbytes)
    for d in devices:
        torch.cuda.synchronize(d)
    if alloc == "uva":
        xaddr = [per[d]["xbuf"].data_ptr() for d in devices]
        saddr = [per[d]["sync"].data_ptr() for d in devices]
    events = {}
    kernel = None
    # Every address table exists before the first GPU starts: the first GPU spins on the exchange from its launch on.
    for rank, d in enumerate(devices):
        torch.cuda.set_device(d)
        p = per[d]
        if alloc == "uva":
            p["peers_t"] = torch.tensor(xaddr, device=f"cuda:{d}", dtype=torch.int64)
            p["syncs_t"] = torch.tensor(saddr, device=f"cuda:{d}", dtype=torch.int64)
        else:  # LSA addresses of the session's ranks; the kernel reads the first P entries
            p["peers_t"], p["syncs_t"] = sess.table(xid, rank), sess.table(sid, rank)
    for d in devices:
        torch.cuda.synchronize(d)
    for rank, d in enumerate(devices):
        torch.cuda.set_device(d)
        p = per[d]
        peers_t, syncs_t = p["peers_t"], p["syncs_t"]
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        kernel = _rtiled_mg[(splits_per, slots)](
            p["wx"], p["r"], p["bias"], p["init"], p["out"], p["final"], peers_t, syncs_t, p["err"], p["zero"],
            batch, steps, groups, slots, rank, rank * dg, splits_per, width // slice_width,
            heads, width, cell == "slstm", 16, slice_width, kc, nch, 0 if output_mode == "full" else 1, SPIN_MG,
            peers, SCHEDULES[schedule], stride, rounds, dg, 0 if rmode == "smem" else 1, int(profile),
            num_warps=num_warps, enable_fp_fusion=False, launch_cooperative_grid=True)
        e1.record()
        events[d] = (e0, e1)
    for d in devices:
        torch.cuda.synchronize(d)
    ms = {d: events[d][0].elapsed_time(events[d][1]) for d in devices}
    errors = {d: int(per[d]["err"][0].item()) for d in devices}
    if any(errors.values()):
        raise RuntimeError(f"exchange barrier timed out: {errors}")
    first = f"cuda:{devices[0]}"
    out = per[devices[0]]["out"].clone()
    final = per[devices[0]]["final"].clone()
    for rank, d in enumerate(devices):
        cols = slice(rank * dg, (rank + 1) * dg)
        out[..., cols] = per[d]["out"][..., cols].to(first)
        final[..., cols] = per[d]["final"][..., cols].to(first)
    if diagnostics is not None:
        diagnostics.update(alloc=alloc, ms=ms, kernel_ms=max(ms.values()), registers=kernel.n_regs, spills=kernel.n_spills,
                           shared_bytes=kernel.metadata.shared, splits_per=splits_per, slots=slots, rounds=rounds,
                           groups=groups, k_chunk=kc, chunks=nch)
        if profile:
            phases = {}
            for d in devices:
                sums = per[d]["err"][32:].view(splits_per * slots, 8)[:, :4].tolist()
                n = max(1, sums[0][3])
                phases[d] = dict(names=["compute", "publish", "wait"],
                                 mean_ns=[sum(r[i] for r in sums) / len(sums) / n for i in range(3)],
                                 max_ns=[max(r[i] for r in sums) / n for i in range(3)], steps=n)
            diagnostics["phases"] = phases
    return out, final, ms


def geometry(batch, heads, width, peers, slice_width=16, schedule="direct", k_chunk=512, sms=132, element_size=2):
    """Per-GPU launch geometry of a P-GPU call and its exchange-buffer sizes (identical GPUs assumed)."""
    kc = min(k_chunk, triton.next_power_of_2(width))
    splits_per = width // peers // slice_width
    rounds = rounds_of(schedule, peers)
    groups = triton.cdiv(batch, 16) * heads
    slots = min(groups, sms // splits_per)
    return dict(kc=kc, nch=triton.cdiv(width, kc), dg=width // peers, splits_per=splits_per, rounds=rounds,
                groups=groups, slots=slots, xbytes=slots * 2 * 16 * width * element_size,
                sbytes=(slots * (rounds + 1) + 8) * 4)


def recurrence_rank(wx, recurrent, bias, initial, cell="lstm", *, rank, peers, xtable, stable, slice_width=16,
                    schedule="direct", stride=1, rmode="smem", num_warps=4, output_mode="hidden", k_chunk=512):
    """One GPU's share of a P-GPU call, for one process per GPU (e.g. NVSHMEM): inputs on the current device, the
    exchange buffers already zeroed on every GPU, ``xtable`` / ``stable`` int64 device tensors with every GPU's
    exchange-buffer / counter address as seen from this GPU. Launches on the current stream and returns
    (output, FP32 final carry, events); only this GPU's columns rank * D / P .. of the outputs are written."""
    batch, steps, gates, heads, width = wx.shape
    dev = wx.device
    g = geometry(batch, heads, width, peers, slice_width, schedule, k_chunk,
                 torch.cuda.get_device_properties(dev).multi_processor_count, wx.element_size())
    states = 4 if cell == "slstm" else 2
    ctx = begin_sequence(initial, cell, wx.dtype)
    out = torch.zeros((states, batch, steps, heads, width) if output_mode == "full" else (batch, steps, heads, width),
                      device=dev, dtype=torch.float32)
    final = torch.zeros((states, batch, 1, heads, width), device=dev, dtype=torch.float32)
    err = torch.zeros((1,), device=dev, dtype=torch.int32)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    _rtiled_mg[(g["splits_per"], g["slots"])](
        wx.contiguous(), recurrent.contiguous(), bias.contiguous(), initial.contiguous(), out, final, xtable, stable,
        err, ctx.zero_normalizer.to(dev), batch, steps, g["groups"], g["slots"], rank, rank * g["dg"], g["splits_per"],
        width // slice_width, heads, width, cell == "slstm", 16, slice_width, g["kc"], g["nch"],
        0 if output_mode == "full" else 1, SPIN_MG, peers, SCHEDULES[schedule], stride, g["rounds"], g["dg"],
        0 if rmode == "smem" else 1, 0, num_warps=num_warps, enable_fp_fusion=False, launch_cooperative_grid=True)
    e1.record()
    return out, final, (e0, e1, err)
