"""Recurrent-weight tiling across CTAs for generic LSTM/sLSTM inference, D > 128.

R is split along its output (gate) axis: CTA ``s`` of a group holds
R[:, head, s*S:(s+1)*S, :] for all four gates in shared memory and the matching
c/n/m slices in registers for the whole time loop. Only the rounded hidden
operand h_t crosses CTAs. It is exchanged through a double-buffered L2-resident
global buffer (loads bypass L1) and ordered by a per-group counter barrier
(release arrive, acquire spin) once per step.

All CTAs of a launch must be co-resident. The persistent path uses a CUDA
cooperative launch, so the driver rejects a grid that cannot be resident. The
launcher bounds the grid to one CTA per SM and lets each slot of k CTAs process
several (batch tile, head) groups serially. A bounded spin raises an error flag
instead of hanging.

``sync="launch"`` replaces every in-kernel barrier by a kernel boundary (one
launch per step through the FP32 compute carry). It performs the same
arithmetic and is the bitwise oracle for the persistent ``sync="grid"`` path.
Forward only; FlashRNN generic semantics only (no TiRex/xLSTM arithmetic).
"""

import torch
import triton
import triton.language as tl

from .state_contract import SequenceContext, state_contract

SPIN_LIMIT = 1 << 20


@triton.jit
def _group_barrier(SYNC, slot, target, ERR, SPIN: tl.constexpr):
    # Every thread's exchange stores precede the single-thread release arrive.
    tl.debug_barrier()
    tl.atomic_add(SYNC + slot, 1, sem="release", scope="gpu")
    seen = tl.atomic_add(SYNC + slot, 0, sem="acquire", scope="gpu")
    spins = seen * 0
    failed = seen * 0
    # Stop early once any CTA of the launch has already reported a failed barrier.
    while (seen < target) & (spins < SPIN) & (failed == 0):
        seen = tl.atomic_add(SYNC + slot, 0, sem="acquire", scope="gpu")
        failed = tl.load(ERR, volatile=True)
        spins += 1
    if seen < target:
        tl.atomic_max(ERR, 1)
    tl.debug_barrier()


@triton.jit
def _verify_delay(seed, a, b, c, POINT: tl.constexpr):
    # Race-verification only: a pseudo-random sleep of 0-65 us per (CTA, step, point), from the runtime seed.
    x = (a * 7919) ^ (b * 104729) ^ (c * 1299709) ^ (seed * 40503 + POINT * 977)
    ns = ((x ^ (x >> 13)) * 48271) & 65535
    tl.inline_asm_elementwise("nanosleep.u32 $1; mov.u32 $0, 0;", "=r,r", [tl.full((1,), 0, tl.int32) + ns],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _verify_operand(h1, h2, OUTPUT, COUNTER, rows, row_valid, head, prev, TOTAL_T, H: tl.constexpr, D: tl.constexpr,
                    K1: tl.constexpr, K2: tl.constexpr, OUTPUT_MODE: tl.constexpr):
    # Received hidden operand vs the sender's stored hidden state of step `prev` (hidden or full output mode with an
    # FP32 snapshot, so rounding it reproduces the operand exactly); mismatches are counted in COUNTER.
    k1 = tl.arange(0, K1)
    base = ((rows[:, None] * TOTAL_T + prev) * H + head) * D
    m1 = row_valid[:, None] & (k1[None, :] < D)
    want1 = tl.load(OUTPUT + base + k1[None, :], m1, 0, cache_modifier=".cg").to(h1.dtype)
    bad = tl.sum(tl.sum(((h1 != want1) & m1).to(tl.int32), 1), 0)
    if K2 > 0:
        k2 = K1 + tl.arange(0, K2)
        m2 = row_valid[:, None] & (k2[None, :] < D)
        want2 = tl.load(OUTPUT + base + k2[None, :], m2, 0, cache_modifier=".cg").to(h2.dtype)
        bad += tl.sum(tl.sum(((h2 != want2) & m2).to(tl.int32), 1), 0)
    tl.atomic_add(COUNTER, bad)


# Race-verification bits (VERIFY, compile-time; 0 compiles to the unmodified kernel). CHECK compares every received
# hidden operand with the sender's stored output, after the barrier and again after the MMAs; DELAY injects seeded
# sleeps before the producer's store and after the consumer's barrier; the BROKEN_* bits are negative controls.
V_CHECK = tl.constexpr(1)
V_DELAY = tl.constexpr(2)
V_BROKEN_SINGLE = tl.constexpr(4)
V_BROKEN_NO_BARRIER = tl.constexpr(8)
V_BROKEN_ARRIVE_FIRST = tl.constexpr(16)


# Runtime integers are not value-specialized: one compiled kernel per
# (dtype, H, D, S, K parts, output mode, snapshot dtype) serves every B/T/chunk.
# H stays a compile-time constant: as a runtime value it raised spills.
@triton.jit(do_not_specialize=["B", "T", "TOTAL_T", "STEP_BASE", "SEQUENCE_START", "GROUPS", "SLOTS",
                               "PERSISTENT"])
def _rtiled(
    WX,
    R,
    BIAS,
    INITIAL,
    OUTPUT,
    FINAL,
    XBUF,
    SYNC,
    ERR,
    ZERO_NORMALIZER,
    B,
    T,
    TOTAL_T,
    STEP_BASE,
    SEQUENCE_START,
    GROUPS,
    SLOTS,
    PERSISTENT,
    H: tl.constexpr,
    D: tl.constexpr,
    SLSTM: tl.constexpr,
    BM: tl.constexpr,
    S: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    OUTPUT_MODE: tl.constexpr,
    SPIN: tl.constexpr,
    VERIFY: tl.constexpr = 0,
):
    split = tl.program_id(0)
    slot = tl.program_id(1)
    splits = tl.num_programs(0)
    if VERIFY & V_DELAY:
        seed = tl.load(ERR + 3)
    local = tl.arange(0, BM)
    o = split * S + tl.arange(0, S)
    k1 = tl.arange(0, K1)
    o_valid = o < D
    dtype = WX.dtype.element_ty
    stride = B * H * D
    output_stride = B * TOTAL_T * H * D
    bars = split * 0
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
        # Tile [k, o] = R[g, head, o, k]: output slice as MMA operand b.
        w1 = (head * D + o[None, :]) * D + k1[:, None]
        w1_mask = o_valid[None, :] & (k1[:, None] < D)
        ri1 = tl.load(R + w1, w1_mask, 0)
        rf1 = tl.load(R + H * D * D + w1, w1_mask, 0)
        rz1 = tl.load(R + 2 * H * D * D + w1, w1_mask, 0)
        ro1 = tl.load(R + 3 * H * D * D + w1, w1_mask, 0)
        if K2 > 0:
            k2 = K1 + tl.arange(0, K2)
            w2 = (head * D + o[None, :]) * D + k2[:, None]
            w2_mask = o_valid[None, :] & (k2[:, None] < D)
            ri2 = tl.load(R + w2, w2_mask, 0)
            rf2 = tl.load(R + H * D * D + w2, w2_mask, 0)
            rz2 = tl.load(R + 2 * H * D * D + w2, w2_mask, 0)
            ro2 = tl.load(R + 3 * H * D * D + w2, w2_mask, 0)
        bi = tl.load(BIAS + head * D + o, o_valid, 0).to(tl.float32)
        bf = tl.load(BIAS + H * D + head * D + o, o_valid, 0).to(tl.float32)
        bz = tl.load(BIAS + 2 * H * D + head * D + o, o_valid, 0).to(tl.float32)
        bo = tl.load(BIAS + 3 * H * D + head * D + o, o_valid, 0).to(tl.float32)
        h_row = (rows[:, None] * H + head) * D
        h1 = tl.load(INITIAL + h_row + k1[None, :], row_valid[:, None] & (k1[None, :] < D), 0)
        h1 = h1.to(tl.float32).to(dtype)
        if K2 > 0:
            h2 = tl.load(INITIAL + h_row + k2[None, :], row_valid[:, None] & (k2[None, :] < D), 0)
            h2 = h2.to(tl.float32).to(dtype)
        hidden = tl.zeros((BM, S), tl.float32)
        for step in range(T):
            if step > 0:
                parity = bars % 2
                if VERIFY & V_BROKEN_SINGLE:
                    parity = 0
                if VERIFY & V_DELAY:
                    _verify_delay(seed, split + splits * slot, group, step, 1)
                read = XBUF + ((slot * 2 + parity) * BM + local[:, None]) * D
                h1 = tl.load(read + k1[None, :], k1[None, :] < D, 0, cache_modifier=".cg")
                if K2 > 0:
                    h2 = tl.load(read + k2[None, :], k2[None, :] < D, 0, cache_modifier=".cg")
                if VERIFY & V_CHECK:
                    if K2 > 0:
                        _verify_operand(h1, h2, OUTPUT, ERR + 1, rows, row_valid, head, STEP_BASE + step - 1,
                                        TOTAL_T, H, D, K1, K2, OUTPUT_MODE)
                    else:
                        _verify_operand(h1, h1, OUTPUT, ERR + 1, rows, row_valid, head, STEP_BASE + step - 1,
                                        TOTAL_T, H, D, K1, K2, OUTPUT_MODE)
            ai = tl.dot(h1, ri1)
            af = tl.dot(h1, rf1)
            az = tl.dot(h1, rz1)
            ao = tl.dot(h1, ro1)
            if K2 > 0:
                ai = tl.dot(h2, ri2, ai)
                af = tl.dot(h2, rf2, af)
                az = tl.dot(h2, rz2, az)
                ao = tl.dot(h2, ro2, ao)
            if VERIFY & V_CHECK:
                if step > 0:
                    # The buffer must still hold step - 1's operand after the MMAs (no early overwrite).
                    parity = bars % 2
                    if VERIFY & V_BROKEN_SINGLE:
                        parity = 0
                    read = XBUF + ((slot * 2 + parity) * BM + local[:, None]) * D
                    g1 = tl.load(read + k1[None, :], k1[None, :] < D, 0, cache_modifier=".cg")
                    g2 = g1
                    if K2 > 0:
                        g2 = tl.load(read + k2[None, :], k2[None, :] < D, 0, cache_modifier=".cg")
                    _verify_operand(g1, g2, OUTPUT, ERR + 2, rows, row_valid, head, STEP_BASE + step - 1, TOTAL_T,
                                    H, D, K1, K2, OUTPUT_MODE)
            x_offset = (((rows[:, None] * TOTAL_T + STEP_BASE + step) * 4) * H + head) * D + o[None, :]
            i = ai + tl.load(WX + x_offset, valid, 0).to(tl.float32) + bi[None, :]
            f = af + tl.load(WX + x_offset + H * D, valid, 0).to(tl.float32) + bf[None, :]
            z = az + tl.load(WX + x_offset + 2 * H * D, valid, 0).to(tl.float32) + bz[None, :]
            o_gate = ao + tl.load(WX + x_offset + 3 * H * D, valid, 0).to(tl.float32) + bo[None, :]
            z_value = 2.0 * tl.sigmoid(2.0 * z) - 1.0
            if SLSTM:
                logf = tl.minimum(f, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(f))) + m
                next_m = tl.where(
                    (SEQUENCE_START != 0) & (step == 0) & zero_normalizer,
                    i,
                    tl.maximum(i, logf),
                )
                igate = tl.exp(i - next_m)
                fgate = tl.exp(logf - next_m)
                c = fgate * c + igate * z_value
                n = tl.maximum(fgate * n + igate, 1.0)
                hidden = tl.sigmoid(o_gate) * c / n
                m = next_m
            else:
                c = tl.sigmoid(f) * c + tl.sigmoid(i) * z_value
                hidden = tl.sigmoid(o_gate) * (2.0 * tl.sigmoid(2.0 * c) - 1.0)
            if OUTPUT_MODE == 2:
                if STEP_BASE + step == TOTAL_T - 1:
                    tl.store(OUTPUT + state_offset, hidden, valid)
            else:
                output_offset = ((rows[:, None] * TOTAL_T + STEP_BASE + step) * H + head) * D + o[None, :]
                tl.store(OUTPUT + output_offset, hidden, valid)
                if OUTPUT_MODE == 0:
                    tl.store(OUTPUT + output_stride + output_offset, c, valid)
                    if SLSTM:
                        tl.store(OUTPUT + 2 * output_stride + output_offset, n, valid)
                        tl.store(OUTPUT + 3 * output_stride + output_offset, m, valid)
            if PERSISTENT != 0:
                if (step < T - 1) | (group + SLOTS < GROUPS):
                    parity = (bars + 1) % 2
                    if VERIFY & V_BROKEN_SINGLE:
                        parity = 0
                    write = XBUF + ((slot * 2 + parity) * BM + local[:, None]) * D + o[None, :]
                    if VERIFY & V_DELAY:
                        _verify_delay(seed, split + splits * slot, group, step, 0)
                    if VERIFY & V_BROKEN_ARRIVE_FIRST:
                        bars += 1
                        _group_barrier(SYNC, slot, bars * splits, ERR, SPIN)
                        tl.store(write, hidden.to(dtype), o_valid[None, :])
                    elif VERIFY & V_BROKEN_NO_BARRIER:
                        tl.store(write, hidden.to(dtype), o_valid[None, :])
                        bars += 1
                    else:
                        tl.store(write, hidden.to(dtype), o_valid[None, :])
                        bars += 1
                        _group_barrier(SYNC, slot, bars * splits, ERR, SPIN)
        tl.store(FINAL + state_offset, hidden, valid)
        tl.store(FINAL + stride + state_offset, c, valid)
        if SLSTM:
            tl.store(FINAL + 2 * stride + state_offset, n, valid)
            tl.store(FINAL + 3 * stride + state_offset, m, valid)


def verify_bits(options):
    """VERIFY bitmask and seed of a race-verification request (0, 0 when off)."""
    if not options:
        return 0, 0
    broken = {None: 0, "single": int(V_BROKEN_SINGLE), "no_barrier": int(V_BROKEN_NO_BARRIER),
              "arrive_first": int(V_BROKEN_ARRIVE_FIRST)}[options.get("broken")]
    seed = options.get("delay_seed")
    bits = (int(V_CHECK) if options.get("check", True) else 0) | (int(V_DELAY) if seed is not None else 0) | broken
    return bits, int(seed or 0)


def k_parts(width, pad=False):
    """Power-of-two reduction parts covering D exactly (or padded)."""
    top = 1 << (width.bit_length() - 1)
    if pad or top == width:
        return triton.next_power_of_2(width), 0
    rest = width - top
    if rest < 16 or rest & (rest - 1):
        return triton.next_power_of_2(width), 0
    return top, rest


def plan(batch, heads, width, slice_width, *, batch_tile=16, sms=None, max_ctas=None):
    splits = triton.cdiv(width, slice_width)
    groups = triton.cdiv(batch, batch_tile) * heads
    limit = sms if max_ctas is None else min(sms, max_ctas)
    if splits > limit:
        raise ValueError("R split exceeds the co-resident CTA bound")
    return splits, groups, min(groups, limit // splits)


def recurrence(
    wx: torch.Tensor,
    recurrent: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    cell: str = "lstm",
    *,
    sequence_context: SequenceContext,
    slice_width: int = 64,
    sync: str = "grid",
    num_warps: int = 4,
    output_mode: str = "full",
    snapshot_dtype: torch.dtype | None = None,
    pad_k: bool = False,
    max_ctas: int | None = None,
    diagnostics: dict[str, object] | None = None,
    _verify: dict[str, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward recurrence with R split along gate outputs across CTAs.

    ``_verify`` (race verification only, never served): {"check": bool, "delay_seed": int | None, "broken": None |
    "single" | "no_barrier" | "arrive_first"}; the violation counters are written back into the dict. The content
    check needs output_mode "full" or "hidden" with an FP32 snapshot.

    Always returns (output, FP32 compute carry) under the PR #79 contract.
    ``full`` returns [S,B,T,H,D] states, ``hidden`` [B,T,H,D], ``last`` [B,1,H,D].
    """
    if output_mode not in ("full", "hidden", "last"):
        raise ValueError("output mode must be full, hidden or last")
    if sync not in ("grid", "launch"):
        raise ValueError("sync must be grid or launch")
    if num_warps not in (4, 8):
        raise ValueError("warp count must be 4 or 8")
    if cell not in ("lstm", "slstm"):
        raise ValueError("R tiling supports LSTM and sLSTM")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("R tiling requires CUDA FP16/BF16 inputs")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (wx, recurrent, bias, initial)):
        raise ValueError("R tiling has no backward implementation")
    contract = state_contract(cell, wx.dtype)
    if sequence_context is None or sequence_context.contract != contract:
        raise ValueError("R tiling requires a matching full logical-call sequence context")
    if sequence_context.zero_normalizer.device != wx.device:
        raise ValueError("sequence context must be on the input device")
    if snapshot_dtype not in (None, wx.dtype, torch.float32):
        raise ValueError("snapshot dtype must be input dtype or FP32")
    batch, steps, gates, heads, width = wx.shape
    states = len(contract.components)
    if gates != 4 or steps < 1 or batch < 1 or heads < 1 or width < 16 or width % 16:
        raise ValueError("R tiling needs nonempty B/T/H and D a multiple of 16")
    if slice_width not in (16, 32, 64, 128) or slice_width > triton.next_power_of_2(width):
        raise ValueError("slice width must be a power of two in [16, 128]")
    if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
        raise ValueError("expected public weight and bias layouts")
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("expected public initial state layout")
    if recurrent.dtype != wx.dtype or bias.dtype != wx.dtype:
        raise ValueError("weights must match the input dtype")
    if initial.dtype not in (wx.dtype, torch.float32):
        raise ValueError("initial state must be input dtype or FP32 carry")
    if sequence_context.step_offset > 0 and initial.dtype != torch.float32:
        raise ValueError("continuation requires compute carry, not a rounded snapshot")
    if any(x.device != wx.device for x in (recurrent, bias, initial)):
        raise ValueError("all inputs must share device")
    wx, recurrent, bias, initial = (x.contiguous() for x in (wx, recurrent, bias, initial))
    device = wx.device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    splits, groups, slots = plan(batch, heads, width, slice_width, sms=sms, max_ctas=max_ctas)
    k1, k2 = k_parts(width, pad_k)
    output = torch.empty(
        (states, batch, steps, heads, width) if output_mode == "full"
        else (batch, 1 if output_mode == "last" else steps, heads, width),
        device=device,
        dtype=wx.dtype if snapshot_dtype is None else snapshot_dtype,
    )
    carries = [torch.empty(initial.shape, device=device, dtype=torch.float32) for _ in range(2)]
    xbuf = torch.empty((slots, 2, 16, width), device=device, dtype=wx.dtype)
    launches = 1 if sync == "grid" else steps
    sync_counts = torch.zeros((launches, slots), device=device, dtype=torch.int32)
    verify, seed = verify_bits(_verify)
    if verify & 1 and (output_mode == "last" or snapshot_dtype != torch.float32):
        raise ValueError("the race check reads the FP32 hidden output")
    err = torch.zeros((4 if verify else 1,), device=device, dtype=torch.int32)
    if verify:
        err[3] = seed
    mode = {"full": 0, "hidden": 1, "last": 2}[output_mode]
    start = sequence_context.step_offset == 0
    previous = initial
    kernel = None
    for index in range(launches):
        length = steps if sync == "grid" else 1
        base = 0 if sync == "grid" else index
        carry = carries[index % 2]
        kernel = _rtiled[(splits, slots)](
            wx, recurrent, bias, previous, output, carry, xbuf, sync_counts[index], err,
            sequence_context.zero_normalizer,
            batch, length, steps, base, int(start and base == 0), groups, slots,
            int(sync == "grid"), heads, width, cell == "slstm", 16, slice_width, k1, k2, mode, SPIN_LIMIT,
            num_warps=num_warps,
            enable_fp_fusion=False,
            # The driver rejects a non-co-resident grid instead of deadlocking.
            # Both sync modes share the compiled kernel, so both launch cooperatively.
            launch_cooperative_grid=True,
            **(dict(VERIFY=verify) if verify else {}),
        )
        previous = carry
    if verify:
        counts = err.tolist()
        _verify.update(barrier_error=counts[0], after_barrier=counts[1], after_mma=counts[2])
    if diagnostics is not None:
        per_slot = [len(range(s, groups, slots)) * (steps if sync == "grid" else 1) for s in range(slots)]
        diagnostics.update(
            registers=kernel.n_regs,
            spills=kernel.n_spills,
            shared_bytes=kernel.metadata.shared,
            ptx=kernel.asm["ptx"],
            splits=splits,
            slice_width=slice_width,
            k_parts=(k1, k2),
            groups=groups,
            slots=slots,
            ctas=splits * slots,
            sms=sms,
            synchronization="group_counter_barrier" if sync == "grid" else "kernel_boundary",
            sync_counts=sync_counts,
            expected_sync_counts=[splits * (n - 1) if sync == "grid" else 0 for n in per_slot],
            error_flag=err,
        )
    return output, previous
