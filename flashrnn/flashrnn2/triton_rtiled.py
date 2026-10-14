"""Recurrent-weight tiling across CTAs for generic LSTM/sLSTM inference, D > 128.

R is split along its output (gate) axis: CTA ``s`` of a group holds
R[:, head, s*S:(s+1)*S, :] for all four gates in shared memory and the matching
c/n/m slices in registers for the whole time loop. Only the rounded hidden
operand h_t crosses CTAs. It is exchanged through a double-buffered L2-resident
global buffer (loads bypass L1) and ordered by a per-group counter barrier
(release arrive, acquire spin) once per step.

All CTAs of a launch must be co-resident. The launcher bounds the grid to one
CTA per SM and lets each slot of k CTAs process several (batch tile, head)
groups serially. A bounded spin raises an error flag instead of hanging.

``sync="launch"`` replaces every in-kernel barrier by a kernel boundary (one
launch per step through the FP32 compute carry). It performs the same
arithmetic and is the bitwise oracle for the persistent ``sync="grid"`` path.
Forward only; FlashRNN generic semantics only (no TiRex/xLSTM arithmetic).
"""

import torch
import triton
import triton.language as tl

from .state_contract import SequenceContext, state_contract

SPIN_LIMIT = 1 << 26


@triton.jit
def _group_barrier(SYNC, slot, target, ERR, SPIN: tl.constexpr):
    # Every thread's exchange stores precede the single-thread release arrive.
    tl.debug_barrier()
    tl.atomic_add(SYNC + slot, 1, sem="release", scope="gpu")
    seen = tl.atomic_add(SYNC + slot, 0, sem="acquire", scope="gpu")
    spins = seen * 0
    while (seen < target) & (spins < SPIN):
        seen = tl.atomic_add(SYNC + slot, 0, sem="acquire", scope="gpu")
        spins += 1
    if seen < target:
        tl.atomic_max(ERR, 1)
    tl.debug_barrier()


@triton.jit
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
    H: tl.constexpr,
    D: tl.constexpr,
    SLSTM: tl.constexpr,
    BM: tl.constexpr,
    S: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    OUTPUT_MODE: tl.constexpr,
    PERSISTENT: tl.constexpr,
    SPIN: tl.constexpr,
):
    split = tl.program_id(0)
    slot = tl.program_id(1)
    splits = tl.num_programs(0)
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
                read = XBUF + ((slot * 2 + bars % 2) * BM + local[:, None]) * D
                h1 = tl.load(read + k1[None, :], k1[None, :] < D, 0, cache_modifier=".cg")
                if K2 > 0:
                    h2 = tl.load(read + k2[None, :], k2[None, :] < D, 0, cache_modifier=".cg")
            ai = tl.dot(h1, ri1)
            af = tl.dot(h1, rf1)
            az = tl.dot(h1, rz1)
            ao = tl.dot(h1, ro1)
            if K2 > 0:
                ai = tl.dot(h2, ri2, ai)
                af = tl.dot(h2, rf2, af)
                az = tl.dot(h2, rz2, az)
                ao = tl.dot(h2, ro2, ao)
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
            if PERSISTENT:
                if (step < T - 1) | (group + SLOTS < GROUPS):
                    write = XBUF + ((slot * 2 + (bars + 1) % 2) * BM + local[:, None]) * D + o[None, :]
                    tl.store(write, hidden.to(dtype), o_valid[None, :])
                    bars += 1
                    _group_barrier(SYNC, slot, bars * splits, ERR, SPIN)
        tl.store(FINAL + state_offset, hidden, valid)
        tl.store(FINAL + stride + state_offset, c, valid)
        if SLSTM:
            tl.store(FINAL + 2 * stride + state_offset, n, valid)
            tl.store(FINAL + 3 * stride + state_offset, m, valid)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward recurrence with R split along gate outputs across CTAs.

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
    err = torch.zeros((1,), device=device, dtype=torch.int32)
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
            heads, width, cell == "slstm", 16, slice_width, k1, k2, mode,
            sync == "grid", SPIN_LIMIT,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
        previous = carry
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
