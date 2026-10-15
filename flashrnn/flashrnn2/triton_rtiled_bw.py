"""Backward recurrence for LSTM/sLSTM with the recurrent weights tiled across CTAs (Triton).

Gradient convention as ``grad_reference``: the sLSTM stabilizer m carries no gradient and the normalizer clamp is
straight-through (FlashRNN v1).

Work split. The sequential kernel computes only the gate-preactivation gradients dG_t and the state gradients carried
from step t to t-1; dR, dbias and dWx are formed afterwards from dG with deterministic FP32 GEMMs/reductions.
CTA ``s`` of a group owns the hidden units p in [s*S, (s+1)*S): it holds the column slice R[:, head, :, p] (all four
gates, every output row) and the c/n carries of those units. Per step (t = T-1 ... 0) it

  1. receives dh_t for its units: dH_t (from the outputs) + sum_g dG_{t+1,g} R_g[:, p] (the GEMM over the full,
     rounded dG_{t+1} of every CTA of the group), or the final-carry gradient at t = T-1,
  2. applies the pointwise cell backward with the saved FP32 states and the recomputed preactivations,
  3. stores dG_t (FP32) and publishes it rounded to the input dtype, the operand of the next GEMM.

After step 0 the same GEMM gives the gradient of the initial hidden state. The exchange uses a double-buffered
L2-resident global buffer and the forward's counter barrier under cooperative launch (``sync="grid"``).
``sync="launch"`` runs one launch per step and takes the GEMM operand from the stored dG instead: the same values
and arithmetic, so it is the bitwise oracle of the persistent path. D <= 128 runs the same kernel with one CTA per
group.
"""

import torch
import triton
import triton.language as tl

from .triton_rtiled import (SPIN_LIMIT, V_BROKEN_ARRIVE_FIRST, V_BROKEN_NO_BARRIER, V_BROKEN_SINGLE, V_CHECK, V_DELAY,
                            _group_barrier, _verify_delay, k_parts, plan, verify_bits)


@triton.jit
def _dg_dot(SRC, rows, cols, mask, w, acc, ROUND: tl.constexpr):
    g = tl.load(SRC + rows[:, None] + cols[None, :], mask, 0, cache_modifier=".cg")
    if ROUND:
        g = g.to(w.dtype)
    return tl.dot(g, w, acc)


@triton.jit
def _dh_from_dg(SRC, src_row, row_mask, w_i1, w_f1, w_z1, w_o1, w_i2, w_f2, w_z2, w_o2, gate_stride,
                K1: tl.constexpr, K2: tl.constexpr, D: tl.constexpr, ROUND: tl.constexpr, BM: tl.constexpr,
                S: tl.constexpr):
    """sum_g dG_g[BM, D] @ R_g[D, S]; dG is read as FP32 and rounded here, or already in the input dtype.

    One gate operand is loaded per MMA so that only one [BM, K] operand is live at a time."""
    k1 = tl.arange(0, K1)
    m1 = row_mask[:, None] & (k1[None, :] < D)
    acc = tl.zeros((BM, S), tl.float32)
    acc = _dg_dot(SRC, src_row, k1, m1, w_i1, acc, ROUND)
    acc = _dg_dot(SRC, src_row + gate_stride, k1, m1, w_f1, acc, ROUND)
    acc = _dg_dot(SRC, src_row + 2 * gate_stride, k1, m1, w_z1, acc, ROUND)
    acc = _dg_dot(SRC, src_row + 3 * gate_stride, k1, m1, w_o1, acc, ROUND)
    if K2 > 0:
        k2 = K1 + tl.arange(0, K2)
        m2 = row_mask[:, None] & (k2[None, :] < D)
        acc = _dg_dot(SRC, src_row, k2, m2, w_i2, acc, ROUND)
        acc = _dg_dot(SRC, src_row + gate_stride, k2, m2, w_f2, acc, ROUND)
        acc = _dg_dot(SRC, src_row + 2 * gate_stride, k2, m2, w_z2, acc, ROUND)
        acc = _dg_dot(SRC, src_row + 3 * gate_stride, k2, m2, w_o2, acc, ROUND)
    return acc


@triton.jit
def _verify_dg(XBUF, src_row, DG, dg_row, row_valid, COUNTER, gate_stride, K1: tl.constexpr, K2: tl.constexpr,
               D: tl.constexpr):
    # Race verification: the exchanged operand must equal the rounded dG the senders stored (FP32) before publishing.
    k1 = tl.arange(0, K1)
    m1 = row_valid[:, None] & (k1[None, :] < D)
    bad = tl.sum(row_valid.to(tl.int32), 0) * 0
    for g in tl.static_range(4):
        x = tl.load(XBUF + src_row[:, None] + g * D + k1[None, :], m1, 0, cache_modifier=".cg")
        w = tl.load(DG + dg_row[:, None] + g * gate_stride + k1[None, :], m1, 0, cache_modifier=".cg").to(x.dtype)
        bad += tl.sum(tl.sum(((x != w) & m1).to(tl.int32), 1), 0)
        if K2 > 0:
            k2 = K1 + tl.arange(0, K2)
            m2 = row_valid[:, None] & (k2[None, :] < D)
            x = tl.load(XBUF + src_row[:, None] + g * D + k2[None, :], m2, 0, cache_modifier=".cg")
            w = tl.load(DG + dg_row[:, None] + g * gate_stride + k2[None, :], m2, 0, cache_modifier=".cg").to(x.dtype)
            bad += tl.sum(tl.sum(((x != w) & m2).to(tl.int32), 1), 0)
    tl.atomic_add(COUNTER, bad)


@triton.jit(do_not_specialize=["B", "T", "T_START", "NSTEPS", "FROM_FINAL", "TAIL", "GROUPS", "SLOTS",
                               "PERSISTENT"])
def _rtiled_bw(
    PRE,  # [B, T, 4, H, D] FP32 recomputed gate preactivations
    STATES,  # [S, B, T, H, D] FP32 forward state history
    INITIAL,  # [S, B, 1, H, D] initial state (input dtype or FP32)
    DSTATES,  # [S, B, T, H, D] FP32 gradient of the state history
    DFINAL,  # [S, B, 1, H, D] FP32 gradient of the final carry
    R,  # [4, H, D, D] input dtype
    DG,  # [B, T, 4, H, D] FP32 out: gate preactivation gradients
    DINIT,  # [S, B, 1, H, D] FP32 out: initial-state gradient
    CARRY,  # [2, B, H, D] FP32 dc/dn carry between launches (launch mode)
    XBUF,  # [SLOTS, 2, BM, 4 * D] input dtype exchange buffer (persistent mode)
    SYNC,
    ERR,
    B,
    T,
    T_START,
    NSTEPS,
    FROM_FINAL,
    TAIL,
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
    SPIN: tl.constexpr,
    VERIFY: tl.constexpr = 0,
):
    split = tl.program_id(0)
    slot = tl.program_id(1)
    splits = tl.num_programs(0)
    if VERIFY & V_DELAY:
        seed = tl.load(ERR + 3)
    local = tl.arange(0, BM)
    p = split * S + tl.arange(0, S)
    p_valid = p < D
    dtype = R.dtype.element_ty
    hd = H * D
    state_stride = B * hd
    history_stride = B * T * hd
    bars = split * 0
    for group in range(slot, GROUPS, SLOTS):
        rows = (group // H) * BM + local
        head = group % H
        row_valid = rows < B
        valid = row_valid[:, None] & p_valid[None, :]
        unit = (rows[:, None] * H + head) * D + p[None, :]  # [S, B, 1, H, D] / carry offsets
        # Tile [k, p] = R[g, head, k, p]: every output row k, this CTA's input columns p (MMA operand b).
        k1 = tl.arange(0, K1)
        w1 = (head * D + k1[:, None]) * D + p[None, :]
        w1_mask = (k1[:, None] < D) & p_valid[None, :]
        w_i1 = tl.load(R + w1, w1_mask, 0)
        w_f1 = tl.load(R + hd * D + w1, w1_mask, 0)
        w_z1 = tl.load(R + 2 * hd * D + w1, w1_mask, 0)
        w_o1 = tl.load(R + 3 * hd * D + w1, w1_mask, 0)
        w_i2, w_f2, w_z2, w_o2 = w_i1, w_f1, w_z1, w_o1
        if K2 > 0:
            k2 = K1 + tl.arange(0, K2)
            w2 = (head * D + k2[:, None]) * D + p[None, :]
            w2_mask = (k2[:, None] < D) & p_valid[None, :]
            w_i2 = tl.load(R + w2, w2_mask, 0)
            w_f2 = tl.load(R + hd * D + w2, w2_mask, 0)
            w_z2 = tl.load(R + 2 * hd * D + w2, w2_mask, 0)
            w_o2 = tl.load(R + 3 * hd * D + w2, w2_mask, 0)
        if FROM_FINAL != 0:
            dh_in = tl.load(DFINAL + unit, valid, 0)
            dc = tl.load(DFINAL + state_stride + unit, valid, 0)
            dn = tl.zeros((BM, S), tl.float32)
            if SLSTM:
                dn = tl.load(DFINAL + 2 * state_stride + unit, valid, 0)
        else:
            dh_in = tl.zeros((BM, S), tl.float32)
            dc = tl.load(CARRY + unit, valid, 0)
            dn = tl.zeros((BM, S), tl.float32)
            if SLSTM:
                dn = tl.load(CARRY + state_stride + unit, valid, 0)
        for index in range(NSTEPS):
            t = T_START - index
            if (index > 0) | (FROM_FINAL == 0):
                # dh_t from the exchanged, rounded dG_{t+1} of every CTA of the group.
                if PERSISTENT != 0:
                    parity = (bars + 1) % 2
                    if VERIFY & V_BROKEN_SINGLE:
                        parity = 0
                    if VERIFY & V_DELAY:
                        _verify_delay(seed, split + splits * slot, group, t, 1)
                    src_row = ((slot * 2 + parity) * BM + local) * (4 * D)
                    dg_row = ((rows * T + t + 1) * 4 * H + head) * D
                    if VERIFY & V_CHECK:
                        _verify_dg(XBUF, src_row, DG, dg_row, row_valid, ERR + 1, hd, K1, K2, D)
                    dh_in = _dh_from_dg(XBUF, src_row, local < BM, w_i1, w_f1, w_z1, w_o1, w_i2, w_f2, w_z2, w_o2,
                                        D, K1, K2, D, False, BM, S)
                    if VERIFY & V_CHECK:
                        _verify_dg(XBUF, src_row, DG, dg_row, row_valid, ERR + 2, hd, K1, K2, D)
                else:
                    src_row = ((rows * T + t + 1) * 4 * H + head) * D
                    dh_in = _dh_from_dg(DG, src_row, row_valid, w_i1, w_f1, w_z1, w_o1, w_i2, w_f2, w_z2, w_o2,
                                        hd, K1, K2, D, True, BM, S)
            here = ((rows[:, None] * T + t) * H + head) * D + p[None, :]
            dh = dh_in + tl.load(DSTATES + here, valid, 0)
            dc = dc + tl.load(DSTATES + history_stride + here, valid, 0)
            pre = ((rows[:, None] * T + t) * 4 * H + head) * D + p[None, :]
            gi = tl.load(PRE + pre, valid, 0)
            gf = tl.load(PRE + pre + hd, valid, 0)
            gz = tl.load(PRE + pre + 2 * hd, valid, 0)
            go = tl.load(PRE + pre + 3 * hd, valid, 0)
            c_t = tl.load(STATES + history_stride + here, valid, 0)
            if t > 0:
                c_prev = tl.load(STATES + history_stride + here - hd, valid, 0)
            else:
                c_prev = tl.load(INITIAL + state_stride + unit, valid, 0).to(tl.float32)
            z = 2.0 * tl.sigmoid(2.0 * gz) - 1.0
            o = tl.sigmoid(go)
            if SLSTM:
                dn = dn + tl.load(DSTATES + 2 * history_stride + here, valid, 0)
                n_t = tl.load(STATES + 2 * history_stride + here, valid, 1.0)
                m_t = tl.load(STATES + 3 * history_stride + here, valid, 0)
                if t > 0:
                    n_prev = tl.load(STATES + 2 * history_stride + here - hd, valid, 0)
                    m_prev = tl.load(STATES + 3 * history_stride + here - hd, valid, 0)
                else:
                    n_prev = tl.load(INITIAL + 2 * state_stride + unit, valid, 0).to(tl.float32)
                    m_prev = tl.load(INITIAL + 3 * state_stride + unit, valid, 0).to(tl.float32)
                logf = tl.minimum(gf, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(gf))) + m_prev
                fgate = tl.exp(logf - m_t)
                igate = tl.exp(gi - m_t)
                dc = dc + dh * o / n_t
                dn = dn - dh * o * c_t / (n_t * n_t)
                d_o = dh * (c_t / n_t) * o * (1.0 - o)
                d_f = (dc * c_prev + dn * n_prev) * fgate * (1.0 - tl.sigmoid(gf))
                d_i = (dc * z + dn) * igate
                d_z = dc * igate * (1.0 - z * z)
                dc = dc * fgate
                dn = dn * fgate
            else:
                igate = tl.sigmoid(gi)
                fgate = tl.sigmoid(gf)
                tc = 2.0 * tl.sigmoid(2.0 * c_t) - 1.0
                dc = dc + dh * o * (1.0 - tc * tc)
                d_o = dh * tc * o * (1.0 - o)
                d_f = dc * c_prev * fgate * (1.0 - fgate)
                d_i = dc * z * igate * (1.0 - igate)
                d_z = dc * igate * (1.0 - z * z)
                dc = dc * fgate
            tl.store(DG + pre, d_i, valid)
            tl.store(DG + pre + hd, d_f, valid)
            tl.store(DG + pre + 2 * hd, d_z, valid)
            tl.store(DG + pre + 3 * hd, d_o, valid)
            if PERSISTENT != 0:
                parity = bars % 2
                if VERIFY & V_BROKEN_SINGLE:
                    parity = 0
                write = XBUF + ((slot * 2 + parity) * BM + local[:, None]) * (4 * D) + p[None, :]
                if VERIFY & V_DELAY:
                    _verify_delay(seed, split + splits * slot, group, t, 0)
                if VERIFY & V_BROKEN_ARRIVE_FIRST:
                    bars += 1
                    _group_barrier(SYNC, slot, bars * splits, ERR, SPIN)
                tl.store(write, d_i.to(dtype), p_valid[None, :])
                tl.store(write + D, d_f.to(dtype), p_valid[None, :])
                tl.store(write + 2 * D, d_z.to(dtype), p_valid[None, :])
                tl.store(write + 3 * D, d_o.to(dtype), p_valid[None, :])
                if VERIFY & V_BROKEN_ARRIVE_FIRST:
                    pass
                elif VERIFY & V_BROKEN_NO_BARRIER:
                    bars += 1
                else:
                    bars += 1
                    _group_barrier(SYNC, slot, bars * splits, ERR, SPIN)
        if TAIL != 0:
            if PERSISTENT != 0:
                parity = (bars + 1) % 2
                if VERIFY & V_BROKEN_SINGLE:
                    parity = 0
                if VERIFY & V_DELAY:
                    _verify_delay(seed, split + splits * slot, group, -1, 1)
                src_row = ((slot * 2 + parity) * BM + local) * (4 * D)
                dg_row = ((rows * T) * 4 * H + head) * D
                if VERIFY & V_CHECK:
                    _verify_dg(XBUF, src_row, DG, dg_row, row_valid, ERR + 1, hd, K1, K2, D)
                dh_in = _dh_from_dg(XBUF, src_row, local < BM, w_i1, w_f1, w_z1, w_o1, w_i2, w_f2, w_z2, w_o2,
                                    D, K1, K2, D, False, BM, S)
                if VERIFY & V_CHECK:
                    _verify_dg(XBUF, src_row, DG, dg_row, row_valid, ERR + 2, hd, K1, K2, D)
            else:
                src_row = ((rows * T) * 4 * H + head) * D
                dh_in = _dh_from_dg(DG, src_row, row_valid, w_i1, w_f1, w_z1, w_o1, w_i2, w_f2, w_z2, w_o2,
                                    hd, K1, K2, D, True, BM, S)
            tl.store(DINIT + unit, dh_in, valid)
            tl.store(DINIT + state_stride + unit, dc, valid)
            if SLSTM:
                tl.store(DINIT + 2 * state_stride + unit, dn, valid)
                tl.store(DINIT + 3 * state_stride + unit, tl.zeros((BM, S), tl.float32), valid)
        else:
            tl.store(CARRY + unit, dc, valid)
            if SLSTM:
                tl.store(CARRY + state_stride + unit, dn, valid)


def preactivations(wx, recurrent, bias, initial, history):
    """FP32 gate preactivations Wx_t + R round(h_{t-1}) + b from the forward's FP32 state history.

    The hidden operand is rounded to the input dtype as in the forward MMA; products of two input-dtype values are
    exact in FP32, so only the summation order differs from the forward kernel.
    """
    dtype = wx.dtype
    h_prev = torch.cat((initial[0].to(torch.float32), history[0][:, :-1]), dim=1).to(dtype)  # [B, T, H, D]
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ry = torch.einsum("btnp,gnop->btgno", h_prev.float(), recurrent.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return (wx.float() + ry + bias.float()).contiguous(), h_prev


def weight_gradients(dg, h_prev):
    """dR[g,n,o,p] = sum_{b,t} dG[b,t,g,n,o] round(h_{t-1})[b,t,n,p] and dbias, FP32, no TF32, no atomics."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        d_recurrent = torch.einsum("btgno,btnp->gnop", dg, h_prev.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return d_recurrent, dg.sum(dim=(0, 1))


def backward(
    wx: torch.Tensor,
    recurrent: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    history: torch.Tensor,
    d_history: torch.Tensor,
    d_final: torch.Tensor,
    cell: str = "lstm",
    *,
    slice_width: int = 64,
    sync: str = "grid",
    num_warps: int = 4,
    pad_k: bool = False,
    max_ctas: int | None = None,
    diagnostics: dict[str, object] | None = None,
    _verify: dict[str, object] | None = None,
):
    """Gradients (dWx FP32 [B,T,4,H,D], dR FP32, dbias FP32, d initial FP32 [S,B,1,H,D]) of one whole logical call.

    ``history`` is the forward's FP32 state history [S,B,T,H,D]; ``d_history`` and ``d_final`` are the FP32
    gradients of the history and of the final carry (zeros where a component is unused).
    """
    if sync not in ("grid", "launch"):
        raise ValueError("sync must be grid or launch")
    if num_warps not in (4, 8):
        raise ValueError("warp count must be 4 or 8")
    if cell not in ("lstm", "slstm"):
        raise ValueError("the R-tiled backward supports LSTM and sLSTM")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("the R-tiled backward requires CUDA FP16/BF16 inputs")
    batch, steps, gates, heads, width = wx.shape
    states = 4 if cell == "slstm" else 2
    if gates != 4 or steps < 1 or batch < 1 or heads < 1 or width < 16 or width % 16:
        raise ValueError("needs nonempty B/T/H and D a multiple of 16")
    if slice_width not in (16, 32, 64, 128) or slice_width > triton.next_power_of_2(width):
        raise ValueError("slice width must be a power of two in [16, 128]")
    if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
        raise ValueError("expected public weight and bias layouts")
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("expected public initial state layout")
    if history.shape != (states, batch, steps, heads, width) or history.dtype != torch.float32:
        raise ValueError("expected the forward's FP32 state history")
    if d_history.shape != history.shape or d_history.dtype != torch.float32:
        raise ValueError("expected an FP32 history gradient")
    if d_final.shape != initial.shape or d_final.dtype != torch.float32:
        raise ValueError("expected an FP32 final-carry gradient")
    if recurrent.dtype != wx.dtype:
        raise ValueError("weights must match the input dtype")
    wx, recurrent, bias, initial, history, d_history, d_final = (
        x.contiguous() for x in (wx, recurrent, bias, initial, history, d_history, d_final))
    device = wx.device
    pre, h_prev = preactivations(wx, recurrent, bias, initial, history)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    splits, groups, slots = plan(batch, heads, width, slice_width, sms=sms, max_ctas=max_ctas)
    k1, k2 = k_parts(width, pad_k)
    dg = torch.empty((batch, steps, 4, heads, width), device=device, dtype=torch.float32)
    d_init = torch.empty((states, batch, 1, heads, width), device=device, dtype=torch.float32)
    carry = torch.zeros((2, batch, heads, width), device=device, dtype=torch.float32)
    xbuf = torch.empty((slots, 2, 16, 4 * width), device=device, dtype=wx.dtype)
    launches = 1 if sync == "grid" else steps + 1
    sync_counts = torch.zeros((launches, slots), device=device, dtype=torch.int32)
    verify, seed = verify_bits(_verify)
    err = torch.zeros((4 if verify else 1,), device=device, dtype=torch.int32)
    if verify:
        err[3] = seed
    kernel = None
    for index in range(launches):
        if sync == "grid":
            t_start, nsteps, from_final, tail = steps - 1, steps, 1, 1
        elif index < steps:
            t_start, nsteps, from_final, tail = steps - 1 - index, 1, int(index == 0), 0
        else:
            t_start, nsteps, from_final, tail = 0, 0, 0, 1
        kernel = _rtiled_bw[(splits, slots)](
            pre, history, initial, d_history, d_final, recurrent, dg, d_init, carry, xbuf, sync_counts[index], err,
            batch, steps, t_start, nsteps, from_final, tail, groups, slots, int(sync == "grid"),
            heads, width, cell == "slstm", 16, slice_width, k1, k2, SPIN_LIMIT,
            num_warps=num_warps,
            enable_fp_fusion=False,
            launch_cooperative_grid=True,
            **(dict(VERIFY=verify) if verify else {}),
        )
    if verify:
        counts = err.tolist()
        _verify.update(barrier_error=counts[0], after_barrier=counts[1], after_mma=counts[2])
    d_recurrent, d_bias = weight_gradients(dg, h_prev)
    if diagnostics is not None:
        per_slot = [len(range(s, groups, slots)) * steps for s in range(slots)]
        diagnostics.update(
            registers=kernel.n_regs, spills=kernel.n_spills, shared_bytes=kernel.metadata.shared,
            splits=splits, slice_width=slice_width, k_parts=(k1, k2), groups=groups, slots=slots,
            ctas=splits * slots, sms=sms,
            synchronization="group_counter_barrier" if sync == "grid" else "kernel_boundary",
            sync_counts=sync_counts,
            expected_sync_counts=[splits * n if sync == "grid" else 0 for n in per_slot],
            error_flag=err,
        )
    return dg, d_recurrent, d_bias, d_init
