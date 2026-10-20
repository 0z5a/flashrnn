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
                            _group_arrive, _group_barrier, _group_wait, _verify_delay, k_parts, plan, verify_bits)


@triton.jit
def _now():
    """PROF: global nanosecond timer, read after a block barrier so that every warp has finished the phase."""
    tl.debug_barrier()
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _dg_dot(SRC, rows, cols, mask, w, acc, ROUND: tl.constexpr):
    g = tl.load(SRC + rows[:, None] + cols[None, :], mask, 0, cache_modifier=".cg")
    if ROUND:
        g = g.to(w.dtype)
    return tl.dot(g, w, acc)


@triton.jit
def _dh_gate_batched(SRC, src_row, row_mask, wg1, wg2, gate_stride, K1: tl.constexpr, K2: tl.constexpr,
                     D: tl.constexpr, ROUND: tl.constexpr):
    """SPLITG: the four gates as one batched MMA [4, BM, K] x [4, K, S] -> [4, BM, S], summed over the gate axis, so
    that the warps split the reduction by gate instead of repeating the same [BM, S] tile."""
    g = tl.arange(0, 4)
    k1 = tl.arange(0, K1)
    m1 = row_mask[None, :, None] & (k1[None, None, :] < D) & (g[:, None, None] < 4)
    a = tl.load(SRC + src_row[None, :, None] + g[:, None, None] * gate_stride + k1[None, None, :], m1, 0,
                cache_modifier=".cg")
    if ROUND:
        a = a.to(wg1.dtype)
    acc = tl.dot(a, wg1)
    if K2 > 0:
        k2 = K1 + tl.arange(0, K2)
        m2 = row_mask[None, :, None] & (k2[None, None, :] < D) & (g[:, None, None] < 4)
        a = tl.load(SRC + src_row[None, :, None] + g[:, None, None] * gate_stride + k2[None, None, :], m2, 0,
                    cache_modifier=".cg")
        if ROUND:
            a = a.to(wg2.dtype)
        acc = tl.dot(a, wg2, acc)
    return tl.sum(acc, 0)


@triton.jit
def _dh_from_dg(SRC, src_row, row_mask, w_i1, w_f1, w_z1, w_o1, w_i2, w_f2, w_z2, w_o2, gate_stride,
                K1: tl.constexpr, K2: tl.constexpr, D: tl.constexpr, ROUND: tl.constexpr, BM: tl.constexpr,
                S: tl.constexpr, SPLITG: tl.constexpr = 0, wg1=None, wg2=None):
    """sum_g dG_g[BM, D] @ R_g[D, S]; dG is read as FP32 and rounded here, or already in the input dtype.

    One gate operand is loaded per MMA so that only one [BM, K] operand is live at a time (SPLITG: _dh_gate_batched)."""
    if SPLITG:
        acc = _dh_gate_batched(SRC, src_row, row_mask, wg1, wg2, gate_stride, K1, K2, D, ROUND)
    else:
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


@triton.jit
def _step_inputs(PRE, STATES, INITIAL, DSTATES, here, pre, unit, valid, t, hd, state_stride, history_stride,
                 SLSTM: tl.constexpr, DS_C: tl.constexpr, DS_N: tl.constexpr, NO_PRE: tl.constexpr = 0):
    """Pointwise inputs of step t except the step's own c/n/m (FAST path; see _rtiled_bw). With NO_PRE the gate
    preactivations are not loaded (zeros; the FUSE path computes them with _fused_gates)."""
    if NO_PRE:
        gi = (here * 0).to(tl.float32)
        gf, gz, go = gi, gi, gi
    else:
        gi = tl.load(PRE + pre, valid, 0)
        gf = tl.load(PRE + pre + hd, valid, 0)
        gz = tl.load(PRE + pre + 2 * hd, valid, 0)
        go = tl.load(PRE + pre + 3 * hd, valid, 0)
    dsh = tl.load(DSTATES + here, valid, 0)
    dsc = gi * 0.0
    if DS_C:
        dsc = tl.load(DSTATES + history_stride + here, valid, 0)
    dsn = gi * 0.0
    n_prev = gi * 0.0 + 1.0
    m_prev = gi * 0.0
    if t > 0:
        c_prev = tl.load(STATES + history_stride + here - hd, valid, 0)
    else:
        c_prev = tl.load(INITIAL + state_stride + unit, valid, 0).to(tl.float32)
    if SLSTM:
        if DS_N:
            dsn = tl.load(DSTATES + 2 * history_stride + here, valid, 0)
        if t > 0:
            n_prev = tl.load(STATES + 2 * history_stride + here - hd, valid, 1.0)
            m_prev = tl.load(STATES + 3 * history_stride + here - hd, valid, 0)
        else:
            n_prev = tl.load(INITIAL + 2 * state_stride + unit, valid, 1.0).to(tl.float32)
            m_prev = tl.load(INITIAL + 3 * state_stride + unit, valid, 0).to(tl.float32)
    return gi, gf, gz, go, dsh, dsc, dsn, c_prev, n_prev, m_prev


@triton.jit
def _fused_gates(WX, HPREV, rows, row_valid, head, pre, valid, t, T, hd,
                 r_i1, r_f1, r_z1, r_o1, r_i2, r_f2, r_z2, r_o2, rc1, rc2, b_i, b_f, b_z, b_o,
                 H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, BM: tl.constexpr, K1: tl.constexpr,
                 K2: tl.constexpr, FUSE: tl.constexpr):
    """Gate preactivations of step t for this CTA's units, recomputed in the kernel (FUSE path):
    Wx_t + R[:, p, :] round(h_{t-1}) + b. HPREV[b, t] holds round(h_{t-1}) in the input dtype (the dR GEMM operand).
    FUSE = 1: one [BM, S] MMA per gate; FUSE = 2: one [BM, 4S] MMA over the gate-interleaved tile (column 4 p + g)."""
    src = HPREV + ((rows * T + t) * H + head) * D
    k1 = tl.arange(0, K1)
    h1 = tl.load(src[:, None] + k1[None, :], row_valid[:, None] & (k1[None, :] < D), 0)
    if FUSE == 2:
        acc = tl.dot(h1, rc1)
        if K2 > 0:
            k2 = K1 + tl.arange(0, K2)
            h2 = tl.load(src[:, None] + k2[None, :], row_valid[:, None] & (k2[None, :] < D), 0)
            acc = tl.dot(h2, rc2, acc)
        even, odd = tl.split(tl.reshape(acc, (BM, S, 2, 2)))  # column 4 p + 2 a + b -> split b, then a
        ai, az = tl.split(even)
        af, ao = tl.split(odd)
    else:
        ai = tl.dot(h1, r_i1)
        af = tl.dot(h1, r_f1)
        az = tl.dot(h1, r_z1)
        ao = tl.dot(h1, r_o1)
        if K2 > 0:
            k2 = K1 + tl.arange(0, K2)
            h2 = tl.load(src[:, None] + k2[None, :], row_valid[:, None] & (k2[None, :] < D), 0)
            ai = tl.dot(h2, r_i2, ai)
            af = tl.dot(h2, r_f2, af)
            az = tl.dot(h2, r_z2, az)
            ao = tl.dot(h2, r_o2, ao)
    gi = ai + tl.load(WX + pre, valid, 0).to(tl.float32) + b_i[None, :]
    gf = af + tl.load(WX + pre + hd, valid, 0).to(tl.float32) + b_f[None, :]
    gz = az + tl.load(WX + pre + 2 * hd, valid, 0).to(tl.float32) + b_z[None, :]
    go = ao + tl.load(WX + pre + 3 * hd, valid, 0).to(tl.float32) + b_o[None, :]
    return gi, gf, gz, go


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
    FAST: tl.constexpr = 0,
    DS_C: tl.constexpr = 1,
    DS_N: tl.constexpr = 1,
    FUSE: tl.constexpr = 0,
    SPLITG: tl.constexpr = 0,
    PROF: tl.constexpr = 0,
):
    # FAST (same values, same arithmetic): step t's c/n/m are step t+1's c_{t-1}/n_{t-1}/m_{t-1} (no reload), the
    # next step's pointwise inputs are loaded before this step's barrier, and the c/n history-gradient loads are
    # skipped when DS_C / DS_N say they are zero (an exact +0.0 is added instead).
    # FUSE (with FAST): PRE carries Wx (input dtype); the bias and round(h_{t-1}) [B, T, H, D] (input dtype) ride
    # after the exchange buffer in XBUF; each step's gate preactivations are recomputed in the kernel with the row
    # slice R[:, p, :], issued between the barrier's arrive and its wait so that the extra MMA overlaps the exchange.
    # FUSE = 1 runs one MMA per gate, FUSE = 2 one MMA over the four gates interleaved (see _fused_gates).
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
    if PROF:
        # PROF (persistent mode): per-CTA sums (ns) of the step phases, written behind ERR[32]:
        # 0 dh reduction (exchange load + MMA), 1 pointwise + dG store, 2 publish + next-step inputs (and the fused
        # recompute), 3 barrier (arrive/wait), 4 steps counted.
        p_dh = tl.zeros((), tl.int64)
        p_pw = tl.zeros((), tl.int64)
        p_pub = tl.zeros((), tl.int64)
        p_bar = tl.zeros((), tl.int64)
        p_n = tl.zeros((), tl.int64)
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
        if SPLITG:
            # [4, K, S] tiles: R[g, head, k, p] for every gate, every output row k, this CTA's columns p.
            g4 = tl.arange(0, 4)
            wg1 = tl.load(R + g4[:, None, None] * hd * D + w1[None, :, :], (g4[:, None, None] < 4) & w1_mask[None, :, :], 0)
            wg2 = wg1
            if K2 > 0:
                k2 = K1 + tl.arange(0, K2)
                w2 = (head * D + k2[:, None]) * D + p[None, :]
                w2_mask = (k2[:, None] < D) & p_valid[None, :]
                wg2 = tl.load(R + g4[:, None, None] * hd * D + w2[None, :, :],
                              (g4[:, None, None] < 4) & w2_mask[None, :, :], 0)
            w_i1, w_f1, w_z1, w_o1, w_i2, w_f2, w_z2, w_o2 = w1, w1, w1, w1, w1, w1, w1, w1
        else:
            wg1, wg2 = w1, w1
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
        if FUSE:
            # Tile [k, p] = R[g, head, p, k]: this CTA's output rows p, every input column k (forward layout).
            r1 = (head * D + p[None, :]) * D + k1[:, None]
            r1_mask = p_valid[None, :] & (k1[:, None] < D)
            if FUSE == 2:
                # Gate-interleaved tile [k, 4 j + g] = R[g, head, p_j, k].
                c = tl.arange(0, 4 * S)
                pc = split * S + c // 4
                rc = (c % 4) * hd * D + (head * D + pc[None, :]) * D
                rc1 = tl.load(R + rc + k1[:, None], (pc[None, :] < D) & (k1[:, None] < D), 0)
                rc2 = rc1
                if K2 > 0:
                    rc2 = tl.load(R + rc + k2[:, None], (pc[None, :] < D) & (k2[:, None] < D), 0)
                r_i1, r_f1, r_z1, r_o1, r_i2, r_f2, r_z2, r_o2 = rc1, rc1, rc1, rc1, rc1, rc1, rc1, rc1
            else:
                r_i1 = tl.load(R + r1, r1_mask, 0)
                r_f1 = tl.load(R + hd * D + r1, r1_mask, 0)
                r_z1 = tl.load(R + 2 * hd * D + r1, r1_mask, 0)
                r_o1 = tl.load(R + 3 * hd * D + r1, r1_mask, 0)
                r_i2, r_f2, r_z2, r_o2 = r_i1, r_f1, r_z1, r_o1
                if K2 > 0:
                    r2 = (head * D + p[None, :]) * D + k2[:, None]
                    r2_mask = p_valid[None, :] & (k2[:, None] < D)
                    r_i2 = tl.load(R + r2, r2_mask, 0)
                    r_f2 = tl.load(R + hd * D + r2, r2_mask, 0)
                    r_z2 = tl.load(R + 2 * hd * D + r2, r2_mask, 0)
                    r_o2 = tl.load(R + 3 * hd * D + r2, r2_mask, 0)
                rc1, rc2 = r_i1, r_i1
            bias_base = XBUF + SLOTS * 2 * BM * 4 * D + head * D + p
            b_i = tl.load(bias_base, p_valid, 0).to(tl.float32)
            b_f = tl.load(bias_base + hd, p_valid, 0).to(tl.float32)
            b_z = tl.load(bias_base + 2 * hd, p_valid, 0).to(tl.float32)
            b_o = tl.load(bias_base + 3 * hd, p_valid, 0).to(tl.float32)
            HPREV = XBUF + SLOTS * 2 * BM * 4 * D + 4 * hd
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
        if FAST:
            here0 = ((rows[:, None] * T + T_START) * H + head) * D + p[None, :]
            pre0 = ((rows[:, None] * T + T_START) * 4 * H + head) * D + p[None, :]
            q_gi, q_gf, q_gz, q_go, q_dh, q_dc, q_dn, q_cp, q_np, q_mp = _step_inputs(
                PRE, STATES, INITIAL, DSTATES, here0, pre0, unit, valid, T_START, hd, state_stride, history_stride,
                SLSTM, DS_C, DS_N, FUSE)
            if FUSE:
                q_gi, q_gf, q_gz, q_go = _fused_gates(
                    PRE, HPREV, rows, row_valid, head, pre0, valid, T_START, T, hd,
                    r_i1, r_f1, r_z1, r_o1, r_i2, r_f2, r_z2, r_o2, rc1, rc2, b_i, b_f, b_z, b_o,
                    H, D, S, BM, K1, K2, FUSE)
            q_ct = tl.load(STATES + history_stride + here0, valid, 0)
            q_nt = q_ct * 0.0 + 1.0
            q_mt = q_ct * 0.0
            if SLSTM:
                q_nt = tl.load(STATES + 2 * history_stride + here0, valid, 1.0)
                q_mt = tl.load(STATES + 3 * history_stride + here0, valid, 0)
        for index in range(NSTEPS):
            t = T_START - index
            if PROF:
                t_a = _now()
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
                                        D, K1, K2, D, False, BM, S, SPLITG, wg1, wg2)
                    if VERIFY & V_CHECK:
                        _verify_dg(XBUF, src_row, DG, dg_row, row_valid, ERR + 2, hd, K1, K2, D)
                else:
                    src_row = ((rows * T + t + 1) * 4 * H + head) * D
                    dh_in = _dh_from_dg(DG, src_row, row_valid, w_i1, w_f1, w_z1, w_o1, w_i2, w_f2, w_z2, w_o2,
                                        hd, K1, K2, D, True, BM, S, SPLITG, wg1, wg2)
            if PROF:
                t_b = _now()
            here = ((rows[:, None] * T + t) * H + head) * D + p[None, :]
            if FAST:
                pre = ((rows[:, None] * T + t) * 4 * H + head) * D + p[None, :]
                gi, gf, gz, go, c_t, c_prev = q_gi, q_gf, q_gz, q_go, q_ct, q_cp
                dh = dh_in + q_dh
                dc = dc + q_dc
            else:
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
                if FAST:
                    dn = dn + q_dn
                    n_t, m_t, n_prev, m_prev = q_nt, q_mt, q_np, q_mp
                else:
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
            if PROF:
                t_c = _now()
                t_d = t_c
            if FUSE:
                # Publish dG, arrive, compute the next step's inputs (incl. its gate preactivations), then wait.
                if PERSISTENT != 0:
                    parity = bars % 2
                    if VERIFY & V_BROKEN_SINGLE:
                        parity = 0
                    write = XBUF + ((slot * 2 + parity) * BM + local[:, None]) * (4 * D) + p[None, :]
                    if VERIFY & V_DELAY:
                        _verify_delay(seed, split + splits * slot, group, t, 0)
                    if VERIFY & V_BROKEN_ARRIVE_FIRST:
                        bars += 1
                        _group_arrive(SYNC, slot)
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
                        _group_arrive(SYNC, slot)
                if index + 1 < NSTEPS:
                    q_ct, q_nt, q_mt = c_prev, q_np, q_mp
                    q_gi, q_gf, q_gz, q_go, q_dh, q_dc, q_dn, q_cp, q_np, q_mp = _step_inputs(
                        PRE, STATES, INITIAL, DSTATES, here - hd, pre - 4 * hd, unit, valid, t - 1, hd, state_stride,
                        history_stride, SLSTM, DS_C, DS_N, FUSE)
                    q_gi, q_gf, q_gz, q_go = _fused_gates(
                        PRE, HPREV, rows, row_valid, head, pre - 4 * hd, valid, t - 1, T, hd,
                        r_i1, r_f1, r_z1, r_o1, r_i2, r_f2, r_z2, r_o2, rc1, rc2, b_i, b_f, b_z, b_o,
                        H, D, S, BM, K1, K2, FUSE)
                if PROF:
                    t_d = _now()
                if PERSISTENT != 0:
                    if VERIFY & V_BROKEN_NO_BARRIER:
                        pass
                    else:
                        _group_wait(SYNC, slot, bars * splits, ERR, SPIN)
            else:
                if FAST:
                    if index + 1 < NSTEPS:
                        # Next step's inputs, issued before this step's exchange so their latency hides behind it.
                        q_ct, q_nt, q_mt = c_prev, q_np, q_mp
                        q_gi, q_gf, q_gz, q_go, q_dh, q_dc, q_dn, q_cp, q_np, q_mp = _step_inputs(
                            PRE, STATES, INITIAL, DSTATES, here - hd, pre - 4 * hd, unit, valid, t - 1, hd, state_stride,
                            history_stride, SLSTM, DS_C, DS_N)
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
                    if PROF:
                        t_d = _now()
                    if VERIFY & V_BROKEN_ARRIVE_FIRST:
                        pass
                    elif VERIFY & V_BROKEN_NO_BARRIER:
                        bars += 1
                    else:
                        bars += 1
                        _group_barrier(SYNC, slot, bars * splits, ERR, SPIN)
            if PROF:
                t_e = _now()
                p_dh += t_b - t_a
                p_pw += t_c - t_b
                p_pub += t_d - t_c
                p_bar += t_e - t_d
                p_n += 1
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
                                    D, K1, K2, D, False, BM, S, SPLITG, wg1, wg2)
                if VERIFY & V_CHECK:
                    _verify_dg(XBUF, src_row, DG, dg_row, row_valid, ERR + 2, hd, K1, K2, D)
            else:
                src_row = ((rows * T) * 4 * H + head) * D
                dh_in = _dh_from_dg(DG, src_row, row_valid, w_i1, w_f1, w_z1, w_o1, w_i2, w_f2, w_z2, w_o2,
                                    hd, K1, K2, D, True, BM, S, SPLITG, wg1, wg2)
            tl.store(DINIT + unit, dh_in, valid)
            tl.store(DINIT + state_stride + unit, dc, valid)
            if SLSTM:
                tl.store(DINIT + 2 * state_stride + unit, dn, valid)
                tl.store(DINIT + 3 * state_stride + unit, tl.zeros((BM, S), tl.float32), valid)
        else:
            tl.store(CARRY + unit, dc, valid)
            if SLSTM:
                tl.store(CARRY + state_stride + unit, dn, valid)
    if PROF:
        prof = ERR + 32 + (split + splits * slot) * 8
        tl.atomic_add(prof, p_dh.to(tl.int32))
        tl.atomic_add(prof + 1, p_pw.to(tl.int32))
        tl.atomic_add(prof + 2, p_pub.to(tl.int32))
        tl.atomic_add(prof + 3, p_bar.to(tl.int32))
        tl.atomic_add(prof + 4, p_n.to(tl.int32))


def rounded_hidden(wx, initial, history, out=None):
    """round(h_{t-1}) for every step [B, T, H, D] in the input dtype (the forward MMA operand)."""
    batch, steps, gates, heads, width = wx.shape
    h_prev = torch.empty((batch, steps, heads, width), device=wx.device, dtype=wx.dtype) if out is None else out
    h_prev[:, :1] = initial[0]
    h_prev[:, 1:] = history[0][:, :-1]
    return h_prev


def fused_buffer(slots, wx, bias, initial, history):
    """Exchange buffer of the FUSE kernel: [exchange (SLOTS, 2, 16, 4D) | bias (4, H, D) | round(h_{t-1}) (B, T, H, D)],
    all in the input dtype; returns the flat buffer and the round(h_{t-1}) view (also the dR GEMM operand)."""
    batch, steps, gates, heads, width = wx.shape
    exchange, hd = slots * 2 * 16 * 4 * width, heads * width
    buf = torch.empty((exchange + 4 * hd + batch * steps * hd,), device=wx.device, dtype=wx.dtype)
    buf[exchange:exchange + 4 * hd] = bias.reshape(-1)
    h_prev = rounded_hidden(wx, initial, history, buf[exchange + 4 * hd:].view(batch, steps, heads, width))
    return buf, h_prev


def preactivations(wx, recurrent, bias, initial, history, tf32=False):
    """FP32 gate preactivations Wx_t + R round(h_{t-1}) + b from the forward's FP32 state history.

    The hidden operand is rounded to the input dtype as in the forward MMA; products of two input-dtype values are
    exact in FP32 (and in TF32), so only the summation order differs from the forward kernel. One head is a single
    [B*T, D] x [D, 4D] GEMM written straight into the [B, T, 4, 1, D] layout.
    """
    dtype = wx.dtype
    batch, steps, gates, heads, width = wx.shape
    h_prev = torch.empty((batch, steps, heads, width), device=wx.device, dtype=dtype)
    h_prev[:, :1] = initial[0]  # rounded to the input dtype on assignment, as in the forward MMA operand
    h_prev[:, 1:] = history[0][:, :-1]
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        if heads == 1:
            pre = torch.addmm(bias.float().reshape(1, gates * width), h_prev.reshape(batch * steps, width).float(),
                              recurrent.reshape(gates * width, width).float().t())
            pre = pre.view(batch, steps, gates, 1, width)
            pre += wx
        else:
            ry = torch.einsum("btnp,gnop->btgno", h_prev.float(), recurrent.float())
            pre = (wx.float() + ry + bias.float()).contiguous()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return pre, h_prev


def weight_gradients(dg, h_prev, tf32=False, method="bmm"):
    """dR[g,n,o,p] = sum_{b,t} dG[b,t,g,n,o] round(h_{t-1})[b,t,n,p] and dbias, FP32, no atomics.

    H > 1: method "bmm" (default) runs one strided batched GEMM per gate on views of dG and h (batch = head, no
    permuted copy of dG); "einsum" is the earlier path (permutes dG into a contiguous copy first).
    Method "bmm_lp" (opt-in, heads.md section 8.3): the GEMM operands are dG rounded to the input dtype and h as
    stored (input dtype, no FP32 copy of h), FP32 accumulation, result rounded to the input dtype (the dtype dR is
    returned in) -- FlashRNN1's arrangement; dbias stays an FP32 sum of the FP32 dG."""
    batch, steps, gates, heads, width = dg.shape
    if method == "bmm_lp":
        m = torch.backends.cuda.matmul  # FP32 accumulation: no reduced-precision split-K reductions
        saved = (m.allow_fp16_reduced_precision_reduction, m.allow_bf16_reduced_precision_reduction)
        m.allow_fp16_reduced_precision_reduction = m.allow_bf16_reduced_precision_reduction = False
        try:
            lp, bt = h_prev.dtype, batch * steps
            dgl = dg if dg.dtype == lp else dg.to(lp)
            if heads == 1:
                d_recurrent = (dgl.view(bt, gates * width).t() @ h_prev.reshape(bt, width)).float().view(
                    gates, 1, width, width)
            else:
                hp = h_prev.view(bt, heads, width).transpose(0, 1)  # [H, BT, D], input dtype
                dgv = dgl.view(bt, gates, heads, width)
                out = torch.empty((gates, heads, width, width), device=dg.device, dtype=lp)
                for g in range(gates):
                    torch.bmm(dgv[:, g].transpose(0, 1).transpose(1, 2), hp, out=out[g])
                d_recurrent = out.float()
        finally:
            m.allow_fp16_reduced_precision_reduction, m.allow_bf16_reduced_precision_reduction = saved
        return d_recurrent, torch.sum(dg, dim=(0, 1), dtype=torch.float32).view(gates, heads, width)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32  # TF32 rounds the FP32 dG operand
    try:
        if heads == 1:
            flat = dg.view(batch * steps, gates * width)
            d_recurrent = (flat.t() @ h_prev.reshape(batch * steps, width).float()).view(gates, 1, width, width)
            d_bias = flat.sum(dim=0).view(gates, 1, width)
        elif method == "einsum":
            d_recurrent = torch.einsum("btgno,btnp->gnop", dg, h_prev.float())
            d_bias = dg.sum(dim=(0, 1))
        else:
            bt = batch * steps
            hp = h_prev.float().view(bt, heads, width).transpose(0, 1)  # [H, BT, D]
            dgv = dg.view(bt, gates, heads, width)
            d_recurrent = torch.empty((gates, heads, width, width), device=dg.device, dtype=torch.float32)
            for g in range(gates):
                torch.bmm(dgv[:, g].transpose(0, 1).transpose(1, 2), hp, out=d_recurrent[g])  # [H,D,BT] x [H,BT,D]
            d_bias = dg.sum(dim=(0, 1))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return d_recurrent, d_bias


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
    fast: bool = False,
    fuse_pre: bool | int = False,
    dg_dtype: str | None = None,
    wgrad: str = "bmm",
    split_gates: bool = False,
    profile_steps: bool = False,
    history_grad_components: tuple[bool, ...] | None = None,
    tf32: bool = False,
):
    """Gradients (dWx FP32 [B,T,4,H,D], dR FP32, dbias FP32, d initial FP32 [S,B,1,H,D]) of one whole logical call.

    ``history`` is the forward's FP32 state history [S,B,T,H,D]; ``d_history`` and ``d_final`` are the FP32
    gradients of the history and of the final carry (zeros where a component is unused).
    ``fast`` selects the bitwise-identical kernel variant (no reload of the step's c/n/m, next-step inputs loaded
    before the barrier); ``history_grad_components`` (h, c, n) lets it skip history-gradient components that are
    zero. ``tf32`` runs the preactivation and dR GEMMs with TF32 (not bitwise identical). ``fuse_pre`` (with ``fast``)
    recomputes the gate preactivations inside the kernel between the barrier's arrive and wait instead of a separate
    GEMM (not bitwise identical to the GEMM path); 1 (or True) runs one MMA per gate, 2 one MMA over the four
    gates interleaved. ``split_gates`` runs the dh reduction as one gate-batched MMA summed over the gates (the warps
    split the reduction by gate); not bitwise identical to the per-gate accumulation. ``profile_steps`` (persistent
    mode, with ``diagnostics``) times the step phases in the kernel (block barriers around timer reads perturb the
    timing; compare with an unprofiled call) and returns them in ``diagnostics["phases_ns"]``.
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
    # With ``fast`` and history_grad_components saying that only h carries gradient, the kernel never reads the other
    # components, so d_history may be the h component alone ([1, B, T, H, D]; heads.md section 8.7).
    h_only = fast and history_grad_components is not None and not any(tuple(history_grad_components)[1:])
    if d_history.dtype != torch.float32 or d_history.shape[1:] != history.shape[1:] or not (
            d_history.shape[0] == states or (h_only and d_history.shape[0] == 1)):
        raise ValueError("expected an FP32 history gradient")
    if d_final.shape != initial.shape or d_final.dtype != torch.float32:
        raise ValueError("expected an FP32 final-carry gradient")
    if recurrent.dtype != wx.dtype:
        raise ValueError("weights must match the input dtype")
    wx, recurrent, bias, initial, history, d_history, d_final = (
        x.contiguous() for x in (wx, recurrent, bias, initial, history, d_history, d_final))
    device = wx.device
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if (diagnostics or {}).get("time") else None
    if events:
        events[0].record()
    fuse = int(fuse_pre)
    if fuse not in (0, 1, 2):
        raise ValueError("fuse_pre must be 0/False, 1/True or 2")
    if fuse and not fast:
        raise ValueError("fuse_pre needs the fast kernel variant")
    if dg_dtype not in (None, "input") or (dg_dtype == "input" and wgrad != "bmm_lp"):
        raise ValueError('dg_dtype must be None or "input", and "input" needs wgrad="bmm_lp"')
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    splits, groups, slots = plan(batch, heads, width, slice_width, sms=sms, max_ctas=max_ctas)
    if fuse:  # preactivations are recomputed in the kernel; bias and round(h_{t-1}) ride behind the exchange buffer
        xbuf, h_prev = fused_buffer(slots, wx, bias, initial, history)
        pre = wx
    else:
        pre, h_prev = preactivations(wx, recurrent, bias, initial, history, tf32)
        xbuf = torch.empty((slots, 2, 16, 4 * width), device=device, dtype=wx.dtype)
    if events:
        events[1].record()
    k1, k2 = k_parts(width, pad_k)
    # dg_dtype="input" (opt-in, heads.md section 8.5): the kernel stores dG in the input dtype. Every read of dG inside
    # the kernel rounds it to the input dtype first (_dh_from_dg, _verify_dg), so the recurrence is unchanged; the
    # returned dG is the input-dtype tensor the caller would otherwise cast to, and bmm_lp uses it directly.
    dg = torch.empty((batch, steps, 4, heads, width), device=device,
                     dtype=wx.dtype if dg_dtype == "input" else torch.float32)
    d_init = torch.empty((states, batch, 1, heads, width), device=device, dtype=torch.float32)
    carry = torch.zeros((2, batch, heads, width), device=device, dtype=torch.float32)
    launches = 1 if sync == "grid" else steps + 1
    sync_counts = torch.zeros((launches, slots), device=device, dtype=torch.int32)
    verify, seed = verify_bits(_verify)
    # Which history components can carry gradient (h, c, n); with ``fast`` the others are not read.
    ds = tuple(history_grad_components or (True, True, True)) + (True,) * 3
    if profile_steps and (sync != "grid" or diagnostics is None):
        raise ValueError("profile_steps needs the persistent (grid) mode and a diagnostics dict")
    err = torch.zeros((32 + 8 * splits * slots if profile_steps else 4 if verify else 1,), device=device,
                      dtype=torch.int32)
    if verify:
        err[3].fill_(seed)  # in place, no host copy: CUDA-graph safe
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
            **(dict(FAST=1, DS_C=int(ds[1]), DS_N=int(ds[2])) if fast else {}),
            **(dict(FUSE=fuse) if fuse else {}),
            **(dict(SPLITG=1) if split_gates else {}),
            **(dict(PROF=1) if profile_steps else {}),
        )
    if verify:
        counts = err.tolist()
        _verify.update(barrier_error=counts[0], after_barrier=counts[1], after_mma=counts[2])
    if events:
        events[2].record()
    d_recurrent, d_bias = weight_gradients(dg, h_prev, tf32, wgrad)
    if events:
        events[3].record()
        torch.cuda.synchronize()
        diagnostics["ms"] = dict(preactivation=events[0].elapsed_time(events[1]),
                                 kernel=events[1].elapsed_time(events[2]),
                                 weight_gradients=events[2].elapsed_time(events[3]))
    if diagnostics is not None:
        per_slot = [len(range(s, groups, slots)) * steps for s in range(slots)]
        diagnostics.update(
            registers=kernel.n_regs, spills=kernel.n_spills, shared_bytes=kernel.metadata.shared,
            splits=splits, slice_width=slice_width, k_parts=(k1, k2), groups=groups, slots=slots,
            ctas=splits * slots, sms=sms,
            synchronization="group_counter_barrier" if sync == "grid" else "kernel_boundary",
            sync_counts=sync_counts,
            expected_sync_counts=[splits * n if sync == "grid" else 0 for n in per_slot],
            error_flag=err[:1] if profile_steps else err,
        )
        if profile_steps:
            sums = err[32:].view(splits * slots, 8)[:, :5].tolist()
            diagnostics["phases_ns"] = dict(names=["dh", "pointwise", "publish", "barrier", "steps"], per_cta=sums)
    return dg, d_recurrent, d_bias, d_init
