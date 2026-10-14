"""Experimental one-row CTA recurrence for small-batch generic LSTM/sLSTM.

The FP32 reduction tree differs from Tensor Core accumulation. Admission must
use the unchanged numerical budget; this is never selected by default.
"""
import triton
import triton.language as tl


@triton.jit
def _sequence_simt(
    WX, R, BIAS, INITIAL, OUTPUT, FINAL,
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    SLSTM: tl.constexpr, ZERO_NORMALIZER, BD: tl.constexpr,
    TOTAL_T: tl.constexpr, STEP_BASE: tl.constexpr, OUTPUT_MODE: tl.constexpr,
    SEQUENCE_START: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BD)
    valid = d < D
    offset = (row * H + head) * D + d
    stride = B * H * D
    hidden = tl.load(INITIAL + offset, valid, 0).to(tl.float32)
    c = tl.load(INITIAL + stride + offset, valid, 0).to(tl.float32)
    n = tl.full((BD,), 1.0, tl.float32)
    m = tl.zeros((BD,), tl.float32)
    if SLSTM:
        n = tl.load(INITIAL + 2 * stride + offset, valid, 0).to(tl.float32)
        m = tl.load(INITIAL + 3 * stride + offset, valid, 0).to(tl.float32)
        zero = tl.load(ZERO_NORMALIZER)
    # Public R is [gate,head,output,input]. Keep the reduction axis contiguous.
    roffset = (head * D + d[:, None]) * D + d[None, :]
    rmask = (d[:, None] < D) & (d[None, :] < D)
    ri = tl.load(R + roffset, rmask, 0).to(tl.float32)
    rf = tl.load(R + H * D * D + roffset, rmask, 0).to(tl.float32)
    rz = tl.load(R + 2 * H * D * D + roffset, rmask, 0).to(tl.float32)
    rr = tl.load(R + 3 * H * D * D + roffset, rmask, 0).to(tl.float32)
    bi = tl.load(BIAS + head * D + d, valid, 0).to(tl.float32)
    bf = tl.load(BIAS + H * D + head * D + d, valid, 0).to(tl.float32)
    bz = tl.load(BIAS + 2 * H * D + head * D + d, valid, 0).to(tl.float32)
    bo = tl.load(BIAS + 3 * H * D + head * D + d, valid, 0).to(tl.float32)
    for step in range(T):
        h = hidden.to(WX.dtype.element_ty).to(tl.float32)
        xoff = (((row * TOTAL_T + STEP_BASE + step) * 4) * H + head) * D + d
        i = tl.sum(ri * h[None, :], 1) + tl.load(WX + xoff, valid, 0).to(tl.float32) + bi
        f = tl.sum(rf * h[None, :], 1) + tl.load(WX + xoff + H * D, valid, 0).to(tl.float32) + bf
        z = tl.sum(rz * h[None, :], 1) + tl.load(WX + xoff + 2 * H * D, valid, 0).to(tl.float32) + bz
        o = tl.sum(rr * h[None, :], 1) + tl.load(WX + xoff + 3 * H * D, valid, 0).to(tl.float32) + bo
        zv = 2.0 * tl.sigmoid(2.0 * z) - 1.0
        if SLSTM:
            logf = tl.minimum(f, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(f))) + m
            next_m = tl.where(SEQUENCE_START & (step == 0) & zero, i, tl.maximum(i, logf))
            ig, fg = tl.exp(i - next_m), tl.exp(logf - next_m)
            c = fg * c + ig * zv
            n = tl.maximum(fg * n + ig, 1.0)
            hidden = tl.sigmoid(o) * c / n
            m = next_m
        else:
            c = tl.sigmoid(f) * c + tl.sigmoid(i) * zv
            hidden = tl.sigmoid(o) * (2.0 * tl.sigmoid(2.0 * c) - 1.0)
        if OUTPUT_MODE == 2:
            if STEP_BASE + step == TOTAL_T - 1:
                tl.store(OUTPUT + offset, hidden, valid)
        else:
            outoff = ((row * TOTAL_T + STEP_BASE + step) * H + head) * D + d
            tl.store(OUTPUT + outoff, hidden, valid)
            if OUTPUT_MODE == 0:
                outstride = B * TOTAL_T * H * D
                tl.store(OUTPUT + outstride + outoff, c, valid)
                if SLSTM:
                    tl.store(OUTPUT + 2 * outstride + outoff, n, valid)
                    tl.store(OUTPUT + 3 * outstride + outoff, m, valid)
    tl.store(FINAL + offset, hidden, valid)
    tl.store(FINAL + stride + offset, c, valid)
    if SLSTM:
        tl.store(FINAL + 2 * stride + offset, n, valid)
        tl.store(FINAL + 3 * stride + offset, m, valid)
