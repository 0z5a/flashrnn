"""Explicit register operands for the single-CTA MMA recurrence."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def sigmoid(x):
    return 1.0 / (1.0 + gl.exp(-x))


@gluon.jit
def _sequence(
    WX,
    R,
    BIAS,
    INITIAL,
    OUTPUT,
    B: gl.constexpr,
    T: gl.constexpr,
    H: gl.constexpr,
    D: gl.constexpr,
    SLSTM: gl.constexpr,
    ZERO_NORMALIZER,
    BM: gl.constexpr,
    BD: gl.constexpr,
    WEIGHT_A: gl.constexpr,
    TIREX: gl.constexpr,
):
    gl.static_assert(not WEIGHT_A)
    gl.static_assert(not TIREX)
    state_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [1, 8], [1, 0])
    weight_layout: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [1, 8], [0, 1])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout([2, 0], [1, 8], [16, 8])
    lhs_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 2)
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 2)
    rows = gl.program_id(0) * BM + gl.arange(
        0, BM, layout=gl.SliceLayout(1, state_layout)
    )
    head = gl.program_id(1)
    d = gl.arange(0, BD, layout=gl.SliceLayout(0, state_layout))
    k = gl.arange(0, BD, layout=gl.SliceLayout(1, weight_layout))
    out = gl.arange(0, BD, layout=gl.SliceLayout(0, weight_layout))
    valid = (rows[:, None] < B) & (d[None, :] < D)
    state_offset = (rows[:, None] * H + head) * D + d[None, :]
    stride = B * H * D
    hidden = gl.load(INITIAL + state_offset, valid, 0).to(gl.float32)
    c = gl.load(INITIAL + stride + state_offset, valid, 0).to(gl.float32)
    n = gl.full((BM, BD), 1.0, gl.float32, state_layout)
    m = gl.full((BM, BD), 0.0, gl.float32, state_layout)
    if SLSTM:
        n = gl.load(INITIAL + 2 * stride + state_offset, valid, 0).to(gl.float32)
        m = gl.load(INITIAL + 3 * stride + state_offset, valid, 0).to(gl.float32)
        zero_normalizer = gl.load(ZERO_NORMALIZER)
    weight_offset = (head * D + out[None, :]) * D + k[:, None]
    weight_mask = (k[:, None] < D) & (out[None, :] < D)
    ri = gl.convert_layout(gl.load(R + weight_offset, weight_mask, 0), rhs_layout)
    rf = gl.convert_layout(
        gl.load(R + H * D * D + weight_offset, weight_mask, 0), rhs_layout
    )
    rz = gl.convert_layout(
        gl.load(R + 2 * H * D * D + weight_offset, weight_mask, 0), rhs_layout
    )
    ro = gl.convert_layout(
        gl.load(R + 3 * H * D * D + weight_offset, weight_mask, 0), rhs_layout
    )
    bi = gl.load(BIAS + head * D + d, d < D, 0).to(gl.float32)
    bf = gl.load(BIAS + H * D + head * D + d, d < D, 0).to(gl.float32)
    bz = gl.load(BIAS + 2 * H * D + head * D + d, d < D, 0).to(gl.float32)
    bo = gl.load(BIAS + 3 * H * D + head * D + d, d < D, 0).to(gl.float32)
    zero = gl.full((BM, BD), 0.0, gl.float32, mma_layout)
    for step in range(T):
        h_mma = gl.convert_layout(hidden.to(WX.dtype.element_ty), lhs_layout)
        ai = gl.convert_layout(mma_v2(h_mma, ri, zero), state_layout)
        af = gl.convert_layout(mma_v2(h_mma, rf, zero), state_layout)
        az = gl.convert_layout(mma_v2(h_mma, rz, zero), state_layout)
        ao = gl.convert_layout(mma_v2(h_mma, ro, zero), state_layout)
        offset = (((rows[:, None] * T + step) * 4) * H + head) * D + d[None, :]
        i = ai + gl.load(WX + offset, valid, 0).to(gl.float32) + bi[None, :]
        f = af + gl.load(WX + offset + H * D, valid, 0).to(gl.float32) + bf[None, :]
        z = az + gl.load(WX + offset + 2 * H * D, valid, 0).to(gl.float32) + bz[None, :]
        o = ao + gl.load(WX + offset + 3 * H * D, valid, 0).to(gl.float32) + bo[None, :]
        z_value = 2.0 * sigmoid(2.0 * z) - 1.0
        if SLSTM:
            logf = gl.minimum(f, 0.0) - gl.log(1.0 + gl.exp(-gl.abs(f))) + m
            next_m = gl.where((step == 0) & zero_normalizer, i, gl.maximum(i, logf))
            igate = gl.exp(i - next_m)
            fgate = gl.exp(logf - next_m)
            c = fgate * c + igate * z_value
            n = gl.maximum(fgate * n + igate, 1.0)
            hidden = sigmoid(o) * c / n
            m = next_m
        else:
            c = sigmoid(f) * c + sigmoid(i) * z_value
            hidden = sigmoid(o) * (2.0 * sigmoid(2.0 * c) - 1.0)
        output_offset = ((rows[:, None] * T + step) * H + head) * D + d[None, :]
        output_stride = B * T * H * D
        gl.store(OUTPUT + output_offset, hidden, valid)
        gl.store(OUTPUT + output_stride + output_offset, c, valid)
        if SLSTM:
            gl.store(OUTPUT + 2 * output_stride + output_offset, n, valid)
            gl.store(OUTPUT + 3 * output_stride + output_offset, m, valid)
