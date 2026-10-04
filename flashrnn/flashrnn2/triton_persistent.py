"""Single-CTA recurrence with weights loaded before the timestep loop."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sequence(
    WX,
    R,
    BIAS,
    INITIAL,
    OUTPUT,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SLSTM: tl.constexpr,
    ZERO_NORMALIZER,
    BM: tl.constexpr,
    BD: tl.constexpr,
    WEIGHT_A: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    d = tl.arange(0, BD)
    valid = (rows[:, None] < B) & (d[None, :] < D)
    state_offset = (rows[:, None] * H + head) * D + d[None, :]
    stride = B * H * D
    hidden = tl.load(INITIAL + state_offset, valid, 0).to(tl.float32)
    c = tl.load(INITIAL + stride + state_offset, valid, 0).to(tl.float32)
    n = tl.full((BM, BD), 1.0, tl.float32)
    m = tl.zeros((BM, BD), tl.float32)
    if SLSTM:
        n = tl.load(INITIAL + 2 * stride + state_offset, valid, 0).to(tl.float32)
        m = tl.load(INITIAL + 3 * stride + state_offset, valid, 0).to(tl.float32)
        zero_normalizer = tl.load(ZERO_NORMALIZER)
    weight_offset = (head * D + d[None, :]) * D + d[:, None]
    weight_mask = (d[:, None] < D) & (d[None, :] < D)
    ri = tl.load(R + weight_offset, weight_mask, 0)
    rf = tl.load(R + H * D * D + weight_offset, weight_mask, 0)
    rz = tl.load(R + 2 * H * D * D + weight_offset, weight_mask, 0)
    ro = tl.load(R + 3 * H * D * D + weight_offset, weight_mask, 0)
    bi = tl.load(BIAS + head * D + d, d < D, 0).to(tl.float32)
    bf = tl.load(BIAS + H * D + head * D + d, d < D, 0).to(tl.float32)
    bz = tl.load(BIAS + 2 * H * D + head * D + d, d < D, 0).to(tl.float32)
    bo = tl.load(BIAS + 3 * H * D + head * D + d, d < D, 0).to(tl.float32)
    dtype = WX.dtype.element_ty
    for step in range(T):
        h_mma = hidden.to(dtype)
        x_offset = (((rows[:, None] * T + step) * 4) * H + head) * D + d[None, :]
        if WEIGHT_A:
            ai = tl.trans(tl.dot(tl.trans(ri), tl.trans(h_mma)))
            af = tl.trans(tl.dot(tl.trans(rf), tl.trans(h_mma)))
            az = tl.trans(tl.dot(tl.trans(rz), tl.trans(h_mma)))
            ao = tl.trans(tl.dot(tl.trans(ro), tl.trans(h_mma)))
        else:
            ai = tl.dot(h_mma, ri)
            af = tl.dot(h_mma, rf)
            az = tl.dot(h_mma, rz)
            ao = tl.dot(h_mma, ro)
        i = ai + tl.load(WX + x_offset, valid, 0).to(tl.float32) + bi[None, :]
        f = af + tl.load(WX + x_offset + H * D, valid, 0).to(tl.float32) + bf[None, :]
        z = (
            az
            + tl.load(WX + x_offset + 2 * H * D, valid, 0).to(tl.float32)
            + bz[None, :]
        )
        o = (
            ao
            + tl.load(WX + x_offset + 3 * H * D, valid, 0).to(tl.float32)
            + bo[None, :]
        )
        z_value = 2.0 * tl.sigmoid(2.0 * z) - 1.0
        if SLSTM:
            logf = tl.minimum(f, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(f))) + m
            next_m = tl.where((step == 0) & zero_normalizer, i, tl.maximum(i, logf))
            igate = tl.exp(i - next_m)
            fgate = tl.exp(logf - next_m)
            c = fgate * c + igate * z_value
            n = tl.maximum(fgate * n + igate, 1.0)
            hidden = tl.sigmoid(o) * c / n
            m = next_m
        else:
            c = tl.sigmoid(f) * c + tl.sigmoid(i) * z_value
            hidden = tl.sigmoid(o) * (2.0 * tl.sigmoid(2.0 * c) - 1.0)
        output_offset = ((rows[:, None] * T + step) * H + head) * D + d[None, :]
        output_stride = B * T * H * D
        tl.store(OUTPUT + output_offset, hidden, valid)
        tl.store(OUTPUT + output_stride + output_offset, c, valid)
        if SLSTM:
            tl.store(OUTPUT + 2 * output_stride + output_offset, n, valid)
            tl.store(OUTPUT + 3 * output_stride + output_offset, m, valid)


def recurrence(
    wx: torch.Tensor,
    recurrent: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    cell: str = "lstm",
    diagnostics: dict[str, object] | None = None,
    snapshot_dtype: torch.dtype | None = None,
    weight_operand: str = "b",
    register_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one program per batch tile/head; optional FP32 snapshots are diagnostic."""
    if register_layout and weight_operand != "b":
        raise ValueError("explicit register layout uses weights as operand b")
    if weight_operand not in ("a", "b"):
        raise ValueError("weight operand must be a or b")
    if cell not in ("lstm", "slstm"):
        raise ValueError("persistent prototype supports LSTM and sLSTM")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("persistent prototype requires CUDA FP16/BF16 inputs")
    if torch.is_grad_enabled() and any(
        x.requires_grad for x in (wx, recurrent, bias, initial)
    ):
        raise ValueError("persistent prototype has no backward implementation yet")
    batch, steps, gates, heads, width = wx.shape
    states = 4 if cell == "slstm" else 2
    if gates != 4 or steps < 1 or not 16 <= width <= 128:
        raise ValueError("single CTA supports 16 <= D <= 128 and nonempty sequences")
    if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
        raise ValueError("expected public weight and bias layouts")
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("expected public initial state layout")
    if any(
        x.dtype != wx.dtype or x.device != wx.device for x in (recurrent, bias, initial)
    ):
        raise ValueError("all inputs must share dtype and device")
    wx, recurrent, bias, initial = (
        x.contiguous() for x in (wx, recurrent, bias, initial)
    )
    zero_normalizer = torch.all(initial[2] == 0) if cell == "slstm" else initial
    output = torch.empty(
        (states, batch, steps, heads, width),
        device=wx.device,
        dtype=wx.dtype if snapshot_dtype is None else snapshot_dtype,
    )
    sequence = _sequence
    if register_layout:
        from .gluon_persistent import _sequence as sequence
    kernel = sequence[(triton.cdiv(batch, 16), heads)](
        wx,
        recurrent,
        bias,
        initial,
        output,
        batch,
        steps,
        heads,
        width,
        cell == "slstm",
        zero_normalizer,
        16,
        triton.next_power_of_2(width),
        weight_operand == "a",
        num_warps=8 if register_layout else 4,
        enable_fp_fusion=False,
    )
    if diagnostics is not None:
        diagnostics.update(
            registers=kernel.n_regs,
            spills=kernel.n_spills,
            shared_bytes=kernel.metadata.shared,
            ptx=kernel.asm["ptx"],
            weight_operand=weight_operand,
            explicit_register_layout=register_layout,
            persistent=True,
            synchronization="single_cta",
        )
    return output, output[:, :, -1:]
