"""Single-CTA GRU and Elman inference in the public FlashRNN layout."""

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
    GRU: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    d = tl.arange(0, BD)
    valid = (rows[:, None] < B) & (d[None, :] < D)
    state_offset = (rows[:, None] * H + head) * D + d[None, :]
    hidden = tl.load(INITIAL + state_offset, valid, 0).to(tl.float32)
    weight_offset = (head * D + d[None, :]) * D + d[:, None]
    weight_mask = (d[:, None] < D) & (d[None, :] < D)
    r0 = tl.load(R + weight_offset, weight_mask, 0)
    b0 = tl.load(BIAS + head * D + d, d < D, 0).to(tl.float32)
    if GRU:
        r1 = tl.load(R + H * D * D + weight_offset, weight_mask, 0)
        r2 = tl.load(R + 2 * H * D * D + weight_offset, weight_mask, 0)
        b1 = tl.load(BIAS + H * D + head * D + d, d < D, 0).to(tl.float32)
        b2 = tl.load(BIAS + 2 * H * D + head * D + d, d < D, 0).to(tl.float32)
        b3 = tl.load(BIAS + 3 * H * D + head * D + d, d < D, 0).to(tl.float32)
    dtype = WX.dtype.element_ty
    gates = 3 if GRU else 1
    for step in range(T):
        h_mma = hidden.to(dtype)
        x_offset = (((rows[:, None] * T + step) * gates) * H + head) * D + d[None, :]
        x0 = tl.load(WX + x_offset, valid, 0).to(tl.float32)
        if GRU:
            candidate = tl.dot(h_mma, r0) + b0[None, :]
            reset_arg = x0 + tl.dot(h_mma, r1) + b1[None, :]
            update_arg = (
                tl.load(WX + x_offset + H * D, valid, 0).to(tl.float32)
                + tl.dot(h_mma, r2)
                + b2[None, :]
            )
            value_arg = (
                tl.load(WX + x_offset + 2 * H * D, valid, 0).to(tl.float32)
                + b3[None, :]
                + tl.sigmoid(reset_arg) * candidate
            )
            update = tl.sigmoid(update_arg)
            value = 2.0 * tl.sigmoid(2.0 * value_arg) - 1.0
            hidden = update * hidden + (1.0 - update) * value
        else:
            value_arg = x0 + tl.dot(h_mma, r0) + b0[None, :]
            hidden = 2.0 * tl.sigmoid(2.0 * value_arg) - 1.0
        output_offset = ((rows[:, None] * T + step) * H + head) * D + d[None, :]
        tl.store(OUTPUT + output_offset, hidden, valid)


def recurrence(
    wx: torch.Tensor,
    recurrent: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    cell: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cell not in ("gru", "elman"):
        raise ValueError("basic persistent inference supports GRU and Elman")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("basic persistent inference requires CUDA FP16/BF16")
    if torch.is_grad_enabled() and any(
        value.requires_grad for value in (wx, recurrent, bias, initial)
    ):
        raise ValueError("basic persistent inference has no backward implementation")
    batch, steps, gates, heads, width = wx.shape
    expected_gates, bias_gates = (3, 4) if cell == "gru" else (1, 1)
    if gates != expected_gates or steps < 1 or not 16 <= width <= 128:
        raise ValueError("expected nonempty GRU/Elman sequence with 16 <= D <= 128")
    if recurrent.shape != (gates, heads, width, width):
        raise ValueError("expected public recurrent-weight layout")
    if bias.shape != (bias_gates, heads, width):
        raise ValueError("expected public bias layout")
    if initial.shape != (1, batch, 1, heads, width):
        raise ValueError("expected one initial state")
    if any(
        value.dtype != wx.dtype or value.device != wx.device
        for value in (recurrent, bias, initial)
    ):
        raise ValueError("all inputs must share dtype and device")
    wx, recurrent, bias, initial = (
        value.contiguous() for value in (wx, recurrent, bias, initial)
    )
    output = torch.empty(
        (1, batch, steps, heads, width), device=wx.device, dtype=wx.dtype
    )
    _sequence[(triton.cdiv(batch, 16), heads)](
        wx,
        recurrent,
        bias,
        initial,
        output,
        batch,
        steps,
        heads,
        width,
        cell == "gru",
        16,
        triton.next_power_of_2(width),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, output[:, :, -1:]
