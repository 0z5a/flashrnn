"""Output-coordinate tiling with a kernel boundary between timesteps.

Each program owns all four gates for its output coordinates. This is a
nonpersistent synchronization reference: weights are read on every timestep.
Local states are FP32; recurrent h is rounded to the input dtype before MMA.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _step(
    WX,
    R,
    BIAS,
    INITIAL,
    OLD,
    NEW,
    OUTPUT,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    STEP,
    FIRST: tl.constexpr,
    SLSTM: tl.constexpr,
    ZERO_NORMALIZER,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    dtype = WX.dtype.element_ty
    ai = tl.zeros((BM, BN), tl.float32)
    af = tl.zeros((BM, BN), tl.float32)
    az = tl.zeros((BM, BN), tl.float32)
    ao = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(D, BK)):
        k = start * BK + rk
        state_offset = (rows[:, None] * H + head) * D + k[None, :]
        if FIRST:
            hidden = tl.load(
                INITIAL + state_offset, (rows[:, None] < B) & (k[None, :] < D), 0
            )
        else:
            hidden = tl.load(
                OLD + state_offset, (rows[:, None] < B) & (k[None, :] < D), 0
            ).to(dtype)
        weight_offset = (head * D + cols[None, :]) * D + k[:, None]
        valid_weight = (k[:, None] < D) & (cols[None, :] < D)
        ri = tl.load(R + weight_offset, valid_weight, 0)
        rf = tl.load(R + H * D * D + weight_offset, valid_weight, 0)
        rz = tl.load(R + 2 * H * D * D + weight_offset, valid_weight, 0)
        ro = tl.load(R + 3 * H * D * D + weight_offset, valid_weight, 0)
        ai += tl.dot(hidden, ri)
        af += tl.dot(hidden, rf)
        az += tl.dot(hidden, rz)
        ao += tl.dot(hidden, ro)
    valid = (rows[:, None] < B) & (cols[None, :] < D)
    xbase = (((rows[:, None] * T + STEP) * 4) * H + head) * D + cols[None, :]
    bbase = head * D + cols[None, :]
    i = (
        ai
        + tl.load(WX + xbase, valid, 0).to(tl.float32)
        + tl.load(BIAS + bbase, cols[None, :] < D, 0).to(tl.float32)
    )
    f = (
        af
        + tl.load(WX + xbase + H * D, valid, 0).to(tl.float32)
        + tl.load(BIAS + bbase + H * D, cols[None, :] < D, 0).to(tl.float32)
    )
    z = (
        az
        + tl.load(WX + xbase + 2 * H * D, valid, 0).to(tl.float32)
        + tl.load(BIAS + bbase + 2 * H * D, cols[None, :] < D, 0).to(tl.float32)
    )
    o = (
        ao
        + tl.load(WX + xbase + 3 * H * D, valid, 0).to(tl.float32)
        + tl.load(BIAS + bbase + 3 * H * D, cols[None, :] < D, 0).to(tl.float32)
    )
    offset = (rows[:, None] * H + head) * D + cols[None, :]
    stride = B * H * D
    if FIRST:
        old_c = tl.load(INITIAL + stride + offset, valid, 0).to(tl.float32)
    else:
        old_c = tl.load(OLD + stride + offset, valid, 0)
    if SLSTM:
        if FIRST:
            old_n = tl.load(INITIAL + 2 * stride + offset, valid, 0).to(tl.float32)
            old_m = tl.load(INITIAL + 3 * stride + offset, valid, 0).to(tl.float32)
        else:
            old_n = tl.load(OLD + 2 * stride + offset, valid, 0)
            old_m = tl.load(OLD + 3 * stride + offset, valid, 0)
        logf = tl.minimum(f, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(f))) + old_m
        if FIRST:
            m = tl.where(tl.load(ZERO_NORMALIZER), i, tl.maximum(i, logf))
        else:
            m = tl.maximum(i, logf)
        igate = tl.exp(i - m)
        fgate = tl.exp(logf - m)
        c = fgate * old_c + igate * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        n = tl.maximum(fgate * old_n + igate, 1.0)
        hidden_new = tl.sigmoid(o) * c / n
        tl.store(NEW + 2 * stride + offset, n, valid)
        tl.store(NEW + 3 * stride + offset, m, valid)
    else:
        c = tl.sigmoid(f) * old_c + tl.sigmoid(i) * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        hidden_new = tl.sigmoid(o) * (2.0 * tl.sigmoid(2.0 * c) - 1.0)
    tl.store(NEW + offset, hidden_new, valid)
    tl.store(NEW + stride + offset, c, valid)
    output_offset = ((rows[:, None] * T + STEP) * H + head) * D + cols[None, :]
    output_stride = B * T * H * D
    tl.store(OUTPUT + output_offset, hidden_new, valid)
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
    output_tile: int = 32,
    diagnostics: dict[str, object] | None = None,
    snapshot_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cell not in ("lstm", "slstm"):
        raise ValueError("Triton step supports LSTM and sLSTM")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("Triton step requires CUDA FP16/BF16 tensors")
    if torch.is_grad_enabled() and any(
        x.requires_grad for x in (wx, recurrent, bias, initial)
    ):
        raise ValueError("Triton step forward has no backward implementation yet")
    batch, steps, gates, heads, width = wx.shape
    states = 4 if cell == "slstm" else 2
    if gates != 4 or steps < 1 or output_tile not in (16, 32, 64):
        raise ValueError("invalid timestep, gate count, or output tile")
    if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
        raise ValueError("expected public recurrent-weight and bias layouts")
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("expected public initial-state layout")
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
        dtype=wx.dtype if snapshot_dtype is None else snapshot_dtype,
        device=wx.device,
    )
    scratch = torch.empty(
        (2, states, batch, heads, width), dtype=torch.float32, device=wx.device
    )
    grid = (triton.cdiv(batch, 16), heads, triton.cdiv(width, output_tile))
    for step in range(steps):
        kernel = _step[grid](
            wx,
            recurrent,
            bias,
            initial,
            scratch[step % 2],
            scratch[(step + 1) % 2],
            output,
            batch,
            steps,
            heads,
            width,
            step,
            step == 0,
            cell == "slstm",
            zero_normalizer,
            16,
            output_tile,
            32,
            num_warps=4,
            enable_fp_fusion=False,
        )
        if diagnostics is not None and step < 2:
            diagnostics["initial" if step == 0 else "recurrent"] = {
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
                "shared_bytes": kernel.metadata.shared,
                "ptx": kernel.asm["ptx"],
                "persistent": False,
                "synchronization": "kernel_boundary",
            }
    return output, output[:, :, -1:]
