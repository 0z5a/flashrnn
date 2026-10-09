"""Functional Torch backend with explicit recurrent operand arithmetic."""

from typing import Literal

import torch

from .reference import SIZES, Cell, recurrence

Numerics = Literal["mathematical", "fp32_state_bf16_mma"]


def flashrnn_torch(
    wx: torch.Tensor,
    recurrent: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    *,
    cell: Cell = "lstm",
    numerics: Numerics = "mathematical",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full history [S,B,T,H,D] and carry [S,B,1,H,D].

    Inputs must be dense floating tensors of one dtype on one device. Strided
    views are supported. The mathematical policy preserves that dtype. The
    BF16 operand policy requires FP32 inputs and rounds R and the recurrent
    hidden operand through BF16 each step; state and outputs stay FP32.
    Ambient autocast is disabled so it cannot override this contract.
    Torch casts define backward: each step's R cotangent is rounded through
    BF16 before FP32 accumulation, independent of chunk boundaries. This does
    not emulate Tensor Core reduction.

    Inputs are read-only and gradients remain connected to all four inputs.
    Carry can be reused for chunk continuation without detaching. sLSTM uses
    the reference's global zero-normalizer rule over the entire call.
    """
    if cell not in SIZES:
        raise ValueError(f"unsupported cell: {cell}")
    if numerics not in ("mathematical", "fp32_state_bf16_mma"):
        raise ValueError(f"unsupported numerical policy: {numerics}")
    tensors = (wx, recurrent, bias, initial)
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("inputs must have strided tensor layout")
    if tuple(t.ndim for t in tensors) != (5, 4, 3, 5):
        raise ValueError("expected wx/R/b/initial ranks 5/4/3/5")
    if any(size < 1 for size in wx.shape):
        raise ValueError("batch, time, gates, heads and width must be positive")
    if wx.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("inputs must use float16, bfloat16, float32 or float64")
    if any(t.dtype != wx.dtype for t in tensors):
        raise ValueError("all inputs must have the same dtype")
    if any(t.device != wx.device for t in tensors):
        raise ValueError("all inputs must be on the same device")
    mma_dtype = None
    if numerics == "fp32_state_bf16_mma":
        if wx.dtype != torch.float32:
            raise ValueError("fp32_state_bf16_mma requires FP32 inputs")
        mma_dtype = torch.bfloat16
    with torch.autocast(wx.device.type, enabled=False):
        return recurrence(
            wx,
            recurrent,
            bias,
            initial,
            cell=cell,
            mma_dtype=mma_dtype,
            recurrent_mma_dtype=mma_dtype,
        )
