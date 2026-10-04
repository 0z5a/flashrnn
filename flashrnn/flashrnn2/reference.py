"""Differentiable reference in the public FlashRNN tensor layout.

Float64 inputs define the mathematical reference. FP32 inputs with mma_dtype
set define FP32 local states with a rounded recurrent hidden operand. This is
not an emulator of the upstream CUDA backend's default BF16 pointwise/state
arithmetic. sLSTM follows vanilla's global zero-normalizer initialization.
"""

from typing import Literal

import torch

Cell = Literal["lstm", "slstm", "gru", "elman"]
SIZES: dict[str, tuple[int, int, int]] = {
    "lstm": (4, 4, 2),
    "slstm": (4, 4, 4),
    "gru": (3, 4, 1),
    "elman": (1, 1, 1),
}


def recurrence(
    wx: torch.Tensor,
    recurrent: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    cell: Cell = "lstm",
    mma_dtype: torch.dtype | None = None,
    slstm_init: Literal["global", "elementwise"] = "global",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return state history [S,B,T,N,D] and last state [S,B,1,N,D]."""
    batch, steps, gates, heads, width = wx.shape
    if slstm_init not in ("global", "elementwise"):
        raise ValueError("sLSTM initialization must be global or elementwise")
    expected_gates, bias_gates, state_count = SIZES[cell]
    if steps < 1 or gates != expected_gates:
        raise ValueError("expected nonempty sequence and matching input gate count")
    if recurrent.shape != (gates, heads, width, width):
        raise ValueError("recurrent weights must have layout [G,N,D,P]")
    if bias.shape != (bias_gates, heads, width):
        raise ValueError("bias gate count does not match cell")
    if initial.shape != (state_count, batch, 1, heads, width):
        raise ValueError("initial state must have layout [S,B,1,N,D]")
    state = initial[:, :, 0]
    history = []
    for x in wx.unbind(1):
        hidden = (
            state[0] if mma_dtype is None else state[0].to(mma_dtype).to(state.dtype)
        )
        ry = torch.einsum("bnp,gnop->bgno", hidden, recurrent)
        if cell == "gru":
            candidate = ry[:, 0] + bias[0]
            reset = torch.sigmoid(x[:, 0] + ry[:, 1] + bias[1])
            update = torch.sigmoid(x[:, 1] + ry[:, 2] + bias[2])
            value = torch.tanh(x[:, 2] + bias[3] + reset * candidate)
            state = (update * state[0] + (1 - update) * value).unsqueeze(0)
        elif cell == "elman":
            state = torch.tanh(x[:, 0] + ry[:, 0] + bias[0]).unsqueeze(0)
        else:
            i, f, z, o = (x + ry + bias).unbind(1)
            if cell == "lstm":
                c = torch.sigmoid(f) * state[1] + torch.sigmoid(i) * torch.tanh(z)
                state = torch.stack((torch.sigmoid(o) * torch.tanh(c), c))
            else:
                _, old_c, old_n, old_m = state.unbind(0)
                log_f = torch.nn.functional.logsigmoid(f) + old_m
                zero = torch.all(old_n == 0) if slstm_init == "global" else old_n == 0
                m = torch.where(zero, i, torch.maximum(i, log_f))
                input_gate = torch.exp(i - m)
                forget_gate = torch.exp(log_f - m)
                c = forget_gate * old_c + input_gate * torch.tanh(z)
                n = torch.maximum(
                    forget_gate * old_n + input_gate, torch.ones_like(old_n)
                )
                state = torch.stack((torch.sigmoid(o) * c / n, c, n, m))
        history.append(state)
    return torch.stack(history, dim=2), state.unsqueeze(2)
