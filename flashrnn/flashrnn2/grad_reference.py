"""Gradient oracle for the FlashRNN2 LSTM/sLSTM backward (whole logical calls).

The forward values equal ``reference.recurrence`` (same operations, same order). Autograd of this function defines
the reference gradient under the FlashRNN v1 convention, which FlashRNN2 keeps:

* the sLSTM stabilizer m carries no gradient (neither the new m_t nor the m_{t-1} inside log f), so the gradient
  with respect to the initial m is zero;
* the normalizer clamp n_t = max(f n_{t-1} + i, 1) passes its gradient straight through.

h_t = sigmoid(o) c_t / n_t does not depend on the internal stabilizers when the clamp is inactive, and in exact
arithmetic the clamp is inactive whenever the logical call starts from a zero normalizer or from n_0 >= 1. In those
cases the convention gives the true gradient for every input except the initial m. Evaluate in float64.
"""

import torch

from .reference import SIZES


class _ClampStraightThrough(torch.autograd.Function):
    """max(x, 1) in the forward, identity in the backward."""

    @staticmethod
    def forward(ctx, x):
        return torch.maximum(x, torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad):
        return grad


def recurrence(wx, recurrent, bias, initial, cell="lstm"):
    """Return state history [S,B,T,N,D] and last state [S,B,1,N,D], differentiable under the convention above."""
    if cell not in ("lstm", "slstm"):
        raise ValueError("the gradient oracle covers LSTM and sLSTM")
    batch, steps, gates, heads, width = wx.shape
    if gates != 4 or initial.shape != (SIZES[cell][2], batch, 1, heads, width):
        raise ValueError("expected public input and initial-state layouts")
    state = initial[:, :, 0]
    history = []
    for x in wx.unbind(1):
        ry = torch.einsum("bnp,gnop->bgno", state[0], recurrent)
        i, f, z, o = (x + ry + bias).unbind(1)
        if cell == "lstm":
            c = torch.sigmoid(f) * state[1] + torch.sigmoid(i) * torch.tanh(z)
            state = torch.stack((torch.sigmoid(o) * torch.tanh(c), c))
        else:
            _, old_c, old_n, old_m = state.unbind(0)
            log_f = torch.nn.functional.logsigmoid(f) + old_m.detach()
            zero = torch.all(old_n == 0)
            m = torch.where(zero, i, torch.maximum(i, log_f)).detach()
            input_gate = torch.exp(i - m)
            forget_gate = torch.exp(log_f - m)
            c = forget_gate * old_c + input_gate * torch.tanh(z)
            n = _ClampStraightThrough.apply(forget_gate * old_n + input_gate)
            state = torch.stack((torch.sigmoid(o) * c / n, c, n, m))
        history.append(state)
    return torch.stack(history, dim=2), state.unsqueeze(2)
