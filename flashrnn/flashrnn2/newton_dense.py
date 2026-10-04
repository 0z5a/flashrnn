"""Dense Newton/affine-scan experiment for known complete input sequences.

This Torch prototype preserves recurrent mixing and uses unrolled autograd.
It is neither a fused CUDA implementation nor an autoregressive decoder.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .reference import SIZES, Cell


@dataclass
class NewtonResult:
    history: torch.Tensor
    final: torch.Tensor
    residuals: list[float]
    converged: bool
    jacobian_bytes: int


def affine_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive composition of maps x -> A x + b, parallel across time."""
    stride = 1
    while stride < b.shape[1]:
        right = a[:, stride:]
        next_b = b[:, stride:] + (right @ b[:, :-stride, :, None]).squeeze(-1)
        next_a = right @ a[:, :-stride]
        a = torch.cat((a[:, :stride], next_a), 1)
        b = torch.cat((b[:, :stride], next_b), 1)
        stride *= 2
    return b


def recurrence(
    wx: torch.Tensor,
    recurrent: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    cell: Cell = "lstm",
    *,
    max_iterations: int = 16,
    atol: float = 1e-10,
    linear_solver: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = affine_scan,
) -> NewtonResult:
    """Solve the original mathematical cell with a bounded dense Jacobian scan."""
    batch, steps, gates, heads, width = wx.shape
    expected_gates, bias_gates, states = SIZES[cell]
    if gates != expected_gates or steps < 1 or max_iterations < 1 or atol <= 0:
        raise ValueError("invalid gates, sequence length or Newton budget")
    if wx.dtype not in (torch.float32, torch.float64):
        raise ValueError("dense Newton qualification uses FP32 or FP64")
    if recurrent.shape != (gates, heads, width, width) or bias.shape != (
        bias_gates,
        heads,
        width,
    ):
        raise ValueError("invalid recurrent or bias layout")
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("invalid initial-state layout")
    units, size = batch * heads, states * width
    x = wx.permute(0, 3, 1, 2, 4).reshape(units, steps, gates, width)
    r = (
        recurrent.permute(1, 0, 2, 3)
        .expand(batch, -1, -1, -1, -1)
        .reshape(units, gates, width, width)
    )
    b = (
        bias.permute(1, 0, 2)
        .expand(batch, -1, -1, -1)
        .reshape(units, bias_gates, width)
    )
    start = initial[:, :, 0].permute(1, 2, 0, 3).reshape(units, states, width)

    def step(
        s: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        offset: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        ry = torch.einsum("gdp,p->gd", weight, s[0])
        if cell == "elman":
            return torch.tanh(u[0] + ry[0] + offset[0]).unsqueeze(0)
        if cell == "gru":
            reset = torch.sigmoid(u[0] + ry[1] + offset[1])
            update = torch.sigmoid(u[1] + ry[2] + offset[2])
            value = torch.tanh(u[2] + offset[3] + reset * (ry[0] + offset[0]))
            return (update * s[0] + (1 - update) * value).unsqueeze(0)
        i, f, z, o = (u + ry + offset).unbind(0)
        if cell == "lstm":
            c = torch.sigmoid(f) * s[1] + torch.sigmoid(i) * torch.tanh(z)
            return torch.stack((torch.sigmoid(o) * torch.tanh(c), c))
        log_f = torch.nn.functional.logsigmoid(f) + s[3]
        m = torch.where(zero, i, torch.maximum(i, log_f))
        ig, fg = torch.exp(i - m), torch.exp(log_f - m)
        c = fg * s[1] + ig * torch.tanh(z)
        n = torch.maximum(fg * s[2] + ig, torch.ones_like(s[2]))
        return torch.stack((torch.sigmoid(o) * c / n, c, n, m))

    axes = (0, 0, None, None, 0)
    across_units = (0, 0, 0, 0, None)
    evaluate = torch.vmap(torch.vmap(step, in_dims=axes), in_dims=across_units)
    jacobian = torch.vmap(
        torch.vmap(torch.func.jacrev(step), in_dims=axes), in_dims=across_units
    )

    def predecessors(guess: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        previous = torch.cat((start[:, None], guess[:, :-1]), 1)
        zero = (
            (previous[:, :, 2] == 0).all(dim=(0, 2))
            if cell == "slstm"
            else torch.zeros(steps, device=wx.device, dtype=torch.bool)
        )
        return previous, zero

    guess = torch.zeros(units, steps, states, width, device=wx.device, dtype=wx.dtype)
    residuals = []
    for _ in range(max_iterations):
        previous, zero = predecessors(guess)
        values = evaluate(previous, x, r, b, zero).reshape(units, steps, size)
        a = jacobian(previous, x, r, b, zero).reshape(units, steps, size, size)
        offset = values - (a @ previous.reshape(units, steps, size, 1)).squeeze(-1)
        first = offset[:, :1] + (a[:, :1] @ start.reshape(units, 1, size, 1)).squeeze(
            -1
        )
        offset = torch.cat((first, offset[:, 1:]), 1)
        guess = linear_solver(a, offset).reshape(units, steps, states, width)
        previous, zero = predecessors(guess)
        residual = (guess - evaluate(previous, x, r, b, zero)).abs().max().item()
        residuals.append(residual)
        if residual <= atol:
            break
    history = guess.reshape(batch, heads, steps, states, width).permute(3, 0, 2, 1, 4)
    return NewtonResult(
        history,
        history[:, :, -1:],
        residuals,
        residuals[-1] <= atol,
        a.numel() * a.element_size(),
    )
