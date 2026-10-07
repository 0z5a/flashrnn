"""Map TiRex's sLSTM cell tensors to its BF16 recurrence variant."""

from types import MethodType

import torch

from .triton_persistent import recurrence


def slstm_cell(
    input: torch.Tensor,
    state: torch.Tensor | None,
    recurrent_kernel: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return TiRex-style (batch, heads, time, width) outputs and final state."""
    batch, steps, features = input.shape
    heads, width, recurrent_features = recurrent_kernel.shape
    hidden = heads * width
    if features != 4 * hidden or recurrent_features != 4 * width:
        raise ValueError("TiRex sLSTM gate and recurrent shapes do not match")
    if bias.shape != (4 * hidden,):
        raise ValueError("TiRex sLSTM bias shape does not match")
    if state is None:
        state = input.new_zeros((4, batch, hidden))
    if state.shape != (4, batch, hidden):
        raise ValueError("TiRex sLSTM state shape does not match")

    # Preserve the checkpoint's raw gate slots; the projection names differ.
    wx = input.reshape(batch, steps, 4, heads, width).to(torch.bfloat16)
    r = recurrent_kernel.reshape(heads, width, 4, width).permute(2, 0, 1, 3)
    b = bias.reshape(heads, 4, width).permute(1, 0, 2)
    initial = state.reshape(4, batch, 1, heads, width).to(torch.bfloat16)
    history, final = recurrence(
        wx,
        r.to(torch.bfloat16),
        b.to(torch.float32),
        initial,
        cell="slstm",
        slstm_semantics="tirex",
    )
    output = history[0].permute(0, 2, 1, 3).to(input.dtype)
    next_state = final[:, :, 0].reshape(4, batch, hidden).to(input.dtype)
    return output, next_state


def install_tirex_cells(model: torch.nn.Module) -> None:
    """Use the candidate recurrence in an already loaded TiRex model."""

    def forward(self, input: torch.Tensor, state: torch.Tensor | None):
        return slstm_cell(input, state, self._recurrent_kernel_, self._bias_)

    for block in model.blocks:
        cell = block.slstm_layer.slstm_cell
        cell.forward = MethodType(forward, cell)
