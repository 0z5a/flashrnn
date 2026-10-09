"""Map the author's FP32 xLSTM sLSTM cell to FlashRNN2 recurrence."""

from types import MethodType

import torch

from .tirex_layout import recurrent_layout
from .triton_persistent import recurrence


def slstm_cell(
    input: torch.Tensor,
    state: torch.Tensor | None,
    recurrent_kernel: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, steps, features = input.shape
    heads, width, recurrent_features = recurrent_kernel.shape
    hidden = heads * width
    if (
        input.dtype != torch.float32
        or features != 4 * hidden
        or recurrent_features != 4 * width
        or bias.shape != (4 * hidden,)
    ):
        raise ValueError("expected FP32 xLSTM gate, recurrent and bias layouts")
    if state is None:
        state = input.new_zeros((4, batch, hidden))
    if state.shape != (4, batch, hidden):
        raise ValueError("expected four xLSTM states")

    history, final = recurrence(
        input.reshape(batch, steps, 4, heads, width),
        recurrent_layout(recurrent_kernel),
        bias.reshape(heads, 4, width).permute(1, 0, 2),
        state.reshape(4, batch, 1, heads, width),
        cell="slstm",
        slstm_semantics="xlstm",
    )
    return history[0].permute(0, 2, 1, 3), final[:, :, 0].reshape(4, batch, hidden)


def install_xlstm_cell(model: torch.nn.Module) -> None:
    """Replace the pinned model's single sLSTM cell without changing weights."""
    cell = model.xlstm_block_stack.blocks[1].xlstm.slstm_cell

    def forward(self, input: torch.Tensor, state=None, lengths=None):
        if lengths is not None:
            raise ValueError("xLSTM candidate does not support sequence lengths")
        return slstm_cell(input, state, self._recurrent_kernel_, self._bias_)

    cell.forward = MethodType(forward, cell)
