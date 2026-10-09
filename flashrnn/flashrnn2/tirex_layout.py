"""Convert TiRex's input-major recurrent matrix for FlashRNN2."""

import torch


def recurrent_layout(kernel: torch.Tensor) -> torch.Tensor:
    heads, width, _ = kernel.shape
    return kernel.reshape(heads, width, 4, width).permute(2, 0, 3, 1).contiguous()
