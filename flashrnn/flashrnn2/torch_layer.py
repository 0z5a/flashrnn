"""Exact layer mapping to torch.nn recurrent modules (cuDNN on supported CUDA).

The observable contract is hidden history plus final states. cuDNN does not
expose LSTM cell history, so this is not a full-state K1 recurrence baseline.
Parameter packing happens once at construction and belongs to setup cost.
"""

import torch
from torch import nn

from .reference import SIZES


class TorchLayer(nn.Module):
    def __init__(
        self,
        input_weight: torch.Tensor,
        recurrent: torch.Tensor,
        bias: torch.Tensor,
        cell: str,
    ) -> None:
        super().__init__()
        if cell not in ("lstm", "gru", "elman"):
            raise ValueError("torch.nn baseline supports LSTM, GRU and Elman")
        gates, heads, width, input_size = input_weight.shape
        expected_gates, bias_gates, _ = SIZES[cell]
        if gates != expected_gates or recurrent.shape != (gates, heads, width, width):
            raise ValueError("invalid gate or recurrent-weight shape")
        if bias.shape != (bias_gates, heads, width):
            raise ValueError("invalid bias shape")
        self.cell = cell
        self.layers = nn.ModuleList()
        for head in range(heads):
            module_type = {"lstm": nn.LSTM, "gru": nn.GRU, "elman": nn.RNN}[cell]
            layer = module_type(
                input_size,
                width,
                batch_first=True,
                device=input_weight.device,
                dtype=input_weight.dtype,
            )
            recurrent_order = (1, 2, 0) if cell == "gru" else tuple(range(gates))
            with torch.no_grad():
                layer.weight_ih_l0.copy_(
                    input_weight[:, head].reshape(gates * width, input_size)
                )
                layer.weight_hh_l0.copy_(
                    recurrent[list(recurrent_order), head].reshape(gates * width, width)
                )
                layer.bias_hh_l0.zero_()
                if cell == "gru":
                    layer.bias_ih_l0.copy_(bias[[1, 2, 3], head].reshape(-1))
                    layer.bias_hh_l0[2 * width :].copy_(bias[0, head])
                else:
                    layer.bias_ih_l0.copy_(bias[:, head].reshape(-1))
            layer.flatten_parameters()
            self.layers.append(layer)

    def forward(
        self, inputs: torch.Tensor, initial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_history = []
        final_states = []
        for head, layer in enumerate(self.layers):
            hidden = initial[0, :, 0, head].unsqueeze(0).contiguous()
            if self.cell == "lstm":
                cell_state = initial[1, :, 0, head].unsqueeze(0).contiguous()
                output, (last_h, last_c) = layer(inputs, (hidden, cell_state))
                final = torch.stack((last_h[0], last_c[0]))
            else:
                output, last_h = layer(inputs, hidden)
                final = last_h
            hidden_history.append(output)
            final_states.append(final)
        return torch.stack(hidden_history, dim=2), torch.stack(
            final_states, dim=2
        ).unsqueeze(2)
