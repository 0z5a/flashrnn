"""Pinned upstream kernels with an explicit Torch layout adapter.

This avoids importing the unavailable einops wrapper. Kernel definitions and
their original one-config autotuner are executed unchanged from pinned source;
the wrapper is a separate baseline transport, not the stock public API.
"""

import ast
import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch
import triton
import triton.language as tl

SOURCE_HASHES = {
    "lstm": "9d5a940a13b95f2df603b5d61d44d5d4b6778a65e4ee7ab0455b909b528b1cc6",
    "slstm": "4b25ef542b80ab0e0ac7cf17b3cde891a0efcb5932866d9fc7dfef6577bb894c",
}


@lru_cache(maxsize=2)
def kernel_module(cell: str) -> ModuleType:
    source = Path(__file__).parents[1] / "flashrnn/triton_fused" / f"{cell}_fw.py"
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_HASHES[cell]:
        raise ValueError("upstream Triton source differs from the pinned baseline")
    original = ast.parse(payload, filename=str(source))
    nodes = []
    for node in original.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "triton_tanh",
            "_forward_sequence_kernel",
        }:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "ENABLE_AUTOTUNING":
                nodes.append(node)
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "ENABLE_AUTOTUNING"
        ):
            nodes.append(node)
    module = ModuleType(f"flashrnn2_upstream_{cell}")
    module.__file__ = str(source)
    module.__dict__.update(triton=triton, tl=tl)
    sys.modules[module.__name__] = module
    selected = ast.Module(body=nodes, type_ignores=[])
    # Only definitions from the hash-pinned local source are executable here.
    exec(compile(selected, str(source), "exec"), module.__dict__)  # noqa: S102
    return module


def recurrence(
    wx: torch.Tensor,
    recurrent: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    cell: str = "lstm",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the public layout to the unchanged upstream forward kernel."""
    if cell not in SOURCE_HASHES or wx.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("upstream adapter supports BF16/FP16 LSTM and sLSTM")
    if not wx.is_cuda:
        raise ValueError("upstream Triton requires CUDA tensors")
    if torch.is_grad_enabled() and any(
        x.requires_grad for x in (wx, recurrent, bias, initial)
    ):
        raise ValueError("this baseline adapter exposes forward only")
    batch, steps, gates, heads, width = wx.shape
    states = 2 if cell == "lstm" else 4
    if gates != 4 or steps < 1 or width < 16 or width & (width - 1):
        raise ValueError("upstream kernel requires four gates and power-of-two D >= 16")
    if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
        raise ValueError("invalid recurrent or bias shape")
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("invalid initial state shape")
    if any(
        x.dtype != wx.dtype or x.device != wx.device for x in (recurrent, bias, initial)
    ):
        raise ValueError("all baseline inputs must share dtype and device")
    padded = triton.cdiv(batch, 16) * 16
    state = initial[:, :, 0]
    if padded != batch:
        state = torch.cat(
            (state, state.new_zeros(states, padded - batch, heads, width)), 1
        )
        wx = torch.cat(
            (wx, wx.new_zeros(padded - batch, steps, gates, heads, width)), 0
        )
    packed_state = state.permute(2, 0, 1, 3).contiguous()
    packed_wx = wx.permute(3, 1, 2, 0, 4).contiguous()
    packed_r = recurrent.permute(1, 0, 3, 2).contiguous()
    packed_bias = bias.permute(1, 0, 2).contiguous()
    output = wx.new_empty(heads, steps + 1, states, padded, width)
    module = kernel_module(cell)
    module._forward_sequence_kernel[(heads, triton.cdiv(batch, 16))](
        states_initial=packed_state,
        Wx=packed_wx,
        R=packed_r,
        b=packed_bias,
        states_all=output,
        gates_all=None,
        T=steps,
        NS=states,
        B=padded,
        NH=heads,
        DH=width,
        NGI=4,
        NGR=4,
        OUTPUT_GATES=False,
        DTYPE=tl.bfloat16 if wx.dtype == torch.bfloat16 else tl.float16,
    )
    history = output[:, 1:, :, :batch].permute(2, 3, 1, 0, 4)
    final = output[:, -1:, :, :batch].permute(2, 3, 1, 0, 4)
    return history, final
