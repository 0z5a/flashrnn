"""Crosscheck the Torch GDN-2 adapter against the pinned FLA layer body."""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch
from einops import rearrange, repeat
from gdn2_torch_reference import load_reference
from torch import nn
from torch.nn import functional as F


class ShortConvolution(nn.Conv1d):
    def __init__(self, hidden_size: int, kernel_size: int, bias: bool, activation: str):
        assert activation == "silu"
        super().__init__(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias
        )

    def forward(self, x: torch.Tensor, **kwargs):
        values = F.conv1d(
            F.pad(x.transpose(1, 2), (self.kernel_size[0] - 1, 0)),
            self.weight,
            groups=self.in_channels,
        )
        return F.silu(values.transpose(1, 2)), None


class FusedRMSNormGated(nn.Module):
    def __init__(self, dim: int, activation: str, eps: float):
        super().__init__()
        assert activation == "sigmoid"
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor):
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
            * self.weight
            * gate.sigmoid()
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fla-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    model = load_reference(args.model, args.fla_source)
    source = args.fla_source / "fla/layers/gdn2.py"
    raw = source.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "533a29b761885ed0206476a4d2e93f769b1eb6ac"
    )
    naive_source = args.fla_source / "fla/ops/gdn2/naive.py"
    definition = importlib.util.spec_from_file_location(
        "pinned_gdn2_naive_for_layer", naive_source
    )
    assert definition is not None and definition.loader is not None
    naive = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(naive)

    def recurrent(
        q,
        k,
        v,
        g,
        b,
        w,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    ):
        assert (
            initial_state is None
            and not output_final_state
            and use_qk_l2norm_in_kernel
            and cu_seqlens is None
        )
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        return naive.naive_recurrent_gdn2(q, k, v, g, b, w)

    node = next(
        node
        for node in ast.parse(raw).body
        if isinstance(node, ast.ClassDef) and node.name == "GatedDeltaNet2"
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "rearrange": rearrange,
        "repeat": repeat,
        "ShortConvolution": ShortConvolution,
        "FusedRMSNormGated": FusedRMSNormGated,
        "get_layer_cache": lambda self, cache: None,
        "update_layer_cache": lambda *args, **kwargs: None,
        "chunk_gdn2": recurrent,
        "fused_recurrent_gdn2": recurrent,
    }
    tree = ast.Module(
        body=ast.parse("from __future__ import annotations\n").body + [node],
        type_ignores=[],
    )
    exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102
    official = namespace["GatedDeltaNet2"](
        hidden_size=512,
        expand_v=1.0,
        head_dim=128,
        num_heads=4,
        num_v_heads=4,
        mode="chunk",
        use_short_conv=True,
        allow_neg_eigval=False,
        conv_size=4,
        conv_bias=False,
        layer_idx=0,
        norm_eps=1e-6,
    )
    adapter = model.layers[0].token_mixer
    assert set(official.state_dict()) == set(adapter.state_dict())
    official.load_state_dict(adapter.state_dict(), strict=True)
    official.eval()
    cases = []
    with torch.inference_mode():
        for batch, steps in ((1, 5), (2, 5), (4, 17)):
            ids = torch.arange(batch * steps).reshape(batch, steps) + 100
            hidden = model.layers[0].token_mixer_norm(model.embed_tokens(ids))
            left = official(hidden)[0]
            right = adapter(hidden)[0]
            difference = (left - right).abs()
            cases.append(
                {
                    "batch": batch,
                    "steps": steps,
                    "max_abs": difference.max().item(),
                    "pass": torch.allclose(left, right, atol=1e-5, rtol=1e-5),
                }
            )
    result = {
        "status": "PASS" if all(case["pass"] for case in cases) else "NUMERICAL_FAILED",
        "source_git_blob": "533a29b761885ed0206476a4d2e93f769b1eb6ac",
        "layer0_parameters": len(adapter.state_dict()),
        "cases": cases,
        "native_cuda_qualified": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
