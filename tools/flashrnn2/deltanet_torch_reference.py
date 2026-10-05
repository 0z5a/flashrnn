"""Complete DeltaNet checkpoint with pinned FLA bodies and Torch primitives."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import torch
from gla_torch_reference import PINS as COMMON_PINS
from gla_torch_reference import native_namespace as common_namespace
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

PINS = {
    "fla--layers--delta_net.py": "f503aedfa87a78100054c4ab3a66370d13828958ea43cc94201948adf33f3df5",
    "fla--models--delta_net--modeling_delta_net.py": "3c33375ad60a6ddc5640f73cf5a868f5a33799cd883c8572b9fbf2334ad83d12",
    "fla--ops--delta_rule--naive.py": "6ce9b2a7e5e28c59558de0f7c2c5d75ecda47e2986877c22fa8188ce6eafba9f",
    "fla--modules--convolution.py": "11fd87ade687acfb470a0dfbb46bb9256788708c321c149c218c780c6457bab4",
}


def convolution(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    seq_idx: None = None,
) -> torch.Tensor:
    assert seq_idx is None
    output = F.conv1d(
        x, weight[:, None], bias, padding=weight.shape[-1] - 1, groups=x.shape[1]
    )[..., : x.shape[-1]]
    return F.silu(output) if activation is not None else output


def convolution_step(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    conv_state.copy_(conv_state.roll(shifts=-1, dims=-1))
    conv_state[:, :, -1] = x
    output = (conv_state * weight).sum(-1)
    if bias is not None:
        output = output + bias
    return F.silu(output) if activation is not None else output


def native_namespace(directory: Path) -> dict:
    # Reuse the already-qualified Torch RMS/SwiGLU primitives and pinned GatedMLP.
    namespace = common_namespace(directory)
    for filename, names in {
        "fla--modules--convolution.py": {"ShortConvolution"},
        "fla--ops--delta_rule--naive.py": {"delta_rule_recurrence"},
        "fla--layers--delta_net.py": {"DeltaNet", "elu_p1", "sum_norm"},
        "fla--models--delta_net--modeling_delta_net.py": {"DeltaNetBlock"},
    }.items():
        path = directory / filename
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == PINS[filename]
        nodes = [
            n
            for n in ast.parse(data).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        ]
        assert {n.name for n in nodes} == names
        module = ast.Module(
            body=ast.parse("from __future__ import annotations\n").body + nodes,
            type_ignores=[],
        )
        exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    recurrence = namespace["delta_rule_recurrence"]

    def recurrent(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert cu_seqlens is None and use_qk_l2norm_in_kernel
        # FLA's l2norm_fwd uses sqrt(sum(x*x) + 1e-6), not clamp(norm, eps).
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        output, state = recurrence(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            beta.transpose(1, 2),
            initial_state,
            output_final_state,
        )
        return output.transpose(1, 2), state

    namespace.update(
        causal_conv1d_fn=convolution,
        causal_conv1d_update_cuda=convolution_step,
        DeltaNetMLP=namespace["GatedMLP"],
        chunk_delta_rule=recurrent,
        fused_recurrent_delta_rule=recurrent,
    )
    return namespace


class DeltaState(TypedDict):
    recurrent_state: torch.Tensor
    conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DeltaCache:
    def __init__(self) -> None:
        self.states: list[DeltaState] = []

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> DeltaState:
        return self.states[index]

    def update(
        self,
        recurrent_state: torch.Tensor,
        conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layer_idx: int,
        offset: int,
    ) -> None:
        assert offset > 0
        state = DeltaState(recurrent_state=recurrent_state, conv_state=conv_state)
        if layer_idx == len(self.states):
            self.states.append(state)
        else:
            self.states[layer_idx] = state


class DeltaNetReference(nn.Module):
    def __init__(self, model_dir: Path, source_dir: Path) -> None:
        super().__init__()
        manifest = json.loads((model_dir / "verified-manifest.json").read_text())
        assert manifest["revision"] == "b4dcbbafd4fde802717bdec3008d4aba9cb3a1f8"
        checkpoint = model_dir / "model.safetensors"
        with checkpoint.open("rb") as handle:
            assert (
                hashlib.file_digest(handle, "sha256").hexdigest()
                == "b3990ff772f7254009827b333f8ab05e8865d2fc889be71275e6b4756f123817"
            )
        raw = (model_dir / "config.json").read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == "d0a849ebc7d59a5935a451f511a4109e2a8ebc31"
        )
        config = SimpleNamespace(fuse_swiglu=True, **json.loads(raw))
        assert (config.hidden_size, config.num_hidden_layers, config.num_heads) == (
            2048,
            24,
            16,
        )
        assert (
            config.use_short_conv and config.qk_norm == "l2" and config.conv_size == 4
        )
        assert (
            config.attn is None
            and not config.tie_word_embeddings
            and not config.use_gate
        )
        functions = native_namespace(source_dir)
        with torch.device("meta"):
            self.model = nn.Module()
            self.model.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.model.layers = nn.ModuleList(
                functions["DeltaNetBlock"](config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            )
            self.model.norm = functions["RMSNorm"](
                config.hidden_size, eps=config.norm_eps
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
            names = tensors.keys()
            assert set(names) == set(self.state_dict())
            weights = {name: tensors.get_tensor(name).float() for name in names}
        self.load_state_dict(weights, strict=True, assign=True)
        parameters = sum(p.numel() for p in self.parameters())
        assert parameters == 1365677056
        assert self.lm_head.weight is not self.model.embeddings.weight
        self.eval()
        self.provenance = {
            "scope": "FULL_DELTANET_PINNED_FLA_BODIES_WITH_TORCH_PRIMITIVES",
            "checkpoint": manifest,
            "parameters": parameters,
            "checkpoint_tensors": len(weights),
            "layers": 24,
            "source_sha256": {**COMMON_PINS, **PINS},
            "dtype": "torch.float32",
            "accelerated_native_fla_qualified": False,
        }

    def forward(
        self, ids: torch.Tensor, cache: DeltaCache | None = None
    ) -> tuple[torch.Tensor, DeltaCache]:
        if cache is None:
            cache = DeltaCache()
        hidden = self.model.embeddings(ids)
        for layer in self.model.layers:
            hidden, _, cache = layer(hidden, past_key_values=cache, use_cache=True)
        return self.lm_head(self.model.norm(hidden)), cache
