"""CPU-only GDN-2 layer adapter for the pinned Monostich-2 checkpoint."""

import hashlib
import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

WEIGHT_SHA = "60a7bb7ac497c1373ccbe073ebeaed0b54c8af0d488316808082a1e970056508"


def load_reference(model_dir: Path, fla_dir: Path):
    manifest = json.loads((model_dir / "verified-manifest.json").read_text())
    assert manifest["revision"] == "7779182b4dfad5939827f40cd1c819843a6825e2"
    assert manifest["files"]["model.safetensors"]["sha256"] == WEIGHT_SHA
    source = fla_dir / "fla/ops/gdn2/naive.py"
    expected = "1a4f46f5dde3eab9abd361de9136ede65520c07e"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected
    spec = importlib.util.spec_from_file_location("pinned_fla_gdn2_naive", source)
    assert spec is not None and spec.loader is not None
    naive_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(naive_module)
    recurrent = naive_module.naive_recurrent_gdn2

    class NormGate(nn.Module):
        def __init__(self, dim: int, eps: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
            return normalized * self.weight * gate.sigmoid()

    class TorchGDN2(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            expand_v: float,
            head_dim: int,
            num_heads: int,
            num_v_heads: int,
            mode: str,
            use_short_conv: bool,
            allow_neg_eigval: bool,
            conv_size: int,
            conv_bias: bool,
            layer_idx: int,
            norm_eps: float,
        ) -> None:
            super().__init__()
            assert (hidden_size, expand_v, head_dim, num_heads, num_v_heads) == (
                512,
                1.0,
                128,
                4,
                4,
            )
            assert mode == "chunk" and use_short_conv and not allow_neg_eigval
            assert conv_size == 4 and not conv_bias and layer_idx >= 0
            self.q_proj = nn.Linear(512, 512, bias=False)
            self.k_proj = nn.Linear(512, 512, bias=False)
            self.v_proj = nn.Linear(512, 512, bias=False)
            self.q_conv1d = nn.Conv1d(512, 512, 4, groups=512, bias=False)
            self.k_conv1d = nn.Conv1d(512, 512, 4, groups=512, bias=False)
            self.v_conv1d = nn.Conv1d(512, 512, 4, groups=512, bias=False)
            self.f_proj = nn.Sequential(
                nn.Linear(512, 128, bias=False), nn.Linear(128, 512, bias=False)
            )
            self.b_proj = nn.Linear(512, 512, bias=False)
            self.w_proj = nn.Linear(512, 512, bias=False)
            self.A_log = nn.Parameter(torch.zeros(4))
            self.dt_bias = nn.Parameter(torch.zeros(512))
            self.g_proj = nn.Sequential(
                nn.Linear(512, 128, bias=False), nn.Linear(128, 512)
            )
            self.o_norm = NormGate(128, norm_eps)
            self.o_proj = nn.Linear(512, 512, bias=False)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: None = None,
            use_cache: bool = False,
        ) -> tuple[torch.Tensor, None, None]:
            assert attention_mask is None and past_key_values is None and not use_cache
            batch, steps, _ = hidden_states.shape

            def conv(projected: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
                values = F.conv1d(
                    F.pad(projected.transpose(1, 2), (3, 0)), layer.weight, groups=512
                )
                return F.silu(values.transpose(1, 2))

            q = conv(self.q_proj(hidden_states), self.q_conv1d).reshape(
                batch, steps, 4, 128
            )
            k = conv(self.k_proj(hidden_states), self.k_conv1d).reshape(
                batch, steps, 4, 128
            )
            v = conv(self.v_proj(hidden_states), self.v_conv1d).reshape(
                batch, steps, 4, 128
            )
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            decay = F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
            decay = (
                -decay.reshape(batch, steps, 4, 128)
                * self.A_log.exp()[None, None, :, None]
            )
            erase = self.b_proj(hidden_states).sigmoid().reshape(batch, steps, 4, 128)
            write = self.w_proj(hidden_states).sigmoid().reshape(batch, steps, 4, 128)
            output, _ = recurrent(q, k, v, decay, erase, write)
            gate = self.g_proj(hidden_states).reshape(batch, steps, 4, 128)
            output = self.o_norm(output, gate).reshape(batch, steps, 512)
            return self.o_proj(output), None, None

    sys.path.insert(0, str(model_dir))
    from tiny_gdn import model as author
    from tiny_gdn.config import TinyGDNConfig

    author.GatedDeltaNet2 = TorchGDN2
    payload = json.loads((model_dir / "config.json").read_text())
    allowed = {field.name for field in fields(TinyGDNConfig)}
    config = TinyGDNConfig(
        **{key: value for key, value in payload.items() if key in allowed}
    )
    with torch.device("meta"):
        model = author.TinyGDNForCausalLM(config)
    checkpoint = model_dir / "model.safetensors"
    with checkpoint.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == WEIGHT_SHA
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        names = tensors.keys()
        assert len(names) == 586 and set(names) == set(model.state_dict())
        weights = {name: tensors.get_tensor(name).float() for name in names}
    model.load_state_dict(weights, strict=True, assign=True)
    for layer in model.layers:
        if layer.layer_type == "full_attention":
            rotary = layer.token_mixer.rotary
            steps = torch.arange(0, rotary.rotary_dim, 2, dtype=torch.float32)
            rotary.inverse_frequency = 1.0 / (
                config.rope_theta ** (steps / rotary.rotary_dim)
            )
    assert sum(p.numel() for p in model.parameters()) == 149108320
    return model.eval()
