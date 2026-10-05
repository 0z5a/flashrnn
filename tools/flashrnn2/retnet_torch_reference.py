"""Pinned RetNet layers with explicit Torch retention and rotary primitives."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from gla_torch_reference import PINS as GLA_PINS
from gla_torch_reference import RecurrentCache, native_namespace
from safetensors import safe_open
from torch import nn

PINS = {
    "fla--layers--multiscale_retention.py": "0b7d758e9ab4657b1f3d60953b6f603d1ff6b828b8c8a9f600f4da5e15547fce",
    "fla--models--retnet--modeling_retnet.py": "7d094b12a7a6f45880c9e8dafdc906e5f4f65a819d284a918a7a75fce4d755d5",
    "fla--modules--rotary.py": "1696321159779e18de40afff61901e1eb5c2392be2e366764d1fc4dc8dd94004",
    "fla--ops--retention--naive.py": "8ddbd4f02a0c969666784717da5398e69901335402ec726443163132a66d3558",
}


def recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """BTHD inputs; fixed per-head decay, FP32 matrix state, post-update readout."""
    assert cu_seqlens is None
    dtype = q.dtype
    q, k, v = (x.transpose(1, 2).float() for x in (q, k, v))
    batch, heads, steps, width = q.shape
    decay = 1 - torch.exp2(-5.0 - torch.arange(heads, device=q.device).float())
    decay = decay.view(1, heads, 1, 1)
    state = q.new_zeros(batch, heads, width, v.shape[-1])
    if initial_state is not None:
        state = state + initial_state.float()
    outputs = []
    for step in range(steps):
        state = decay * state + k[:, :, step, :, None] * v[:, :, step, None, :]
        outputs.append((q[:, :, step, :, None] * width**-0.5 * state).sum(-2))
    output = torch.stack(outputs, dim=2).transpose(1, 2).to(dtype)
    return output, state if output_final_state else None


def retnet_namespace(source: Path, common: Path) -> dict:
    namespace = native_namespace(common)
    selected = {
        "fla--modules--rotary.py": {
            "rotate_half",
            "rotary_embedding_ref",
            "RotaryEmbedding",
        },
        "fla--ops--retention--naive.py": {"naive_retention"},
        "fla--layers--multiscale_retention.py": {"MultiScaleRetention"},
        "fla--models--retnet--modeling_retnet.py": {"RetNetBlock"},
    }
    for filename, names in selected.items():
        path = source / filename
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == PINS[filename]
        nodes = [
            node
            for node in ast.parse(data).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        ]
        assert {node.name for node in nodes} == names
        module = ast.Module(
            body=ast.parse("from __future__ import annotations\n").body + nodes,
            type_ignores=[],
        )
        exec(compile(module, str(path), "exec"), namespace)  # noqa: S102

    def rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool = False,
        seqlen_offsets: int = 0,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert cu_seqlens is None and isinstance(seqlen_offsets, int)
        end = seqlen_offsets + x.shape[1]
        assert 0 <= seqlen_offsets < end <= cos.shape[0]
        return namespace["rotary_embedding_ref"](
            x, cos[seqlen_offsets:end], sin[seqlen_offsets:end], interleaved
        )

    namespace.update(
        RetNetMLP=namespace["GatedMLP"],
        rotary_embedding=rotary,
        chunk_retention=recurrent_retention,
        fused_chunk_retention=recurrent_retention,
        fused_recurrent_retention=recurrent_retention,
    )
    return namespace


class RetNetCache(RecurrentCache):
    def __init__(self) -> None:
        super().__init__()
        self.lengths: list[int] = []

    def get_seq_length(self, layer_idx: int) -> int:
        return self.lengths[layer_idx] if layer_idx < len(self.lengths) else 0

    def update(
        self,
        recurrent_state: torch.Tensor,
        conv_state: None,
        layer_idx: int,
        offset: int,
    ) -> None:
        super().update(recurrent_state, conv_state, layer_idx, offset)
        if layer_idx == len(self.lengths):
            self.lengths.append(offset)
        else:
            self.lengths[layer_idx] += offset


class RetNetReference(nn.Module):
    def __init__(self, model_dir: Path, source_dir: Path, common_dir: Path) -> None:
        super().__init__()
        config_path = model_dir / "config.json"
        checkpoint = model_dir / "model.safetensors"
        manifest = json.loads((model_dir / "verified-manifest.json").read_text())
        assert manifest["revision"] == "7fddefc4d5e196a8d1f076bb7612d54321b3effe"
        expected = next(x for x in manifest["files"] if x["file"] == checkpoint.name)
        assert (
            expected["checksum"]
            == "de46d5e8ce1d524ac790776abfc8c3293e1c42df86a461a719a101197e8af4bd"
        )
        with checkpoint.open("rb") as handle:
            assert (
                hashlib.file_digest(handle, "sha256").hexdigest()
                == expected["checksum"]
            )
        data = config_path.read_bytes()
        assert (
            hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            == "59f7af9e07410f945bcfa1a745157dacb54e5f97"
        )
        raw = json.loads(data)
        assert raw["hidden_size"] == 2048 and raw["num_hidden_layers"] == 24
        assert raw["num_heads"] == 8 and raw["vocab_size"] == 32000
        assert raw["attn"] is None and raw["feature_map"] is None
        assert not raw["use_short_conv"] and not raw["tie_word_embeddings"]
        assert raw["attn_mode"] == "chunk" and raw["num_kv_heads"] is None
        config = SimpleNamespace(fuse_swiglu=True, **raw)
        functions = retnet_namespace(source_dir, common_dir)
        with torch.device("meta"):
            self.model = nn.Module()
            self.model.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.model.layers = nn.ModuleList(
                functions["RetNetBlock"](config, i)
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
        # inv_freq is nonpersistent and therefore absent from the checkpoint.
        for layer in self.model.layers:
            rotary = layer.attn.rotary
            rotary.inv_freq = rotary._compute_inv_freq(device="cpu")
        assert len(weights) == 267
        assert sum(p.numel() for p in self.parameters()) == 1351727104
        assert all(b.device.type == "cpu" for b in self.buffers())
        assert self.lm_head.weight is not self.model.embeddings.weight
        self.eval()
        self.provenance = {
            "scope": "FULL_RETNET_FLA_LAYER_AST_WITH_EXPLICIT_TORCH_RETENTION",
            "checkpoint": manifest,
            "config_sha256": hashlib.sha256(data).hexdigest(),
            "source_sha256": PINS,
            "common_source_sha256": GLA_PINS,
            "parameters": 1351727104,
            "checkpoint_tensors": 267,
            "layers": 24,
            "dtype": "torch.float32",
            "accelerated_native_fla_qualified": False,
        }

    def forward(
        self, input_ids: torch.Tensor, cache: RetNetCache | None = None
    ) -> tuple[torch.Tensor, RetNetCache]:
        if cache is None:
            cache = RetNetCache()
        hidden = self.model.embeddings(input_ids)
        for layer in self.model.layers:
            hidden, _, cache = layer(hidden, past_key_values=cache, use_cache=True)
        assert len(cache.lengths) == 24 and len(set(cache.lengths)) == 1
        return self.lm_head(self.model.norm(hidden)), cache
