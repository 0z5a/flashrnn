"""Full GLA checkpoint with pinned FLA layers and explicit Torch primitives.

Attention/block/MLP bodies and the recurrent/RMS reference functions are
unchanged upstream AST. Fused RMS/gating/SwiGLU operations use Torch math;
this reference does not claim native FLA CUDA execution or its rounding.
"""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from einops import rearrange, repeat
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

PINS = {
    "fla--layers--gla.py": "ca3e42418ed8e9b0341aa8fb98802e7840f76e35ca1a72eaa78fecea3d7d2004",
    "fla--models--gla--modeling_gla.py": "72eb599f24f8c0f88107254c245e8678a9efc8c704192be06fec481b0f00aea8",
    "fla--ops--gla--naive.py": "ff00ae27619a65f7f7e90364d6a8796bd4cff921d7f92754e3ca20d9c35da20e",
    "fla--modules--mlp.py": "36ccc4be7d9dcc01f98931f3ff6695acad4dd2b7552a7ee904faf3bfda1aff5b",
    "fla--modules--layernorm.py": "75104b5df93526180293ede0b1ed64ca131a6c663e4c77a1d8be03f36bdcf948",
}


def native_namespace(directory: Path) -> dict:
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "rearrange": rearrange,
        "repeat": repeat,
        "ACT2FN": {"swish": F.silu},
    }
    selected = {
        "fla--modules--layernorm.py": {"rms_norm_ref"},
        "fla--ops--gla--naive.py": {"naive_recurrent_gla"},
        "fla--modules--mlp.py": {"GatedMLP", "SwiGLULinear"},
        "fla--layers--gla.py": {"GatedLinearAttention"},
        "fla--models--gla--modeling_gla.py": {"GLABlock"},
    }
    for filename, names in selected.items():
        path = directory / filename
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

    rms = namespace["rms_norm_ref"]
    recurrence = namespace["naive_recurrent_gla"]

    class TorchRMSNorm(nn.Module):
        def __init__(
            self, hidden_size: int, eps: float, elementwise_affine: bool = True
        ) -> None:
            super().__init__()
            assert elementwise_affine
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
            prenorm: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            result = rms(x, self.weight, None, residual, self.eps, prenorm, True)
            if prenorm:
                output, combined = result
                return output, combined.to(x.dtype)
            return result

    class TorchRMSNormGated(TorchRMSNorm):
        def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            # Keep normalization and gating in FP32 until the output cast.
            normalized = rms(x.float(), self.weight.float(), None, eps=self.eps)
            return (normalized * F.silu(gate.float())).to(x.dtype)

    def recurrent(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert cu_seqlens is None
        assert (gk is None) != (g is None)
        return recurrence(
            q, k, v, gk if gk is not None else g, initial_state, output_final_state
        )

    def swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return (F.silu(gate.float()) * value.float()).to(gate.dtype)

    def swiglu_linear(
        gate: torch.Tensor,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return F.linear(swiglu(gate, value), weight, bias)

    namespace.update(
        RMSNorm=TorchRMSNorm,
        FusedRMSNormGated=TorchRMSNormGated,
        GLAMLP=namespace["GatedMLP"],
        fused_recurrent_gla=recurrent,
        chunk_gla=recurrent,
        fused_chunk_gla=recurrent,
        swiglu=swiglu,
        swiglu_linear=swiglu_linear,
    )
    return namespace


class RecurrentCache:
    def __init__(self) -> None:
        self.states: list[dict[str, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.states[index]

    def update(
        self,
        recurrent_state: torch.Tensor,
        conv_state: None,
        layer_idx: int,
        offset: int,
    ) -> None:
        assert conv_state is None and offset > 0
        state = {"recurrent_state": recurrent_state}
        if layer_idx == len(self.states):
            self.states.append(state)
        else:
            self.states[layer_idx] = state


class GLAReference(nn.Module):
    def __init__(
        self, model_dir: Path, source_dir: Path, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        config_path = model_dir / "config.json"
        checkpoint = model_dir / "model.safetensors"
        manifest = json.loads((model_dir / "verified-manifest.json").read_text())
        assert manifest["revision"] == "46b15820a4df269e99aed9d709e017677c15d24b"
        expected = next(x for x in manifest["files"] if x["file"] == checkpoint.name)
        assert (
            expected["checksum"]
            == "97c02567af31fc5ef98280a5d84e1c6b98842d8b92972940626bce8ebb3ed73d"
        )
        with checkpoint.open("rb") as handle:
            assert (
                hashlib.file_digest(handle, "sha256").hexdigest()
                == expected["checksum"]
            )
        raw = json.loads(config_path.read_text())
        assert raw["hidden_size"] == 2048 and raw["num_hidden_layers"] == 24
        assert raw["num_heads"] == 4 and raw["vocab_size"] == 32000
        assert raw["attn"] is None and raw["feature_map"] is None
        assert not raw["use_short_conv"] and not raw["tie_word_embeddings"]
        config = SimpleNamespace(fuse_swiglu=True, **raw)
        functions = native_namespace(source_dir)
        with torch.device("meta"):
            self.model = nn.Module()
            self.model.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.model.layers = nn.ModuleList(
                functions["GLABlock"](config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            )
            self.model.norm = functions["RMSNorm"](
                config.hidden_size, eps=config.norm_eps
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
            names = tensors.keys()
            assert set(names) == set(self.state_dict())
            weights = {name: tensors.get_tensor(name).to(dtype) for name in names}
        self.load_state_dict(weights, strict=True, assign=True)
        assert len(weights) == 339
        assert sum(p.numel() for p in self.parameters()) == 1365514240
        assert self.lm_head.weight is not self.model.embeddings.weight
        self.eval()
        self.provenance = {
            "scope": "FULL_GLA_FLA_LAYER_AST_WITH_EXPLICIT_TORCH_PRIMITIVES",
            "checkpoint": manifest,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_sha256": PINS,
            "parameters": 1365514240,
            "checkpoint_tensors": 339,
            "dtype": str(dtype),
            "accelerated_native_fla_qualified": False,
        }

    def forward(
        self, input_ids: torch.Tensor, cache: RecurrentCache | None = None
    ) -> tuple[torch.Tensor, RecurrentCache]:
        if cache is None:
            cache = RecurrentCache()
        hidden = self.model.embeddings(input_ids)
        for layer in self.model.layers:
            hidden, _, cache = layer(hidden, past_key_values=cache, use_cache=True)
        logits = self.lm_head(self.model.norm(hidden))
        return logits, cache
