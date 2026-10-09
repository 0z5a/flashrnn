"""Complete FLA RWKV6 checkpoint with pinned bodies and Torch reference ops."""

import ast
import hashlib
import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import torch
from einops import rearrange
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers.modeling_utils import no_init_weights

PINS = {
    "fla--layers--rwkv6.py": "5cd488783ed4d8e4cf5c27c8d42fd0f5025e16245fbae8a54fb324759d7c8af1",
    "fla--models--rwkv6--modeling_rwkv6.py": "b5e8a02efb5fb38f54388914ae53a4c3794627adb2fe0d82fdd1e665e8c2cc3a",
    "fla--ops--rwkv6--recurrent_naive.py": "180ff8eee2e53315e6f2ae2553bdfdb1e7455de03c2d1e762778d4edd5ce85c6",
    "fla--modules--token_shift.py": "f893265cb249b4a0735347e13ef0100bd5dced216e2488618eb56fde55df85fb",
    "fla--modules--layernorm.py": "75104b5df93526180293ede0b1ed64ca131a6c663e4c77a1d8be03f36bdcf948",
}


def native_namespace(directory: Path) -> dict:
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "warnings": warnings,
        "rearrange": rearrange,
        "ACT2FN": {"swish": F.silu, "sqrelu": lambda x: F.relu(x).square()},
    }
    selected = {
        "fla--modules--layernorm.py": {"layer_norm_ref", "group_norm_ref"},
        "fla--modules--token_shift.py": {"token_shift_ref"},
        "fla--ops--rwkv6--recurrent_naive.py": {"naive_recurrent_rwkv6"},
        "fla--layers--rwkv6.py": {
            "LoRA",
            "LerpLinear",
            "DDLerpLinear",
            "RWKV6Attention",
        },
        "fla--models--rwkv6--modeling_rwkv6.py": {"RWKV6FeedForward", "RWKV6Block"},
    }
    for filename, names in selected.items():
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
    layer_norm = namespace["layer_norm_ref"]
    group_norm = namespace["group_norm_ref"]
    recurrence = namespace["naive_recurrent_rwkv6"]

    class TorchLayerNorm(nn.LayerNorm):
        def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
            prenorm: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            return layer_norm(
                x, self.weight, self.bias, residual, self.eps, prenorm, True
            )

    class TorchGroupNorm(nn.Module):
        def __init__(
            self,
            groups: int,
            hidden: int,
            elementwise_affine: bool,
            bias: bool,
            eps: float,
        ) -> None:
            super().__init__()
            assert elementwise_affine and bias
            self.weight = nn.Parameter(torch.ones(hidden))
            self.bias = nn.Parameter(torch.zeros(hidden))
            self.groups, self.eps = groups, eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return group_norm(
                x, self.weight, self.bias, self.groups, eps=self.eps, upcast=True
            )

    def recurrent(
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert cu_seqlens is None and scale == 1.0
        output, state = recurrence(
            r.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            w.transpose(1, 2),
            u,
            scale,
            initial_state,
            output_final_state,
        )
        return output.transpose(1, 2), state

    namespace.update(
        LayerNorm=TorchLayerNorm,
        GroupNorm=TorchGroupNorm,
        token_shift=namespace["token_shift_ref"],
        fused_recurrent_rwkv6=recurrent,
        chunk_rwkv6=recurrent,
    )
    return namespace


class RWKV6State(TypedDict):
    recurrent_state: torch.Tensor
    conv_state: torch.Tensor
    ffn_state: torch.Tensor | None


class RWKV6Cache:
    def __init__(self) -> None:
        self.states: list[RWKV6State] = []

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> RWKV6State:
        return self.states[index]

    def update(
        self,
        layer_idx: int,
        offset: int,
        recurrent_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        ffn_state: torch.Tensor | None = None,
    ) -> None:
        if recurrent_state is not None:
            assert conv_state is not None and offset > 0
            if layer_idx == len(self.states):
                self.states.append(
                    RWKV6State(
                        recurrent_state=recurrent_state,
                        conv_state=conv_state,
                        ffn_state=None,
                    )
                )
            else:
                self.states[layer_idx].update(
                    recurrent_state=recurrent_state, conv_state=conv_state
                )
        if ffn_state is not None:
            assert offset == 0
            self.states[layer_idx]["ffn_state"] = ffn_state


class RWKV6Reference(nn.Module):
    def __init__(self, model_dir: Path, source_dir: Path) -> None:
        super().__init__()
        manifest = json.loads((model_dir / "verified-manifest.json").read_text())
        assert manifest["revision"] == "94302fd437462b5110b06a9e83b32b8a1684e8d4"
        checkpoint = model_dir / "model.safetensors"
        with checkpoint.open("rb") as handle:
            assert (
                hashlib.file_digest(handle, "sha256").hexdigest()
                == "db508532c15e96d89566080cc39cadfa517d4cba83f782a59faa7f304b00da0a"
            )
        raw = (model_dir / "config.json").read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == "dfe066d8ea1234de066748d7dd7bde939b01d8d3"
        )
        config = SimpleNamespace(attn=None, **json.loads(raw))
        assert (config.hidden_size, config.num_hidden_layers, config.num_heads) == (
            2048,
            24,
            32,
        )
        assert config.norm_first and config.fuse_norm and not config.tie_word_embeddings
        functions = native_namespace(source_dir)
        with torch.device("meta"), no_init_weights():
            self.model = nn.Module()
            self.model.embeddings = nn.Embedding(
                config.vocab_size, config.hidden_size, config.pad_token_id
            )
            self.model.layers = nn.ModuleList(
                functions["RWKV6Block"](config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            )
            self.model.norm = functions["LayerNorm"](
                config.hidden_size, bias=config.norm_bias, eps=config.norm_eps
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
            names = tensors.keys()
            assert len(names) == 582 and set(names) == set(self.state_dict())
            weights = {name: tensors.get_tensor(name).float() for name in names}
        self.load_state_dict(weights, strict=True, assign=True)
        parameters = sum(p.numel() for p in self.parameters())
        assert parameters == 1599873024
        assert self.lm_head.weight is not self.model.embeddings.weight
        self.eval()
        self.provenance = {
            "scope": "FULL_FLA_RWKV6_PINNED_BODIES_WITH_TORCH_REFERENCE_OPERATORS",
            "checkpoint": manifest,
            "parameters": parameters,
            "checkpoint_tensors": len(names),
            "layers": 24,
            "dtype": "torch.float32",
            "source_sha256": PINS,
            "native_accelerated_fla_qualified": False,
            "blinkdl_original_implementation_crosschecked": False,
            "cache_offset": "Accepted but not accumulated; position counters are not qualified",
        }

    def forward(
        self, ids: torch.Tensor, cache: RWKV6Cache | None = None
    ) -> tuple[torch.Tensor, RWKV6Cache]:
        if cache is None:
            cache = RWKV6Cache()
        hidden = self.model.embeddings(ids)
        for layer in self.model.layers:
            hidden, _, cache = layer(hidden, past_key_values=cache, use_cache=True)
        return self.lm_head(self.model.norm(hidden)), cache
