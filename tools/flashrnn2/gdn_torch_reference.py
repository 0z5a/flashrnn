"""Complete Gated DeltaNet checkpoint with its matching 2024 FLA model body."""

import ast
import hashlib
import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from deltanet_torch_reference import DeltaCache
from deltanet_torch_reference import native_namespace as delta_namespace
from einops import rearrange, repeat
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

PINS = {
    "gated_deltanet.py": "78d193c48856b87101607822ac8ba09e6f479c4d53550b12e534cace003d1010",
    "modeling_gated_deltanet.py": "deee2f344ac7fd6976b0c00a9a2d5d9ea6cac58de113cb38439751b2d6c59e9f",
    "convolution.py": "9ac3e5222945d514b8c849f0c5ba71669543903a9e00d68696fef90bb80e4fbc",
}
FLA_REVISION = "bcd9e79bfea85a394a023663f164587b506e0422"
REVISION = "c83bdada453cde56932f37be71338df22ca29b7d"
WEIGHT_SHA256 = "f599714aff09f07efce88afc4c327806c8d7082ccc4469ff4d1d1a2e707bf4f9"


def gdn_namespace(source_dir: Path, delta_dir: Path) -> dict:
    namespace = delta_namespace(delta_dir)
    namespace["ACT2FN"]["silu"] = F.silu

    def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        value = x.float()
        return (value * torch.rsqrt(value.square().sum(-1, keepdim=True) + eps)).to(
            x.dtype
        )

    def recurrence(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        offsets: None,
        head_first: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert offsets is None and not head_first
        assert q.shape[:-1] == k.shape[:-1] == v.shape[:-1]
        batch, steps, heads, key_width = q.shape
        assert g.shape == beta.shape == (batch, steps, heads)
        q = q.float() * key_width**-0.5
        k = k.float()
        values = v.float()
        state = (
            values.new_zeros(batch, heads, key_width, values.shape[-1])
            if initial_state is None
            else initial_state.float().clone()
        )
        outputs = []
        for step in range(steps):
            state = state * g[:, step].float().exp()[..., None, None]
            key = k[:, step]
            delta = values[:, step] - (state * key[..., None]).sum(-2)
            delta = delta * beta[:, step].float()[..., None]
            state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, step], state))
        output = torch.stack(outputs, dim=1).to(v.dtype)
        return output, state if output_final_state else None

    namespace.update(
        causal_conv1d_fn=None, causal_conv1d_update=None, warnings=warnings
    )
    path = source_dir / "convolution.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[path.name]
    nodes = [
        node
        for node in ast.parse(path.read_bytes()).body
        if isinstance(node, ast.ClassDef) and node.name == "ShortConvolution"
    ]
    assert len(nodes) == 1
    exec(  # noqa: S102
        compile(
            ast.Module(
                body=ast.parse("from __future__ import annotations\n").body + nodes,
                type_ignores=[],
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    old_convolution = namespace["ShortConvolution"]

    class TorchShortConvolution(old_convolution):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, use_fast_conv1d=False, **kwargs)

    namespace.update(
        math=math,
        rearrange=rearrange,
        repeat=repeat,
        ShortConvolution=TorchShortConvolution,
        FusedRMSNormSwishGate=namespace["FusedRMSNormGated"],
        l2_norm=normalize,
        chunk_gated_delta_rule=recurrence,
        fused_recurrent_gated_delta_rule=recurrence,
        swiglu_linear=namespace["swiglu_linear"],
    )

    def rms_norm_linear(
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor | None,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        normalized = (normalized * norm_weight.float()).to(x.dtype)
        if norm_bias is not None:
            normalized = normalized + norm_bias
        return F.linear(normalized, linear_weight, linear_bias)

    namespace["rms_norm_linear"] = rms_norm_linear
    for filename, classnames in (
        ("gated_deltanet.py", {"GatedDeltaNet"}),
        ("modeling_gated_deltanet.py", {"GatedDeltaNetMLP", "GatedDeltaNetBlock"}),
    ):
        path = source_dir / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[filename]
        nodes = [
            node
            for node in ast.parse(path.read_bytes()).body
            if isinstance(node, ast.ClassDef) and node.name in classnames
        ]
        assert {node.name for node in nodes} == classnames
        module = ast.Module(
            body=ast.parse("from __future__ import annotations\n").body + nodes,
            type_ignores=[],
        )
        exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


class GatedDeltaNetReference(nn.Module):
    def __init__(
        self,
        model_dir: Path,
        source_dir: Path,
        delta_dir: Path,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        config_path = model_dir / "config.json"
        checkpoint = model_dir / "model.safetensors"
        manifest = json.loads((model_dir / "verified-manifest.json").read_text())
        assert manifest["revision"] == REVISION
        weight = next(x for x in manifest["files"] if x["file"] == checkpoint.name)
        assert weight["size"] == 799109000 and weight["checksum"] == WEIGHT_SHA256
        with checkpoint.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == WEIGHT_SHA256
        raw = json.loads(config_path.read_text())
        assert raw["model_type"] == "gated_deltanet"
        assert (raw["hidden_size"], raw["num_hidden_layers"], raw["num_heads"]) == (
            1024,
            24,
            4,
        )
        assert raw["head_dim"] == 256 and raw["expand_v"] == 1
        assert raw["use_short_conv"] and raw["use_gate"] and raw["tie_word_embeddings"]
        assert raw["attn"] is None and raw["conv_size"] == 4
        assert raw["norm_first"] is False
        config = SimpleNamespace(**raw)
        functions = gdn_namespace(source_dir, delta_dir)
        with torch.device("meta"):
            self.model = nn.Module()
            self.model.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.model.layers = nn.ModuleList(
                functions["GatedDeltaNetBlock"](config, layer_idx=i)
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
        assert len(weights) == 435
        self.load_state_dict(weights, strict=True, assign=True)
        assert torch.equal(self.lm_head.weight, self.model.embeddings.weight)
        self.lm_head.weight = self.model.embeddings.weight
        parameters = sum(p.numel() for p in self.parameters())
        assert parameters == 399531296 - 32000 * 1024
        self.eval()
        self.provenance = {
            "scope": "FULL_GATED_DELTANET_2024_FLA_LAYER_AST_WITH_EXPLICIT_TORCH_RECURRENCE",
            "checkpoint": manifest,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_sha256": PINS,
            "fla_revision": FLA_REVISION,
            "stored_parameters": 399531296,
            "effective_tied_parameters": parameters,
            "checkpoint_tensors": len(weights),
            "layers": len(self.model.layers),
            "dtype": str(dtype),
            "accelerated_native_fla_qualified": False,
        }

    def forward(
        self, input_ids: torch.Tensor, cache: DeltaCache | None = None
    ) -> tuple[torch.Tensor, DeltaCache]:
        if cache is None:
            cache = DeltaCache()
        hidden = self.model.embeddings(input_ids)
        for layer in self.model.layers:
            hidden, _, cache = layer(hidden, past_key_values=cache, use_cache=True)
        logits = self.lm_head(self.model.norm(hidden))
        return logits, cache
