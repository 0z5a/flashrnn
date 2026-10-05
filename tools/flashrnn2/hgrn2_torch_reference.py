"""Complete HGRN2 checkpoint using pinned FLA bodies and Torch recurrence."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from gla_torch_reference import RecurrentCache, native_namespace
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

PINS = {
    "hgrn2.py": "94886cbdf9b0b15763e271cefbaa9c312af42c4c0a00fda85cfb4411c7766469",
    "modeling_hgrn2.py": "b624b960ddb4b340dfa44654eb1c6cc8c3deb92534004b57561e58a7f226929b",
}
REVISION = "2f413dd9b63591b9b177bbf940942ea7eb70abfe"
WEIGHT_SHA256 = "92ea939b56018e51e696d1cb38970ed2797fad97bfe1eacff144bdc0397cdc7c"


def hgrn2_namespace(source_dir: Path, common_dir: Path) -> dict:
    namespace = native_namespace(common_dir)
    rms = namespace["rms_norm_ref"]

    class TorchRMSNormWithBias(namespace["RMSNorm"]):
        def __init__(
            self, hidden_size: int, eps: float, elementwise_affine: bool = True
        ) -> None:
            super().__init__(hidden_size, eps, elementwise_affine)
            self.register_parameter("bias", None)

    def rms_norm_linear(
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor | None,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = rms(x, norm_weight, norm_bias, eps=1e-6, upcast=True)
        return F.linear(normalized, linear_weight, linear_bias)

    namespace.update(
        RMSNorm=TorchRMSNormWithBias,
        HGRN2MLP=namespace["GatedMLP"],
        rms_norm_linear=rms_norm_linear,
        swish=F.silu,
    )
    for filename, classname in (
        ("hgrn2.py", "HGRN2Attention"),
        ("modeling_hgrn2.py", "HGRN2Block"),
    ):
        path = source_dir / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[filename]
        nodes = [
            node
            for node in ast.parse(path.read_bytes()).body
            if isinstance(node, ast.ClassDef) and node.name == classname
        ]
        assert len(nodes) == 1
        module = ast.Module(
            body=ast.parse("from __future__ import annotations\n").body + nodes,
            type_ignores=[],
        )
        exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


class HGRN2Reference(nn.Module):
    def __init__(
        self,
        model_dir: Path,
        source_dir: Path,
        common_dir: Path,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        config_path = model_dir / "config.json"
        checkpoint = model_dir / "model.safetensors"
        manifest = json.loads((model_dir / "verified-manifest.json").read_text())
        assert manifest["revision"] == REVISION
        entry = next(x for x in manifest["files"] if x["file"] == checkpoint.name)
        assert entry["size"] == 2728818968 and entry["checksum"] == WEIGHT_SHA256
        with checkpoint.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == WEIGHT_SHA256
        raw = json.loads(config_path.read_text())
        assert raw["model_type"] == "hgrn2"
        assert raw["hidden_size"] == 2048 and raw["num_hidden_layers"] == 24
        assert raw["expand_ratio"] == 128 and raw["vocab_size"] == 32000
        assert raw["attn"] is None and raw["use_lower_bound"]
        assert not raw["use_short_conv"] and not raw["tie_word_embeddings"]
        config = SimpleNamespace(**dict(raw, num_heads=16, fuse_swiglu=True))
        functions = hgrn2_namespace(source_dir, common_dir)
        with torch.device("meta"):
            self.model = nn.Module()
            self.model.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.model.lower_bounds = nn.Parameter(
                torch.zeros(config.num_hidden_layers, config.hidden_size)
            )
            self.model.layers = nn.ModuleList(
                functions["HGRN2Block"](config, layer_idx=i)
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
        parameters = sum(p.numel() for p in self.parameters())
        assert self.lm_head.weight is not self.model.embeddings.weight
        self.eval()
        self.provenance = {
            "scope": "FULL_HGRN2_FLA_LAYER_AST_WITH_EXPLICIT_TORCH_RECURRENCE",
            "checkpoint": manifest,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_sha256": PINS,
            "parameters": parameters,
            "checkpoint_tensors": len(weights),
            "layers": len(self.model.layers),
            "dtype": str(dtype),
            "accelerated_native_fla_qualified": False,
        }

    def forward(
        self, input_ids: torch.Tensor, cache: RecurrentCache | None = None
    ) -> tuple[torch.Tensor, RecurrentCache]:
        if cache is None:
            cache = RecurrentCache()
        hidden = self.model.embeddings(input_ids)
        bounds = self.model.lower_bounds.softmax(0, dtype=torch.float)
        bounds = bounds.cumsum(0) - bounds[0]
        for index, layer in enumerate(self.model.layers):
            hidden, _, cache = layer(
                hidden,
                past_key_values=cache,
                use_cache=True,
                lower_bound=bounds[index],
            )
        logits = self.lm_head(self.model.norm(hidden))
        return logits, cache
