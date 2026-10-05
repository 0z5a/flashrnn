"""Complete HGRN checkpoint with pinned FLA bodies and Torch recurrence."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from gla_torch_reference import RecurrentCache, native_namespace
from safetensors import safe_open
from torch import nn

PINS = {
    "hgrn.py": "0e965ba876faf7b6345c889f0b66c94abae5893956cbd9dcc2f6627ba8704323",
    "modeling_hgrn.py": "fd9d206fcd0cd342cde0cd524f128248848e9685276d61ed44d465a7f764f678",
    "naive.py": "ec9207609da6bc2e56493f73766263c734adc3730bb1255698d59c306331504d",
}
REVISION = "1adad50103ad6b9c5f79df6b3ce6c9fa2299a572"
WEIGHT_SHA256 = "480ed12b06d2ebfc604210c90ecb66e922d5825e7260ad49cd1061ef7d52a987"


def hgrn_namespace(source_dir: Path, common_dir: Path) -> dict:
    namespace = native_namespace(common_dir)
    namespace["HGRNMLP"] = namespace["GatedMLP"]
    for filename, names in (
        ("naive.py", {"naive_recurrent_hgrn"}),
        ("hgrn.py", {"HGRNAttention"}),
        ("modeling_hgrn.py", {"HGRNBlock"}),
    ):
        path = source_dir / filename
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
    recurrence = namespace["naive_recurrent_hgrn"]

    def recurrent(
        x: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert cu_seqlens is None
        return recurrence(x, g, initial_state, output_final_state)

    namespace.update(chunk_hgrn=recurrent, fused_recurrent_hgrn=recurrent)
    return namespace


class HGRNReference(nn.Module):
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
        assert raw["model_type"] == "hgrn"
        assert raw["hidden_size"] == 2048 and raw["num_hidden_layers"] == 24
        assert raw["expand_ratio"] == 1 and raw["vocab_size"] == 32000
        assert raw["attn"] is None and raw["use_lower_bound"]
        assert not raw["use_short_conv"] and not raw["tie_word_embeddings"]
        config = SimpleNamespace(**dict(raw, fuse_swiglu=True))
        functions = hgrn_namespace(source_dir, common_dir)
        with torch.device("meta"):
            self.model = nn.Module()
            self.model.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.model.lower_bounds = nn.Parameter(
                torch.zeros(config.num_hidden_layers, config.hidden_size)
            )
            self.model.layers = nn.ModuleList(
                functions["HGRNBlock"](config, layer_idx=index)
                for index in range(config.num_hidden_layers)
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
            "scope": "FULL_HGRN_FLA_LAYER_AST_WITH_PINNED_TORCH_RECURRENCE",
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
        return self.lm_head(self.model.norm(hidden)), cache
