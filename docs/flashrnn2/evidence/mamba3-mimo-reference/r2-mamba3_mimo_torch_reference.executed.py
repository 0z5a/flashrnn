"""Complete Mamba-3 MIMO checkpoint with pinned bodies and Torch recurrence."""

import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from mamba3_torch_reference import (
    SOURCE_SHA256,
    Mamba3Reference,
    RMSNormTorch,
    native_namespace,
)
from torch import nn

REVISION = "8fd6e9eb7b795f2e15d7f6353251d0137980c43e"
WEIGHT_SHA256 = "369db3fb9deedfc98baa95ac9166baf862dd7e03b3143afaf0d19e46a220daf6"


class Mamba3MIMOReference(Mamba3Reference):
    def __init__(self, model_dir: Path, source_dir: Path):
        nn.Module.__init__(self)
        manifest = json.loads((model_dir / "verified-manifest.json").read_text())
        assert manifest["revision"] == REVISION
        weight_path = model_dir / "pytorch_model.bin"
        entry = next(x for x in manifest["files"] if x["file"] == weight_path.name)
        assert entry["size"] == 374640827 and entry["checksum"] == WEIGHT_SHA256
        with weight_path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == WEIGHT_SHA256
        config = json.loads((model_dir / "config.json").read_text())
        assert (config["d_model"], config["n_layer"], config["vocab_size"]) == (
            768,
            12,
            128256,
        )
        assert config["ssm_cfg"]["layer"] == "Mamba3" and config["ssm_cfg"]["is_mimo"]
        assert (
            config["ssm_cfg"]["mimo_rank"] == 4
            and config["ssm_cfg"]["chunk_size"] == 16
        )
        assert config["d_intermediate"] == 1264 and not config["attn_layer_idx"]
        assert (
            config["rms_norm"]
            and not config["fused_add_norm"]
            and config["tie_embeddings"]
        )
        functions = native_namespace(source_dir)
        native = functions["Mamba3"]

        class Mamba3CPU(native):
            def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
                if inference_params is not None and inference_params.seqlen_offset > 0:
                    assert u.shape[1] == 1 and cu_seqlens is None
                    states = self._get_states_from_cache(inference_params, u.shape[0])
                    out, *_ = self.step(u[:, 0], *states)
                    return out[:, None]
                return super().forward(u, seq_idx, cu_seqlens, inference_params)

        with torch.device("meta"):
            self.backbone = nn.Module()
            self.backbone.embedding = nn.Embedding(
                config["vocab_size"], config["d_model"]
            )
            self.backbone.layers = nn.ModuleList()
            for index in range(config["n_layer"]):
                mixer = partial(
                    Mamba3CPU,
                    layer_idx=index,
                    **{
                        key: value
                        for key, value in config["ssm_cfg"].items()
                        if key != "layer"
                    },
                )
                mlp = partial(
                    functions["GatedMLP"],
                    hidden_features=config["d_intermediate"],
                    out_features=config["d_model"],
                )
                self.backbone.layers.append(
                    functions["Block"](
                        config["d_model"],
                        mixer,
                        mlp,
                        norm_cls=RMSNormTorch,
                        fused_add_norm=False,
                        residual_in_fp32=True,
                    )
                )
            self.backbone.norm_f = RMSNormTorch(config["d_model"])
            self.lm_head = nn.Linear(
                config["d_model"], config["vocab_size"], bias=False
            )
        state = torch.load(
            weight_path, map_location="cpu", weights_only=True, mmap=True
        )
        assert len(state) == 183 and set(state) == set(self.state_dict())
        assert torch.equal(state["lm_head.weight"], state["backbone.embedding.weight"])
        self.load_state_dict(
            {name: value.float() for name, value in state.items()},
            strict=True,
            assign=True,
        )
        self.lm_head.weight = self.backbone.embedding.weight
        assert sum(p.numel() for p in self.parameters()) == 187291968
        self.eval()
        self.provenance = {
            "checkpoint": manifest,
            "config_sha256": hashlib.sha256(
                (model_dir / "config.json").read_bytes()
            ).hexdigest(),
            "source_sha256": SOURCE_SHA256,
            "checkpoint_tensors": len(state),
            "effective_tied_parameters": 187291968,
            "layers": len(self.backbone.layers),
            "dtype": "torch.float32",
            "native_cuda_qualified": False,
        }
