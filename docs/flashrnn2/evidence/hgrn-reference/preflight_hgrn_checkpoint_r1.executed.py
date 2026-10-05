"""Compare the complete HGRN checkpoint header with the pinned model on meta."""

import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

from uv_cache_runtime import activate

activate()
import torch
from torch import nn

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "source/tools/flashrnn2"))
from hgrn_torch_reference import (
    PINS,
    REVISION,
    WEIGHT_SHA256,
    hgrn_namespace,
)

model_dir = root / "models/hgrn-1.3b"
manifest = json.loads((model_dir / "verified-manifest.json").read_text())
assert manifest["revision"] == REVISION
weight = next(x for x in manifest["files"] if x["file"] == "model.safetensors")
assert weight["checksum"] == WEIGHT_SHA256 and weight["size"] == 2728818968
raw = json.loads((model_dir / "config.json").read_text())
config = SimpleNamespace(**dict(raw, fuse_swiglu=True))
source = root / "evidence/models/hgrn-sources"
functions = hgrn_namespace(source, root / "evidence/models/gla-sources")
with torch.device("meta"):
    model = nn.Module()
    model.model = nn.Module()
    model.model.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    model.model.lower_bounds = nn.Parameter(
        torch.zeros(config.num_hidden_layers, config.hidden_size)
    )
    model.model.layers = nn.ModuleList(
        functions["HGRNBlock"](config, layer_idx=index)
        for index in range(config.num_hidden_layers)
    )
    model.model.norm = functions["RMSNorm"](config.hidden_size, eps=config.norm_eps)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
path = model_dir / "model.safetensors"
with path.open("rb") as handle:
    size = struct.unpack("<Q", handle.read(8))[0]
    header_bytes = handle.read(size)
    header = json.loads(header_bytes)
actual = {
    name: item["shape"] for name, item in header.items() if name != "__metadata__"
}
expected = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
missing = sorted(expected.keys() - actual.keys())
unexpected = sorted(actual.keys() - expected.keys())
shape_mismatch = {
    name: {"checkpoint": actual[name], "source": expected[name]}
    for name in expected.keys() & actual.keys()
    if actual[name] != expected[name]
}
result = {
    "status": "PASS" if not (missing or unexpected or shape_mismatch) else "FAILED",
    "model": manifest["model"],
    "revision": REVISION,
    "weight_bytes": weight["size"],
    "weight_sha256_verified_by_downloader": WEIGHT_SHA256,
    "tensor_header_sha256": hashlib.sha256(header_bytes).hexdigest(),
    "checkpoint_tensors": len(actual),
    "checkpoint_elements": sum(math.prod(item) for item in actual.values()),
    "pinned_model_tensors": len(expected),
    "source_sha256": PINS,
    "missing": missing,
    "unexpected": unexpected,
    "shape_mismatch": shape_mismatch,
    "full_weight_values_loaded": False,
    "gpu_executed": False,
}
out = root / "evidence/hgrn-checkpoint-r1.json"
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
raise SystemExit(0 if result["status"] == "PASS" else 1)
