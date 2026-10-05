"""Match every checkpoint tensor against the pinned historical FLA model."""

import json
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
from gdn_torch_reference import FLA_REVISION, PINS, gdn_namespace

model_dir = root / "models/gdn-340m"
source_dir = root / "evidence/models/gdn-sources-2024-12-22"
config = SimpleNamespace(**json.loads((model_dir / "config.json").read_text()))
functions = gdn_namespace(source_dir, root / "evidence/models/deltanet-sources")
with torch.device("meta"):
    model = nn.Module()
    model.model = nn.Module()
    model.model.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    model.model.layers = nn.ModuleList(
        functions["GatedDeltaNetBlock"](config, layer_idx=index)
        for index in range(config.num_hidden_layers)
    )
    model.model.norm = functions["RMSNorm"](config.hidden_size, eps=config.norm_eps)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
with (model_dir / "model.safetensors").open("rb") as handle:
    header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
expected = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
actual = {
    name: item["shape"] for name, item in header.items() if name != "__metadata__"
}
missing = sorted(expected.keys() - actual.keys())
unexpected = sorted(actual.keys() - expected.keys())
shape_mismatch = {
    name: {"checkpoint": actual[name], "source": expected[name]}
    for name in expected.keys() & actual.keys()
    if actual[name] != expected[name]
}
result = {
    "status": "PASS" if not (missing or unexpected or shape_mismatch) else "FAILED",
    "checkpoint": "linear-moe-hub/Gated-Deltanet-340M",
    "fla_revision": FLA_REVISION,
    "source_sha256": PINS,
    "checkpoint_tensors": len(actual),
    "historical_source_tensors": len(expected),
    "missing": missing,
    "unexpected": unexpected,
    "shape_mismatch": shape_mismatch,
    "gpu_executed": False,
    "weights_loaded": False,
    "performance_claim": False,
}
out = root / "evidence/gdn-historical-architecture-preflight-r1.json"
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
raise SystemExit(0 if result["status"] == "PASS" else 1)
