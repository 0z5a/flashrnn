"""Check pinned HGRN blocks and lower-bound cache continuation."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "source/tools/flashrnn2"))
from gla_torch_reference import RecurrentCache
from hgrn_torch_reference import hgrn_namespace

source = root / "evidence/models/hgrn-sources"
common = root / "evidence/models/gla-sources"
namespace = hgrn_namespace(source, common)
config = SimpleNamespace(
    hidden_size=16,
    num_hidden_layers=2,
    attn_mode="fused_recurrent",
    expand_ratio=1,
    use_short_conv=False,
    conv_size=4,
    elementwise_affine=True,
    norm_eps=1e-6,
    attn=None,
    fuse_norm=True,
    hidden_ratio=4,
    intermediate_size=None,
    hidden_act="swish",
    fuse_swiglu=True,
)
torch.set_num_threads(1)
torch.manual_seed(20261006)
blocks = [namespace["HGRNBlock"](config, layer_idx=index).eval() for index in range(2)]
inputs = torch.randn(2, 5, 16)
bound = torch.full((16,), 0.3)


def execute(
    ids: torch.Tensor, cache: RecurrentCache, second_bound: torch.Tensor
) -> tuple[torch.Tensor, RecurrentCache]:
    hidden = ids
    for index, block in enumerate(blocks):
        hidden, _, cache = block(
            hidden,
            past_key_values=cache,
            use_cache=True,
            lower_bound=second_bound if index else torch.zeros_like(bound),
        )
    return hidden, cache


with torch.inference_mode():
    full, full_cache = execute(inputs, RecurrentCache(), bound)
    first, split_cache = execute(inputs[:, :3], RecurrentCache(), bound)
    pieces = [first]
    for index in (3, 4):
        piece, split_cache = execute(inputs[:, index : index + 1], split_cache, bound)
        pieces.append(piece)
    split = torch.cat(pieces, dim=1)
    _, unbounded_cache = execute(inputs, RecurrentCache(), torch.zeros_like(bound))
assert full.shape == (2, 5, 16) and len(full_cache) == len(split_cache) == 2
assert full_cache[1]["recurrent_state"].shape == (2, 16)
output_error = float((full - split).abs().max())
state_error = float(
    (full_cache[1]["recurrent_state"] - split_cache[1]["recurrent_state"]).abs().max()
)
bound_effect = float(
    (full_cache[1]["recurrent_state"] - unbounded_cache[1]["recurrent_state"])
    .abs()
    .max()
)
assert output_error < 1e-5 and state_error < 1e-5
assert bound_effect > 1e-6
record = {
    "status": "PASS",
    "source_sha256": {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("hgrn.py", "modeling_hgrn.py", "naive.py")
    },
    "batch": 2,
    "sequence": 5,
    "hidden": 16,
    "layers_in_scope": 2,
    "float32_full_vs_split_output_max_abs": output_error,
    "float32_full_vs_split_state_max_abs": state_error,
    "positive_lower_bound_state_effect": bound_effect,
    "complete_checkpoint_loaded": False,
    "gpu_executed": False,
    "performance_claim": False,
}
path = (
    Path(sys.argv[1])
    if len(sys.argv) == 2
    else root / "evidence/hgrn-primitives-r1.json"
)
assert path.parent == root / "evidence"
assert not path.exists()
path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
