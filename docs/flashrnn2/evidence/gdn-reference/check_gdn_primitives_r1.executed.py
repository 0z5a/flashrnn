"""Exercise pinned Gated DeltaNet layers with a synthetic split-cache control."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "source/tools/flashrnn2"))
import einops
from deltanet_torch_reference import DeltaCache
from gdn_torch_reference import gdn_namespace

torch.manual_seed(43)
torch.set_num_threads(1)
functions = gdn_namespace(
    root / "evidence/models/gdn-sources-2024-12-22",
    root / "evidence/models/deltanet-sources",
)
config = SimpleNamespace(
    hidden_size=16,
    num_heads=2,
    num_v_heads=None,
    head_dim=8,
    expand_v=1,
    conv_size=4,
    attn_mode="chunk",
    attn=None,
    use_gate=True,
    use_short_conv=True,
    allow_neg_eigval=False,
    norm_eps=1e-6,
    fuse_norm=True,
    fuse_swiglu=True,
    hidden_ratio=4,
    intermediate_size=None,
    hidden_act="swish",
    norm_first=False,
)
block = functions["GatedDeltaNetBlock"](config, layer_idx=0).eval()
inputs = torch.randn(2, 5, 16)
with torch.inference_mode():
    full_cache = DeltaCache()
    full, _, full_cache = block(inputs, past_key_values=full_cache, use_cache=True)
    split_cache = DeltaCache()
    first, _, split_cache = block(
        inputs[:, :3], past_key_values=split_cache, use_cache=True
    )
    pieces = [first]
    for index in (3, 4):
        piece, _, split_cache = block(
            inputs[:, index : index + 1],
            past_key_values=split_cache,
            use_cache=True,
        )
        pieces.append(piece)
    split = torch.cat(pieces, dim=1)
    state_error = (
        (
            full_cache.states[0]["recurrent_state"]
            - split_cache.states[0]["recurrent_state"]
        )
        .abs()
        .max()
        .item()
    )
    conv_error = max(
        (a - b).abs().max().item()
        for a, b in zip(
            full_cache.states[0]["conv_state"],
            split_cache.states[0]["conv_state"],
            strict=True,
        )
    )
    output_error = (full - split).abs().max().item()
    full_split_pass = output_error < 1e-5 and state_error < 1e-5 and conv_error < 1e-5
    q = torch.randn(2, 3, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    beta = torch.sigmoid(torch.randn(2, 3, 2))
    gate = -torch.rand(2, 3, 2)
    recurrence = functions["fused_recurrent_gated_delta_rule"]
    _, gated_state = recurrence(q, k, v, gate, beta, None, True, None, False)
    _, ungated_state = recurrence(
        q, k, v, torch.zeros_like(gate), beta, None, True, None, False
    )
    gate_effect = (gated_state - ungated_state).abs().max().item()
    gate_effect_pass = gate_effect > 1e-4
result = {
    "status": "PASS" if full_split_pass and gate_effect_pass else "NUMERICAL_FAILED",
    "batch": 2,
    "sequence": 5,
    "hidden": 16,
    "layers_in_scope": 1,
    "split_scope": "PREFILL3_THEN_TWO_SINGLE_TOKEN_DECODE_STEPS",
    "full_vs_split_output_max_abs": output_error,
    "full_vs_split_recurrent_state_max_abs": state_error,
    "full_vs_split_conv_state_max_abs": conv_error,
    "nonzero_gate_state_effect": gate_effect,
    "full_split_pass": full_split_pass,
    "gate_effect_pass": gate_effect_pass,
    "checkpoint_loaded": False,
    "gpu_executed": False,
    "performance_claim": False,
    "python_executable": sys.executable,
    "torch_version": torch.__version__,
    "einops_version": einops.__version__,
    "einops_file": einops.__file__,
    "source_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            root / "source/tools/flashrnn2/gdn_torch_reference.py",
            root / "evidence/models/gdn-sources-2024-12-22/gated_deltanet.py",
            root / "evidence/models/gdn-sources-2024-12-22/modeling_gated_deltanet.py",
        )
    },
}
out = (
    Path(sys.argv[1])
    if len(sys.argv) == 2
    else root / "evidence/gdn-primitives-r1.json"
)
assert out.parent == root / "evidence"
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
raise SystemExit(0 if result["status"] == "PASS" else 1)
