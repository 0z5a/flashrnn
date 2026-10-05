"""Factorial operator-order diagnostics on the unchanged full DeltaNet weights."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "source/tools/flashrnn2"))
import deltanet_torch_reference as delta
from gla_reference_gate import PROMPTS, compare

torch.set_num_threads(1)
prefix = root / "evidence/deltanet-arithmetic-controls-r1"
assert not prefix.with_suffix(".json").exists()
original_linear, original_convolution = F.linear, delta.convolution


def token_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    output = torch.cat([original_linear(row[None], weight, bias) for row in rows], dim=0)
    return output.reshape(*x.shape[:-1], weight.shape[0])


def sequential_convolution(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                           activation: str | None, seq_idx: None = None) -> torch.Tensor:
    assert seq_idx is None
    state = x.new_zeros(x.shape[0], x.shape[1], weight.shape[-1])
    return torch.stack([delta.convolution_step(x[:, :, t], state, weight, bias, activation)
                        for t in range(x.shape[-1])], dim=-1)


started = time.time()
source_paths = [Path(__file__)] + [root / "source/tools/flashrnn2" / name for name in (
    "gla_torch_reference.py", "gla_reference_gate.py", "deltanet_torch_reference.py"
)]
source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
model = delta.DeltaNetReference(root / "models/deltanet-1.3b", root / "evidence/models/deltanet-sources")
namespace = model.model.layers[0].attn.q_conv1d.forward.__globals__
assert namespace["causal_conv1d_fn"] is original_convolution
tokenizer = Tokenizer.from_file(str(root / "models/deltanet-1.3b/tokenizer.json"))
ids = torch.tensor([tokenizer.encode(PROMPTS[8]).ids[:5]])
rows, saved = [], {}
with torch.inference_mode(), prefix.with_suffix(".jsonl").open("x") as log:
    for name, fixed_linear, fixed_conv in (
        ("linear_only", True, False),
        ("convolution_only", False, True),
        ("linear_and_convolution", True, True),
    ):
        F.linear = token_linear if fixed_linear else original_linear
        namespace["causal_conv1d_fn"] = sequential_convolution if fixed_conv else original_convolution
        full_ids = ids.clone()
        logits, cache = model(ids)
        generated = []
        for step in range(4):
            reference, full_cache = model(full_ids)
            checks = {"logits": compare(logits[:, -1], reference[:, -1], 1e-3)}
            states = [compare(a["recurrent_state"], b["recurrent_state"], 1e-5)
                      for a, b in zip(cache.states, full_cache.states, strict=True)]
            conv = [compare(x, y, 1e-5)
                    for a, b in zip(cache.states, full_cache.states, strict=True)
                    for x, y in zip(a["conv_state"], b["conv_state"], strict=True)]
            for field, values in (("state", states), ("convolution", conv)):
                checks[field] = {"pass": all(v["pass"] for v in values),
                                 "max_abs": max(v["max_abs"] for v in values),
                                 "worst_normalized": max(v["worst_normalized"] for v in values),
                                 "failed_elements": sum(v["failed_elements"] for v in values)}
            next_ids = logits[:, -1].argmax(-1, keepdim=True)
            token_equal = torch.equal(next_ids, reference[:, -1].argmax(-1, keepdim=True))
            generated.append(next_ids[:, 0].tolist())
            row = {"variant": name, "step": step, "checks": checks, "token_equal": token_equal,
                   "pass": token_equal and all(v["pass"] for v in checks.values())}
            rows.append(row)
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
            if step == 3:
                saved[name] = {"tokens": generated, "cached_logits": logits[:, -1].clone(),
                               "full_prefix_logits": reference[:, -1].clone(),
                               "cached_states": cache.states, "full_prefix_states": full_cache.states}
            else:
                full_ids = torch.cat((full_ids, next_ids), dim=1)
                logits, cache = model(next_ids, cache)
F.linear = original_linear
namespace["causal_conv1d_fn"] = original_convolution
torch.save(saved, prefix.with_suffix(".pt"))
with prefix.with_suffix(".pt").open("rb") as handle:
    tensor_sha = hashlib.file_digest(handle, "sha256").hexdigest()
record = {"scope": "FULL_DELTANET_B1_CASE2_P5_G4_OPERATOR_ORDER_DIAGNOSTIC",
          "pid": os.getpid(), "started": started, "finished": time.time(),
          "source_sha256": source_hashes, "model": model.provenance,
          "arithmetic_changed": True, "original_failed_gate_replaced": False,
          "performance_claim": False, "snapshot_sha256": tensor_sha, "rows": rows}
prefix.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"status": "DIAGNOSTIC_COLLECTED", "rows": len(rows), "passed_rows": sum(r["pass"] for r in rows)}), flush=True)
