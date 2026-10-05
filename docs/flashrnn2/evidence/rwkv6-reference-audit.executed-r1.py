"""Independently recompute saved RWKV6 logits and final cache comparisons."""

import ast
import hashlib
import json
import re
from pathlib import Path

import torch


def measurement(left: torch.Tensor, right: torch.Tensor, budget: float) -> dict:
    assert left.shape == right.shape and left.dtype == right.dtype == torch.float32
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    error = (left - right).abs()
    bound = budget * right.abs() + budget
    failures = int(torch.count_nonzero(error > bound))
    return {
        "pass": failures == 0,
        "max_abs": float(error.max()),
        "worst_normalized": float((error / bound).max()),
        "failed_elements": failures,
    }


root = Path(__file__).resolve().parent
evidence = root / "evidence"
stem = evidence / "rwkv6-reference-r1"
meta = json.loads(stem.with_suffix(".meta.json").read_text())
controller = json.loads((evidence / "rwkv6-reference-r1-controller.json").read_text())
rows = [
    json.loads(line) for line in stem.with_suffix(".jsonl").read_text().splitlines()
]
assert meta["status"] in ("PASS", "NUMERICAL_FAILED")
assert controller["returncode"] == (0 if meta["status"] == "PASS" else 1)
assert meta["batch_steps"] == len(rows) == 36 and meta["token_choices"] == 84
assert meta["model"]["parameters"] == 1599873024
assert meta["model"]["checkpoint_tensors"] == 582 and meta["model"]["layers"] == 24
assert meta["logits_budget"] == {"atol": 1e-3, "rtol": 1e-3}
assert meta["state_budget"] == {"atol": 1e-5, "rtol": 1e-5}
for path, digest in meta["source_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
for path, digest in controller["source_sha256"].items():
    assert meta["source_sha256"][path] == digest
for name, digest in meta["model"]["source_sha256"].items():
    assert (
        hashlib.sha256(
            (evidence / "models/rwkv6-sources" / name).read_bytes()
        ).hexdigest()
        == digest
    )
with stem.with_suffix(".pt").open("rb") as handle:
    tensor_sha = hashlib.file_digest(handle, "sha256").hexdigest()
assert tensor_sha == meta["tensor_sha256"]
torch.set_num_threads(1)
snapshots = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
assert len(snapshots) == 9
indexed = {(row["batch"], row["case"], row["step"]): row for row in rows}
assert len(indexed) == len(rows)
assert set(indexed) == {
    (b, c, s) for b in (1, 2, 4) for c in range(3) for s in range(4)
}
assert {(s["batch"], s["case"]) for s in snapshots} == {
    (b, c) for b in (1, 2, 4) for c in range(3)
}

# These fixed ASCII prompts have no special tokens. Reproduce the pinned
# tokenizer's word boundaries and longest byte match without executing it.
vocab = {
    ast.literal_eval(line): i
    for i, line in enumerate(
        (root / "models/rwkv6-1.6b/vocab.txt").read_text().splitlines()
    )
}
assert len(vocab) == 65530 and all(isinstance(key, bytes) for key in vocab)
inputs = []
for prompt in meta["prompts"]:
    assert prompt.isascii()
    ids = []
    for word in re.split(b"(?= )", prompt.encode().strip()):
        while word:
            length = next(n for n in range(len(word), 0, -1) if word[:n] in vocab)
            ids.append(vocab[word[:length]])
            word = word[length:]
    inputs.append(ids[:5])
assert inputs == meta["input_ids"]

state_records, token_choices = [], 0
fields = ("recurrent_state", "conv_state", "ffn_state")
for saved in snapshots:
    batch, case = saved["batch"], saved["case"]
    assert torch.equal(
        saved["input_ids"], torch.tensor(inputs[4 * case : 4 * case + batch])
    )
    left, right = saved["cached_logits"], saved["full_prefix_logits"]
    assert left.shape == right.shape == (4, batch, 65536)
    assert torch.equal(saved["generated_ids"], left.argmax(-1))
    token_choices += saved["generated_ids"].numel()
    for step in range(4):
        row = indexed[batch, case, step]
        assert measurement(left[step], right[step], 1e-3) == row["checks"]["logits"]
        assert (
            torch.equal(left[step].argmax(-1), right[step].argmax(-1))
            == row["token_equal"]
        )
    if case != 0:
        assert (
            "cached_final_states" not in saved
            and "full_prefix_final_states" not in saved
        )
        continue
    cached, full = saved["cached_final_states"], saved["full_prefix_final_states"]
    assert len(cached) == len(full) == 24
    for layer, (a, b) in enumerate(zip(cached, full, strict=True)):
        assert set(a) == set(b) == set(fields)
        for field in fields:
            expected_shape = (
                (batch, 32, 64, 64) if field == "recurrent_state" else (batch, 2048)
            )
            assert a[field].shape == b[field].shape == expected_shape
            observed = measurement(a[field], b[field], 1e-5)
            assert observed == indexed[batch, case, 3]["layers"][field][layer]
            state_records.append(
                {
                    "batch": batch,
                    "case": case,
                    "layer": layer,
                    "field": field,
                    **observed,
                }
            )
assert token_choices == 84 and len(state_records) == 216
for row in rows:
    for field in fields:
        layers = row["layers"][field]
        assert len(layers) == 24
        aggregate = {
            "pass": all(x["pass"] for x in layers),
            "max_abs": max(x["max_abs"] for x in layers),
            "worst_normalized": max(x["worst_normalized"] for x in layers),
            "failed_elements": sum(x["failed_elements"] for x in layers),
        }
        assert aggregate == row["checks"][field]
    assert row["pass"] == (
        row["token_equal"] and all(x["pass"] for x in row["checks"].values())
    )
assert (meta["status"] == "PASS") == all(row["pass"] for row in rows)
summary = []
for batch in (1, 2, 4):
    selected = [row for row in rows if row["batch"] == batch]
    summary.append(
        {
            "batch": batch,
            "steps": len(selected),
            "matching_tokens": sum(batch for row in selected if row["token_equal"]),
            "checks": {
                name: {
                    "passed_steps": sum(
                        row["checks"][name]["pass"] for row in selected
                    ),
                    "max_abs": max(row["checks"][name]["max_abs"] for row in selected),
                    "worst_normalized": max(
                        row["checks"][name]["worst_normalized"] for row in selected
                    ),
                    "failed_case_steps": [
                        [row["case"], row["step"]]
                        for row in selected
                        if not row["checks"][name]["pass"]
                    ],
                }
                for name in ("logits", *fields)
            },
        }
    )
output = evidence / "rwkv6-reference-r1-audit.json"
assert not output.exists()
result = {
    "audit": "PASS",
    "model_result": meta["status"],
    "natural_exit": controller["returncode"],
    "source_hashes_match": True,
    "tensor_sha256": tensor_sha,
    "independent_tokenizer_inputs": inputs,
    "logit_pairs_recomputed": 36,
    "token_choices_recomputed": token_choices,
    "final_state_pairs_recomputed": len(state_records),
    "state_snapshot_scope": "Both final caches, case0 only, B1/B2/B4; other state rows are runner evidence",
    "rows": summary,
    "state_records": state_records,
    "performance_claim": False,
    "GPU_qualification": False,
}
output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {key: result[key] for key in ("audit", "model_result", "natural_exit", "rows")}
    )
)
