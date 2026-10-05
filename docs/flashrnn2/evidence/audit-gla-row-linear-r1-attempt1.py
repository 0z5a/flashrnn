"""Audit retained full-model rows and cached logits without rerunning the model."""

import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent
evidence = root / "evidence"
stem = evidence / "gla-row-linear-reference-r1"
meta = json.loads(stem.with_suffix(".meta.json").read_text())
controller = json.loads((evidence / "gla-row-linear-reference-r1-controller.json").read_text())
rows = [json.loads(line) for line in stem.with_suffix(".jsonl").read_text().splitlines()]
assert controller["returncode"] in (0, 1)
assert meta["status"] in ("PASS", "NUMERICAL_FAILED")
assert meta["batch_steps"] == len(rows) == 36 and meta["token_choices"] == 84
assert meta["model"]["parameters"] == 1365514240
assert meta["model"]["checkpoint_tensors"] == 339
assert meta["model"]["layers"] == 24
assert meta["logits_budget"] == {"atol": 1e-3, "rtol": 1e-3}
assert meta["state_budget"] == {"atol": 1e-5, "rtol": 1e-5}
for name, digest in meta["source_sha256"].items():
    assert controller["source_sha256"][name] == digest
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
with stem.with_suffix(".pt").open("rb") as handle:
    tensor_sha = hashlib.file_digest(handle, "sha256").hexdigest()
assert tensor_sha == meta["tensor_sha256"]
snapshots = torch.load(stem.with_suffix(".pt"), weights_only=True, map_location="cpu")
assert len(snapshots) == 9
tokenizer = Tokenizer.from_file(str(root / "models/deltanet-1.3b/tokenizer.json"))
inputs = [tokenizer.encode(text).ids[:5] for text in meta["prompts"]]
keys = {(b, c, s) for b in (1, 2, 4) for c in range(3) for s in range(4)}
assert len({(r["batch"], r["case"], r["step"]) for r in rows}) == len(rows)
assert {(r["batch"], r["case"], r["step"]) for r in rows} == keys
assert {(s["batch"], s["case"]) for s in snapshots} == {(b, c) for b in (1, 2, 4) for c in range(3)}
for snapshot in snapshots:
    batch, case = snapshot["batch"], snapshot["case"]
    assert snapshot["logits"].shape == (4, batch, 32000)
    assert torch.isfinite(snapshot["logits"]).all()
    assert torch.equal(snapshot["input_ids"], torch.tensor(inputs[4 * case : 4 * case + batch]))
    assert torch.equal(snapshot["generated_ids"], snapshot["logits"].argmax(-1))
for row in rows:
    assert row["token_equal"]
    assert row["pass"] == all(check["pass"] for check in row["checks"].values())
    for check in row["checks"].values():
        assert check["pass"] == (check["failed_elements"] == 0)
        assert check["pass"] == (check["worst_normalized"] <= 1)
summary = []
for batch in (1, 2, 4):
    selected = [row for row in rows if row["batch"] == batch]
    summary.append({
        "batch": batch,
        "batch_steps": len(selected),
        "token_choices": len(selected) * batch,
        "checks": {
            name: {
                "passed_steps": sum(row["checks"][name]["pass"] for row in selected),
                "max_abs": max(row["checks"][name]["max_abs"] for row in selected),
                "worst_normalized": max(row["checks"][name]["worst_normalized"] for row in selected),
                "failed_elements": sum(row["checks"][name]["failed_elements"] for row in selected),
                "failed_case_steps": [[row["case"], row["step"]] for row in selected if not row["checks"][name]["pass"]],
            }
            for name in ("logits", "state")
        },
    })
arithmetic = json.loads(stem.with_suffix(".arithmetic.json").read_text())
original_snapshots = torch.load(evidence / "gla-reference-r1.pt", weights_only=True, map_location="cpu")
original_by_case = {(s["batch"], s["case"]): s for s in original_snapshots}
original_token_differences = sum(
    (s["generated_ids"] != original_by_case[(s["batch"], s["case"])]["generated_ids"]).sum().item()
    for s in snapshots
)
wrapper = root / "source/tools/flashrnn2/gla_row_linear_reference.py"
assert arithmetic["arithmetic_changed"] and not arithmetic["original_gate_replaced"]
for source_name, source_digest in arithmetic["source_sha256"].items():
    assert hashlib.sha256(Path(source_name).read_bytes()).hexdigest() == source_digest
    assert controller["source_sha256"][source_name] == source_digest
assert (meta["status"] == "PASS") == all(row["pass"] for row in rows)
assert controller["returncode"] == (0 if meta["status"] == "PASS" else 1)
result = {
    "arithmetic_control": arithmetic,
    "token_differences_vs_original_cached_run": original_token_differences,
    "scope": "INDEPENDENT_RAW_ROW_AND_CACHED_LOGIT_AUDIT_NOT_FULL_STATE_RECOMPUTATION",
    "source_hashes_match": True,
    "model_result": meta["status"],
    "natural_exit": controller["returncode"],
    "all_84_cached_tokens_match_saved_logit_argmax": True,
    "cached_vs_full_prefix_tokens_equal": "36 raw checks pass; full-prefix logits were not persisted",
    "cached_logits_finite": True,
    "snapshots_sha256": tensor_sha,
    "rows": summary,
    "failures": [row for row in rows if not row["pass"]],
    "state_tensors_persisted": False,
    "performance_claim": False,
}
output = evidence / "gla-row-linear-reference-r1-audit.json"
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"audit": "PASS", "model_result": result["model_result"], "rows": summary}))
