"""Independently recompute saved 164M mLSTM logits and final cache states."""

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

root = Path(__file__).resolve().parent
evidence = root / "evidence"
stem = "xlstm-mlstm-164m-reference-r2"
meta = json.loads((evidence / f"{stem}.meta.json").read_text())
controller = json.loads((evidence / f"{stem}-controller.json").read_text())
rows = [
    json.loads(line) for line in (evidence / f"{stem}.jsonl").read_text().splitlines()
]
assert meta["status"] in ("PASS", "NUMERICAL_FAILED")
assert controller["returncode"] == (0 if meta["status"] == "PASS" else 1)
assert meta["batch_steps"] == len(rows) == 36 and meta["token_choices"] == 84
assert meta["checkpoint_tensors"] == 183 and meta["parameters"] == 164110224
assert meta["fixed_linear_rows"] == 32 and meta["fixed_linear_count"] > 0
assert meta["logits_budget"] == {"atol": 1e-3, "rtol": 1e-3}
assert meta["state_budget"] == {"atol": 1e-4, "rtol": 1e-4}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


frozen = json.loads((evidence / f"{stem}.sources/manifest.json").read_text())
for name, digest in controller["source_sha256"].items():
    assert frozen[name]["sha256"] == digest
    assert sha256(evidence / f"{stem}.sources" / frozen[name]["file"]) == digest
for name, digest in meta["source_sha256"].items():
    path = (
        evidence / f"{stem}.sources" / frozen[name]["file"]
        if name in frozen
        else Path(name)
    )
    assert sha256(path) == digest
manifest = meta["checkpoint"]
assert manifest["revision"] == "2ce74c14add515517ae6a32d3ae80cc766f62c03"
model_dir = root / "models/xlstm-mlstm-164m"
for entry in manifest["files"]:
    path = model_dir / entry["file"]
    assert path.stat().st_size == entry["size"]
    if entry["file"] == "model_0.safetensors":
        assert sha256(path) == entry["checksum"]
assert sha256(model_dir / "config.json") == meta["config_sha256"]
for entry in meta["tokenizer"]["files"]:
    path = model_dir / entry["file"]
    assert path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]
assert meta["tokenizer"]["revision"] == "9dc507bd0939cf372a4a4f667335651d8e49dddb"
indexed = {(row["batch"], row["case"], row["step"]): row for row in rows}
assert len(indexed) == 36 and set(indexed) == {
    (batch, case, step) for batch in (1, 2, 4) for case in range(3) for step in range(4)
}
assert len(meta["snapshots"]) == 9
assert {(item["batch"], item["case"]) for item in meta["snapshots"]} == {
    (batch, case) for batch in (1, 2, 4) for case in range(3)
}
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
inputs = [
    [tokenizer.bos_token_id, *tokenizer(text, add_special_tokens=False).input_ids[:4]]
    for text in meta["prompts"]
]
assert inputs == meta["input_ids"]


def measure(actual: torch.Tensor, expected: torch.Tensor, budget: float) -> dict:
    assert (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype == torch.float32
    )
    difference = (actual - expected).abs()
    allowed = budget + budget * expected.abs()
    failed = (
        ~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)
    )
    return {
        "pass": not bool(failed.any()),
        "max_abs": float(difference.max()),
        "worst_normalized": float((difference / allowed).max()),
        "failed_elements": int(failed.sum()),
    }


state_names = ("cell", "normalizer", "stabilizer")
shapes = {"cell": (6, 64, 128), "normalizer": (6, 64), "stabilizer": (6, 1)}
torch.set_num_threads(1)
logit_pairs = 0
token_choices = 0
state_records = []
for entry in meta["snapshots"]:
    path = evidence / entry["file"]
    assert path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    batch, case = entry["batch"], entry["case"]
    assert (saved["batch"], saved["case"]) == (batch, case)
    assert torch.equal(
        saved["input_ids"], torch.tensor(inputs[4 * case : 4 * case + batch])
    )
    left, right = saved["cached_logits"], saved["full_prefix_logits"]
    assert left.shape == right.shape == (4, batch, 50304)
    assert torch.equal(saved["generated_ids"], left.argmax(-1))
    token_choices += saved["generated_ids"].numel()
    for step in range(4):
        row = indexed[batch, case, step]
        assert measure(left[step], right[step], 1e-3) == row["checks"]["logits"]
        assert (
            bool(torch.equal(left[step].argmax(-1), right[step].argmax(-1)))
            == row["token_equal"]
        )
        logit_pairs += 1
    if case == 0:
        cached, full = saved["cached_final_states"], saved["full_prefix_final_states"]
        assert set(cached) == set(full) == set(range(32))
        for layer in range(12, 32):
            assert all(not torch.count_nonzero(tensor) for tensor in cached[layer])
            assert all(not torch.count_nonzero(tensor) for tensor in full[layer])
        for layer in range(12):
            a, b = cached[layer], full[layer]
            assert len(a) == len(b) == 3
            for name, x, y in zip(state_names, a, b, strict=True):
                assert x.shape == y.shape == (batch, *shapes[name])
                result = measure(x, y, 1e-4)
                assert result == indexed[batch, case, 3]["layers"][layer][name]
                state_records.append(
                    {"batch": batch, "layer": layer, "state": name, **result}
                )
    else:
        assert (
            "cached_final_states" not in saved
            and "full_prefix_final_states" not in saved
        )
    del saved, left, right
assert logit_pairs == 36 and token_choices == 84 and len(state_records) == 108
for row in rows:
    assert len(row["layers"]) == 12
    for name in state_names:
        values = [layer[name] for layer in row["layers"]]
        assert row["checks"][name] == {
            "pass": all(item["pass"] for item in values),
            "max_abs": max(item["max_abs"] for item in values),
            "worst_normalized": max(item["worst_normalized"] for item in values),
            "failed_elements": sum(item["failed_elements"] for item in values),
        }
    assert row["pass"] == (
        row["token_equal"] and all(item["pass"] for item in row["checks"].values())
    )
assert (meta["status"] == "PASS") == all(row["pass"] for row in rows)
summary = [
    {
        "batch": batch,
        "steps": 12,
        "matching_tokens": sum(
            batch for row in rows if row["batch"] == batch and row["token_equal"]
        ),
        "checks": {
            name: {
                "passed_steps": sum(
                    row["checks"][name]["pass"] for row in rows if row["batch"] == batch
                ),
                "max_abs": max(
                    row["checks"][name]["max_abs"]
                    for row in rows
                    if row["batch"] == batch
                ),
                "worst_normalized": max(
                    row["checks"][name]["worst_normalized"]
                    for row in rows
                    if row["batch"] == batch
                ),
            }
            for name in ("logits", *state_names)
        },
    }
    for batch in (1, 2, 4)
]
result = {
    "audit": "PASS",
    "model_result": meta["status"],
    "natural_exit": controller["returncode"],
    "logit_pairs_recomputed": logit_pairs,
    "token_choices_recomputed": token_choices,
    "final_state_pairs_recomputed": len(state_records),
    "rows": summary,
    "state_records": state_records,
    "performance_claim": False,
    "gpu_qualification": False,
}
output = evidence / f"{stem}-audit.json"
if output.exists():
    assert json.loads(output.read_text()) == result
else:
    output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {key: result[key] for key in ("audit", "model_result", "natural_exit", "rows")}
    )
)
