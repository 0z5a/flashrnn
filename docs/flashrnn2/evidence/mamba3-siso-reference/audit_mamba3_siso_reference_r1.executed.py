"""Independently recompute saved Mamba-3 SISO logits and final cache states."""

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

root = Path(__file__).resolve().parent
evidence = root / "evidence"
stem = "mamba3-siso-reference-r1"
meta = json.loads((evidence / f"{stem}.meta.json").read_text())
controller = json.loads((evidence / f"{stem}-controller.json").read_text())
rows = [
    json.loads(line) for line in (evidence / f"{stem}.jsonl").read_text().splitlines()
]
assert meta["status"] in ("PASS", "NUMERICAL_FAILED")
assert controller["returncode"] == (0 if meta["status"] == "PASS" else 1)
assert meta["batch_steps"] == len(rows) == 36 and meta["token_choices"] == 84
assert meta["model"]["checkpoint_tensors"] == 147
assert meta["model"]["effective_tied_parameters"] == 186849600
assert meta["model"]["layers"] == 12
assert meta["logits_budget"] == {"atol": 1e-3, "rtol": 1e-3}
assert meta["state_budget"] == {"atol": 1e-5, "rtol": 1e-5}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


frozen = json.loads((evidence / f"{stem}.sources/manifest.json").read_text())
for name, digest in controller["source_sha256"].items():
    assert frozen[name]["sha256"] == digest
    assert sha256(evidence / f"{stem}.sources" / frozen[name]["file"]) == digest
for name, digest in meta["source_sha256"].items():
    assert frozen[name]["sha256"] == digest
for name, digest in meta["model"]["source_sha256"].items():
    assert sha256(evidence / "models/mamba3-sources" / name) == digest
manifest = meta["model"]["checkpoint"]
assert manifest["revision"] == "6792c27c00f3bb41506db1066dcd1c51bb0f4b02"
for entry in manifest["files"]:
    path = root / "models/mamba3-siso-187m" / entry["file"]
    assert path.stat().st_size == entry["size"]
    if entry["file"] == "pytorch_model.bin":
        assert sha256(path) == entry["checksum"]
assert (
    sha256(root / "models/mamba3-siso-187m/config.json")
    == meta["model"]["config_sha256"]
)
for entry in meta["tokenizer"]["files"]:
    path = root / "models/mamba3-siso-187m" / entry["file"]
    assert path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]
assert meta["tokenizer"]["revision"] == "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
indexed = {(row["batch"], row["case"], row["step"]): row for row in rows}
assert len(indexed) == 36 and set(indexed) == {
    (b, c, s) for b in (1, 2, 4) for c in range(3) for s in range(4)
}
assert len(meta["snapshots"]) == 9
assert {(x["batch"], x["case"]) for x in meta["snapshots"]} == {
    (b, c) for b in (1, 2, 4) for c in range(3)
}
tokenizer = AutoTokenizer.from_pretrained(
    root / "models/mamba3-siso-187m", local_files_only=True
)
inputs = [tokenizer(text).input_ids[:5] for text in meta["prompts"]]
assert inputs == meta["input_ids"]


def measure(a: torch.Tensor, b: torch.Tensor, budget: float) -> dict:
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    difference = (a - b).abs()
    allowed = budget + budget * b.abs()
    failed = int(torch.count_nonzero(difference > allowed))
    return {
        "pass": failed == 0,
        "max_abs": float(difference.max()),
        "worst_normalized": float((difference / allowed).max()),
        "failed_elements": failed,
    }


state_names = ("angle", "ssm", "key", "value")
shapes = {
    "angle": (24, 32),
    "ssm": (24, 64, 128),
    "key": (1, 24, 128),
    "value": (24, 64),
}
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
    assert left.shape == right.shape == (4, batch, 128256)
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
        assert set(cached) == set(full) == set(range(12))
        for layer in range(12):
            a, b = cached[layer], full[layer]
            assert len(a) == len(b) == 4
            for name, x, y in zip(state_names, a, b, strict=True):
                assert x.shape == y.shape == (batch, *shapes[name])
                result = measure(x, y, 1e-5)
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
assert logit_pairs == 36 and token_choices == 84 and len(state_records) == 144
for row in rows:
    layers = row["layers"]
    assert len(layers) == 12
    for name in state_names:
        values = [x[name] for x in layers]
        expected = {
            "pass": all(x["pass"] for x in values),
            "max_abs": max(x["max_abs"] for x in values),
            "worst_normalized": max(x["worst_normalized"] for x in values),
            "failed_elements": sum(x["failed_elements"] for x in values),
        }
        assert expected == row["checks"][name]
    assert row["pass"] == (
        row["token_equal"] and all(x["pass"] for x in row["checks"].values())
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
    "state_snapshot_scope": "Both complete case0 final caches at B1/B2/B4; other state rows are runner evidence",
    "performance_claim": False,
    "gpu_qualification": False,
}
output = evidence / f"{stem}-audit.json"
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {k: result[k] for k in ("audit", "model_result", "natural_exit", "rows")}
    )
)
