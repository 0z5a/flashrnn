"""Recompute saved complete Gated DeltaNet logits and selected final caches."""

import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent
evidence = root / "evidence"
stem = "gdn-reference-r3"
meta = json.loads((evidence / f"{stem}.meta.json").read_text())
controller = json.loads((evidence / f"{stem}-controller.json").read_text())
rows = [
    json.loads(line) for line in (evidence / f"{stem}.jsonl").read_text().splitlines()
]
assert meta["status"] in ("PASS", "NUMERICAL_FAILED")
assert controller["returncode"] == (0 if meta["status"] == "PASS" else 1)
assert meta["batch_steps"] == len(rows) == 36 and meta["token_choices"] == 84
assert meta["model"]["stored_parameters"] == 399531296
assert meta["model"]["checkpoint_tensors"] == 435
assert meta["model"]["layers"] == 24
assert meta["logits_budget"] == {"atol": 1e-3, "rtol": 1e-3}
assert meta["state_budget"] == {"atol": 1e-5, "rtol": 1e-5}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


frozen = json.loads((evidence / f"{stem}.sources/manifest.json").read_text())
for name, expected in controller["source_sha256"].items():
    assert frozen[name]["sha256"] == expected
    assert digest(evidence / f"{stem}.sources" / frozen[name]["file"]) == expected
for name, expected in meta["source_sha256"].items():
    path = (
        evidence / f"{stem}.sources" / frozen[name]["file"]
        if name in frozen
        else Path(name)
    )
    assert digest(path) == expected
for name, expected in meta["model"]["source_sha256"].items():
    assert digest(evidence / "models/gdn-sources-2024-12-22" / name) == expected
assert meta["model"]["fla_revision"] == "bcd9e79bfea85a394a023663f164587b506e0422"
manifest = meta["model"]["checkpoint"]
assert manifest["revision"] == "c83bdada453cde56932f37be71338df22ca29b7d"
for entry in manifest["files"]:
    path = root / "models/gdn-340m" / entry["file"]
    assert path.stat().st_size == entry["size"]
    if entry["file"] == "model.safetensors":
        assert digest(path) == entry["checksum"]

indexed = {(row["batch"], row["case"], row["step"]): row for row in rows}
assert len(indexed) == 36
assert set(indexed) == {
    (b, c, s) for b in (1, 2, 4) for c in range(3) for s in range(4)
}
assert len(meta["snapshots"]) == 9
assert {(x["batch"], x["case"]) for x in meta["snapshots"]} == {
    (b, c) for b in (1, 2, 4) for c in range(3)
}
tokenizer = Tokenizer.from_file(str(root / "models/gdn-340m/tokenizer.json"))
inputs = [tokenizer.encode(text).ids[:5] for text in meta["prompts"]]
assert inputs == meta["input_ids"]


def measure(actual: torch.Tensor, expected: torch.Tensor, budget: float) -> dict:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.float32
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    difference = (actual - expected).abs()
    allowed = budget + budget * expected.abs()
    failed = int(torch.count_nonzero(difference > allowed))
    return {
        "pass": failed == 0,
        "max_abs": float(difference.max()),
        "worst_normalized": float((difference / allowed).max()),
        "failed_elements": failed,
    }


torch.set_num_threads(1)
logit_pairs = token_choices = 0
state_records = []
conv_records = []
for entry in meta["snapshots"]:
    path = evidence / entry["file"]
    assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    batch, case = entry["batch"], entry["case"]
    assert (saved["batch"], saved["case"]) == (batch, case)
    assert torch.equal(
        saved["input_ids"], torch.tensor(inputs[4 * case : 4 * case + batch])
    )
    left, right = saved["cached_logits"], saved["full_prefix_logits"]
    assert left.shape == right.shape == (4, batch, 32000)
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
        assert len(cached) == len(full) == 24
        for layer, (a, b) in enumerate(zip(cached, full, strict=True)):
            assert set(a) == set(b) == {"recurrent_state", "conv_state"}
            state = measure(a["recurrent_state"], b["recurrent_state"], 1e-5)
            assert a["recurrent_state"].shape == (batch, 4, 256, 256)
            assert state == indexed[batch, case, 3]["recurrent_layers"][layer]
            state_records.append({"batch": batch, "layer": layer, **state})
            assert len(a["conv_state"]) == len(b["conv_state"]) == 3
            for channel, (x, y) in enumerate(
                zip(a["conv_state"], b["conv_state"], strict=True)
            ):
                assert x.shape == y.shape == (batch, 1024, 4)
                result = measure(x, y, 1e-5)
                assert (
                    result
                    == indexed[batch, case, 3]["convolution_layers"][
                        3 * layer + channel
                    ]
                )
                conv_records.append(
                    {"batch": batch, "layer": layer, "channel": channel, **result}
                )
    else:
        assert (
            "cached_final_states" not in saved
            and "full_prefix_final_states" not in saved
        )
    del saved, left, right
assert logit_pairs == 36 and token_choices == 84
assert len(state_records) == 72 and len(conv_records) == 216
for row in rows:
    for name, detail in (
        ("recurrent_state", "recurrent_layers"),
        ("convolution", "convolution_layers"),
    ):
        values = row[detail]
        summary = {
            "pass": all(value["pass"] for value in values),
            "max_abs": max(value["max_abs"] for value in values),
            "worst_normalized": max(value["worst_normalized"] for value in values),
            "failed_elements": sum(value["failed_elements"] for value in values),
        }
        assert summary == row["checks"][name]
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
            }
            for name in ("logits", "recurrent_state", "convolution")
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
    "final_recurrent_pairs_recomputed": len(state_records),
    "final_convolution_pairs_recomputed": len(conv_records),
    "rows": summary,
    "state_records": state_records,
    "convolution_records": conv_records,
    "performance_claim": False,
    "gpu_qualification": False,
}
output = evidence / f"{stem}-audit.json"
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {key: result[key] for key in ("audit", "model_result", "natural_exit", "rows")}
    )
)
