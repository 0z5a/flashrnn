"""Audit saved BlinkDL/FLA comparisons without another model evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def measure(left: torch.Tensor, right: torch.Tensor, budget: float) -> dict:
    assert left.shape == right.shape and left.dtype == right.dtype == torch.float32
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    difference = (left - right).abs()
    bound = budget * right.abs() + budget
    failed = int(torch.count_nonzero(difference > bound))
    return {
        "pass": failed == 0,
        "max_abs": float(difference.max()),
        "worst_normalized": float((difference / bound).max()),
        "failed_elements": failed,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--stem", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
stem = root / "evidence" / args.stem
meta = json.loads(stem.with_suffix(".meta.json").read_text())
controller = json.loads(stem.with_name(stem.name + "-controller.json").read_text())
assert meta["status"] in ("PASS", "NUMERICAL_FAILED")
assert controller["returncode"] == (0 if meta["status"] == "PASS" else 1)
assert meta["model"]["mapped_checkpoint_tensors"] == 582
assert meta["model"]["mapped_parameters"] == 1599873024
for path, digest in controller["source_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
for path, digest in meta["source_sha256"].items():
    assert controller["source_sha256"][path] == digest
with stem.with_suffix(".pt").open("rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
assert digest == meta["tensor_sha256"]
torch.set_num_threads(1)
saved = torch.load(stem.with_suffix(".pt"), weights_only=True, map_location="cpu")
oracle_path = root / "evidence/rwkv6-reference-r1.pt"
with oracle_path.open("rb") as handle:
    assert (
        hashlib.file_digest(handle, "sha256").hexdigest()
        == meta["oracle_tensor_sha256"]
    )
oracles = torch.load(oracle_path, weights_only=True, map_location="cpu")
oracles = {item["case"]: item for item in oracles if item["batch"] == 1}
rows = [
    json.loads(line) for line in stem.with_suffix(".jsonl").read_text().splitlines()
]
assert len(saved) == 3 and {item["case"] for item in saved} == {0, 1, 2}
indexed = {(row["case"], row["step"]): row for row in rows}
assert len(indexed) == len(rows) == meta["batch_steps"] == 12
assert set(indexed) == {(case, step) for case in range(3) for step in range(4)}
states = []
for item in saved:
    case = item["case"]
    oracle = oracles[case]
    assert item["native_logits"].shape == item["fla_logits"].shape == (4, 65536)
    assert torch.equal(item["fla_logits"], oracle["cached_logits"][:, 0])
    assert torch.equal(item["input_ids"], oracle["input_ids"])
    assert torch.equal(item["teacher_forced_ids"], oracle["generated_ids"])
    for step in range(4):
        row = indexed[case, step]
        left, right = item["native_logits"][step], item["fla_logits"][step]
        assert measure(left, right, 1e-3) == row["logits"]
        assert int(left.argmax()) == row["native_greedy"]
        assert int(right.argmax()) == row["fla_greedy"]
        assert row["token_equal"] == (row["native_greedy"] == row["fla_greedy"])
        assert row["pass"] == (row["token_equal"] and row["logits"]["pass"])
    if case == 0:
        assert (
            len(item["native_final_states"]) == len(oracle["cached_final_states"]) == 24
        )
        for layer, (left, right) in enumerate(
            zip(item["native_final_states"], oracle["cached_final_states"], strict=True)
        ):
            for field in ("recurrent_state", "conv_state", "ffn_state"):
                states.append(
                    {
                        "layer": layer,
                        "field": field,
                        **measure(left[field], right[field], 1e-5),
                    }
                )
assert states == meta["final_state_checks"] and len(states) == 72
assert (meta["status"] == "PASS") == all(row["pass"] for row in rows + states)
result = {
    "audit": "PASS",
    "model_result": meta["status"],
    "natural_exit": controller["returncode"],
    "tensor_sha256": digest,
    "recomputed_logit_pairs": 12,
    "recomputed_state_pairs": 72,
    "matching_tokens": sum(row["token_equal"] for row in rows),
    "passing_logit_pairs": sum(row["logits"]["pass"] for row in rows),
    "max_logit_abs": max(row["logits"]["max_abs"] for row in rows),
    "worst_logit_normalized": max(row["logits"]["worst_normalized"] for row in rows),
    "state_summary": {
        field: {
            "passing_layers": sum(
                row["pass"] for row in states if row["field"] == field
            ),
            "max_abs": max(row["max_abs"] for row in states if row["field"] == field),
        }
        for field in ("recurrent_state", "conv_state", "ffn_state")
    },
    "model_provenance": meta["model"],
    "performance_claim": False,
}
output = stem.with_name(stem.name + "-audit.json")
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
