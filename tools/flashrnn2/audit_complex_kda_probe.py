"""Recompute KDA logit and state failures from the saved raw tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def measure(actual: torch.Tensor, expected: torch.Tensor, budget: float) -> dict:
    difference = (actual.float() - expected.float()).abs()
    allowed = budget * (1 + expected.float().abs())
    failures = (
        ~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)
    )
    return {
        "pass": not failures.any().item(),
        "max_abs": difference.max().item(),
        "failed_elements": int(failures.sum()),
        "worst_normalized": (difference / allowed).max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    result = json.loads(args.result.read_text())
    with args.snapshot.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    assert digest == result["snapshot_sha256"]
    saved = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    assert saved["input_ids"].shape == (
        result.get("batch_size", 1),
        result.get("prompt_tokens", 5),
    )
    assert saved["input_ids"].tolist() == result["input_ids"]
    budget = result["budget"]["atol"]
    assert budget == result["budget"]["rtol"]
    logits = measure(saved["cached_logits"], saved["full_logits"], budget)
    assert logits == result["logits"]
    pairs = saved["state_pairs"]
    assert len(pairs) == result["state_pairs"] == 96
    checks = [
        (name, measure(actual, expected, budget)) for name, actual, expected in pairs
    ]
    assert [name for name, _ in checks] == [
        f"layer_{layer}_{state}"
        for layer in range(24)
        for state in ("recurrent", "conv_0", "conv_1", "conv_2")
    ]
    state_pass = sum(check["pass"] for _, check in checks)
    state_max_abs = max(check["max_abs"] for _, check in checks)
    state_failed_elements = sum(check["failed_elements"] for _, check in checks)
    first_failed_states = [
        (name, check) for name, check in checks if not check["pass"]
    ][:5]
    tokens_equal = torch.equal(
        saved["cached_logits"].argmax(-1), saved["full_logits"].argmax(-1)
    )
    assert state_pass == result["state_pass"]
    assert state_max_abs == result["state_max_abs"]
    assert state_failed_elements == result["state_failed_elements"]
    assert first_failed_states == result["first_failed_states"]
    assert tokens_equal == result["tokens_equal"]
    numerical_pass = logits["pass"] and state_pass == len(checks) and tokens_equal
    assert numerical_pass == (result["status"] == "PASS")
    audit = {
        "status": "PASS",
        "numerical_status": result["status"],
        "snapshot_sha256": digest,
        "logit_failed_elements": logits["failed_elements"],
        "state_failed_elements": state_failed_elements,
        "state_pairs_recomputed": len(checks),
        "tokens_equal": tokens_equal,
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
