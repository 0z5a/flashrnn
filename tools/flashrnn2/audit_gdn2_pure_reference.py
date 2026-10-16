"""Independently recompute pure GDN-2 comparisons from raw tensor snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual - expected).abs()
    allowed = 1e-4 + 1e-4 * expected.abs()
    failed = (
        ~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)
    )
    return {
        "pass": not failed.any().item(),
        "max_abs": difference.max().item(),
        "failed_elements": int(failed.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    meta = json.loads(args.meta.read_text())
    with args.snapshot.open("rb") as handle:
        snapshot_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    assert snapshot_sha256 == meta["tensor_sha256"]
    snapshots = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    assert len(snapshots) == 36
    counts = {batch: 0 for batch in (1, 2, 4)}
    max_abs = {
        "full_vs_serial": 0.0,
        "cached_vs_full": 0.0,
        "state": 0.0,
        "full_state": 0.0,
    }
    failed_elements = {name: 0 for name in max_abs}
    state_pairs = 0
    full_state_pairs = 0
    token_choices = 0
    previous = {}
    passed = True
    for snapshot in snapshots:
        row = snapshot["row"]
        batch, case, generation = row["batch"], row["case"], row["generation"]
        assert batch in counts and 0 <= case < 3 and 0 <= generation < 4
        ids = snapshot["input_ids"]
        assert ids.shape == (batch, 5 + generation)
        if generation == 0:
            assert ids.tolist() == meta["input_ids"][4 * case : 4 * case + batch]
        else:
            earlier = previous[(batch, case)]
            assert torch.equal(
                ids,
                torch.cat((earlier["input_ids"], earlier["tokens"][:, None]), dim=1),
            )
        previous[(batch, case)] = snapshot
        full, serial, cached = (
            snapshot["full_logits"],
            snapshot["serial_logits"],
            snapshot["cached_logits"],
        )
        assert full.shape == serial.shape == cached.shape == (batch, 32000)
        for name, actual, expected in (
            ("full_vs_serial", full, serial),
            ("cached_vs_full", cached, full),
        ):
            check = compare(actual, expected)
            assert check == row[name]
            passed &= check["pass"]
            max_abs[name] = max(max_abs[name], check["max_abs"])
            failed_elements[name] += check["failed_elements"]
        checks = snapshot["state_pairs"]
        assert (
            len(checks) == (48 * batch if generation == 3 else 0) == row["state_pairs"]
        )
        states_pass = True
        for name, actual, expected in checks:
            assert name.startswith("layer_") and name.endswith(
                ("_recurrent", "_conv_0", "_conv_1", "_conv_2")
            )
            check = compare(actual, expected)
            states_pass &= check["pass"]
            max_abs["state"] = max(max_abs["state"], check["max_abs"])
            failed_elements["state"] += check["failed_elements"]
            state_pairs += 1
        assert states_pass == row["state_pass"]
        full_checks = snapshot["full_state_pairs"]
        assert (
            len(full_checks)
            == (48 if generation == 3 else 0)
            == row["full_state_pairs"]
        )
        full_states_pass = True
        for name, actual, expected in full_checks:
            assert name.startswith("layer_") and name.endswith(
                ("_recurrent", "_conv_0", "_conv_1", "_conv_2")
            )
            check = compare(actual, expected)
            full_states_pass &= check["pass"]
            max_abs["full_state"] = max(max_abs["full_state"], check["max_abs"])
            failed_elements["full_state"] += check["failed_elements"]
            full_state_pairs += 1
        assert full_states_pass == row["full_state_pass"]
        tokens = full.argmax(-1)
        tokens_equal = (
            torch.equal(tokens, snapshot["tokens"])
            and torch.equal(tokens, serial.argmax(-1))
            and torch.equal(tokens, cached.argmax(-1))
        )
        assert tokens_equal == row["tokens_equal"]
        passed &= states_pass and full_states_pass and tokens_equal and row["pass"]
        token_choices += batch
        counts[batch] += 1
    assert counts == {1: 12, 2: 12, 4: 12}
    assert state_pairs == 1008 and full_state_pairs == 432 and token_choices == 84
    assert meta["batch_steps"] == 36 and meta["final_state_pairs"] == state_pairs
    assert meta["full_state_pairs"] == full_state_pairs
    assert meta["token_choices"] == token_choices
    assert passed == (meta["status"] == "PASS")
    result = {
        "status": "PASS" if passed else "NUMERICAL_FAILED",
        "snapshot_sha256": snapshot_sha256,
        "batch_steps": counts,
        "token_choices": token_choices,
        "state_tensor_pairs": state_pairs,
        "cached_full_state_tensor_pairs": full_state_pairs,
        "max_abs": max_abs,
        "failed_elements": failed_elements,
        "source": "raw full-vocabulary and recurrent-state tensors; no reported aggregate reused",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
