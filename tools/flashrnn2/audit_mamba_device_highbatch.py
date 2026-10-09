"""Audit high-batch Mamba token records and saved boundary tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def compare(actual: torch.Tensor, reference: torch.Tensor, tolerance: float) -> dict:
    difference = (actual.cpu() - reference.cpu()).abs()
    allowed = tolerance * (1 + reference.cpu().abs())
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    failures = ~finite | (difference > allowed)
    return {
        "shape": list(actual.shape),
        "finite": bool(finite.all()),
        "max_abs": float(difference.max()),
        "failed_elements": int(failures.sum()),
        "pass": not bool(failures.any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    metadata = json.loads(args.result.with_suffix(".meta.json").read_text())
    rows = [json.loads(line) for line in args.result.read_text().splitlines()]
    batch = metadata["batch"]
    assert batch in (16, 32, 64)
    assert metadata["status"] in ("PASS", "NUMERICAL_FAILED")
    assert metadata["device"] in ("cpu", "cuda")
    producer_sha256 = next(
        digest
        for path, digest in metadata["source_sha256"].items()
        if Path(path).name == "mamba_device_generation.py"
    )
    assert (
        producer_sha256
        == hashlib.sha256(
            Path(__file__).with_name("mamba_device_generation.py").read_bytes()
        ).hexdigest()
    )
    inputs = json.loads(args.inputs.read_text())
    input_sha256 = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    expected_input_sha256 = next(
        digest
        for path, digest in metadata["source_sha256"].items()
        if Path(path).name == args.inputs.name
    )
    assert input_sha256 == expected_input_sha256
    assert (
        inputs["dataset_sha256"]
        == "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"
    )
    assert (
        inputs["tokenizer_sha256"]
        == "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf"
    )
    assert len(inputs["batches"][str(batch)]["input_ids"]) == 3
    assert all(
        len(case) == batch and all(len(row) == 128 for row in case)
        for case in inputs["batches"][str(batch)]["input_ids"]
    )
    assert len(rows) == metadata["compared_batch_steps"] == 192
    assert metadata["token_choices"] == 192 * batch
    expected_order = {
        "serial": [(case, step) for case in range(3) for step in range(32)],
        "interleaved": [(case, step) for step in range(32) for case in range(3)],
    }
    tokens = {}
    failed_steps = 0
    for schedule, order in expected_order.items():
        selected = [row for row in rows if row["schedule"] == schedule]
        assert [(row["case"], row["step"]) for row in selected] == order
        assert all(len(row["native_ids"]) == batch for row in selected)
        for row in selected:
            equal = row["native_ids"] == row["traced_ids"]
            passed = equal and all(check["pass"] for check in row["checks"])
            assert row["tokens_equal"] == equal and row["pass"] == passed
            failed_steps += not passed
        tokens[schedule] = {
            arm: {(row["case"], row["step"]): row[f"{arm}_ids"] for row in selected}
            for arm in ("native", "traced")
        }
    parity = {
        arm: tokens["serial"][arm] == tokens["interleaved"][arm]
        for arm in ("native", "traced")
    }
    assert metadata["failed_steps"] == failed_steps
    assert metadata["serial_interleaved_tokens_equal"] == parity
    with args.snapshot.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    assert digest == metadata["tensor_sha256"]
    snapshots = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    assert set(snapshots) == {"0", "31"}
    boundary_failures = 0
    for step in (0, 31):
        saved = snapshots[str(step)]
        assert set(saved) == {"native", "traced"}
        assert len(saved["native"]) == len(saved["traced"]) == 3
        check = [
            compare(actual, reference, 1e-3 if index == 0 else 1e-5)
            for index, (actual, reference) in enumerate(
                zip(saved["traced"], saved["native"], strict=True)
            )
        ]
        row = next(
            row
            for row in rows
            if row["schedule"] == "serial" and row["case"] == 0 and row["step"] == step
        )
        assert check == row["checks"]
        boundary_failures += sum(not item["pass"] for item in check)
    numerical_pass = (
        failed_steps == 0 and all(parity.values()) and boundary_failures == 0
    )
    assert metadata["status"] == ("PASS" if numerical_pass else "NUMERICAL_FAILED")
    audit = {
        "status": "PASS" if numerical_pass else "FAIL_NUMERICAL",
        "scope": "192 token rows and six saved full boundary tensor pairs",
        "batch": batch,
        "token_comparisons": 192 * batch,
        "snapshot_sha256": digest,
        "input_sha256": input_sha256,
        "saved_tensor_pairs_recomputed": 6,
        "saved_boundary_failed_pairs": boundary_failures,
        "failed_complete_model_steps": failed_steps,
        "serial_interleaved_tokens_equal": all(parity.values()),
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))
    raise SystemExit(0 if numerical_pass else 1)


if __name__ == "__main__":
    main()
