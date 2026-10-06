"""Audit Mamba2 high-batch token rows and saved full boundary tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> dict:
    delta = (actual - expected).abs()
    failures = ~torch.isfinite(actual) | ~torch.isfinite(expected)
    failures |= delta > tolerance * (1 + expected.abs())
    return {
        "shape": list(actual.shape),
        "max_abs": float(delta.max()),
        "failed_elements": int(failures.sum()),
        "pass": not bool(failures.any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    metadata = json.loads(args.result.with_suffix(".meta.json").read_text())
    rows = [json.loads(line) for line in args.result.read_text().splitlines()]
    inputs = json.loads(args.inputs.read_text())
    batch = metadata["batch"]
    groups = metadata.get("request_groups", 3)
    assert metadata["status"] == "PASS" and metadata["device"] == "cpu"
    assert batch in (16, 32, 64)
    assert groups in (1, 2, 3)
    if "request_groups" in metadata:
        assert metadata["source_sha256"]["mamba2_highbatch_gate.py"] == digest(
            Path(__file__).with_name("mamba2_highbatch_gate.py")
        )
    assert metadata["input_sha256"] == digest(args.inputs)
    assert metadata["script_artifact_sha256"] == digest(args.script)
    assert metadata["script_trace_batch"] == 1
    assert len(inputs["batches"][str(batch)]["input_ids"]) == 3
    assert len(rows) == metadata["compared_steps"] == 64 * groups
    assert metadata["token_choices"] == 64 * groups * batch
    assert metadata["failed_steps"] == 0
    expected_order = {
        "serial": [(case, step) for case in range(groups) for step in range(32)],
        "interleaved": [(case, step) for step in range(32) for case in range(groups)],
    }
    tokens = {}
    checks = 0
    for schedule, order in expected_order.items():
        selected = [row for row in rows if row["schedule"] == schedule]
        assert [(row["case"], row["step"]) for row in selected] == order
        assert all(
            row["pass"] and row["native_ids"] == row["traced_ids"] for row in selected
        )
        assert all(len(row["native_ids"]) == batch for row in selected)
        for row in selected:
            assert len(row["checks"]) == (3 if row["step"] in (0, 31) else 1)
            assert all(check["pass"] for check in row["checks"])
            checks += len(row["checks"])
        tokens[schedule] = {
            (row["case"], row["step"]): row["native_ids"] for row in selected
        }
    assert tokens["serial"] == tokens["interleaved"]
    assert metadata["schedule_parity"] == {"native": True, "traced": True}
    assert digest(args.snapshot) == metadata["snapshot_sha256"]
    snapshots = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    assert set(snapshots) == {"0", "31"}
    saved_pairs = 0
    for step in (0, 31):
        saved = snapshots[str(step)]
        assert set(saved) == {"native", "traced"}
        check = [
            compare(actual, expected, 1e-3 if index == 0 else 1e-5)
            for index, (actual, expected) in enumerate(
                zip(saved["traced"], saved["native"], strict=True)
            )
        ]
        row = next(
            row
            for row in rows
            if row["schedule"] == "serial" and row["case"] == 0 and row["step"] == step
        )
        assert check == row["checks"] and all(item["pass"] for item in check)
        saved_pairs += len(check)
    audit = {
        "status": "PASS",
        "batch": batch,
        "request_groups": groups,
        "complete_model_steps": 64 * groups,
        "token_choices": 64 * groups * batch,
        "runner_logit_cache_checks": checks,
        "saved_boundary_tensor_pairs_recomputed": saved_pairs,
        "serial_interleaved_tokens_equal": True,
        "input_sha256": digest(args.inputs),
        "script_sha256": digest(args.script),
        "snapshot_sha256": digest(args.snapshot),
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
