"""Audit Mamba2 high-batch token rows and saved full boundary tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mamba2_cohorts import cohort_sha256, select_groups

LEGACY_GATE_SHA256 = "c872deef2ad7979986a99b1ab166c377edb680d4cda8e8be8188dfdff81e4586"


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
    assert metadata["status"] in ("PASS", "FAIL")
    assert metadata["device"] in ("cpu", "cuda")
    assert batch in (1, 4, 16, 32, 64)
    assert groups in (1, 2, 3)
    if "request_groups" in metadata:
        gate_sha = metadata["source_sha256"]["mamba2_highbatch_gate.py"]
        if "selected_input_sha256" in metadata:
            assert gate_sha == digest(
                Path(__file__).with_name("mamba2_highbatch_gate.py")
            )
            assert metadata["source_sha256"]["mamba2_cohorts.py"] == digest(
                Path(select_groups.__code__.co_filename)
            )
        else:
            assert batch >= 16 and gate_sha == LEGACY_GATE_SHA256
    assert metadata["input_sha256"] == digest(args.inputs)
    assert metadata["script_artifact_sha256"] == digest(args.script)
    assert metadata["script_trace_batch"] == 1
    selected_groups = select_groups(inputs, batch)[:groups]
    if "selected_input_sha256" in metadata:
        assert metadata["input_source_batch"] == max(batch, 16)
        assert metadata["selected_input_sha256"] == cohort_sha256(selected_groups)
    assert len(rows) == metadata["compared_steps"] == 64 * groups
    assert metadata["token_choices"] == 64 * groups * batch
    expected_order = {
        "serial": [(case, step) for case in range(groups) for step in range(32)],
        "interleaved": [(case, step) for step in range(32) for case in range(groups)],
    }
    tokens = {}
    checks = failed_steps = 0
    for schedule, order in expected_order.items():
        schedule_rows = [row for row in rows if row["schedule"] == schedule]
        assert [(row["case"], row["step"]) for row in schedule_rows] == order
        assert all(len(row["native_ids"]) == batch for row in schedule_rows)
        for row in schedule_rows:
            assert len(row["checks"]) == (3 if row["step"] in (0, 31) else 1)
            passed = row["native_ids"] == row["traced_ids"] and all(
                check["pass"] for check in row["checks"]
            )
            assert row["pass"] == passed
            failed_steps += not passed
            checks += len(row["checks"])
        tokens[schedule] = {
            arm: {(row["case"], row["step"]): row[f"{arm}_ids"] for row in schedule_rows}
            for arm in ("native", "traced")
        }
    parity = {
        arm: tokens["serial"][arm] == tokens["interleaved"][arm]
        for arm in ("native", "traced")
    }
    assert metadata["failed_steps"] == failed_steps
    assert metadata["schedule_parity"] == parity
    assert digest(args.snapshot) == metadata["snapshot_sha256"]
    snapshots = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    assert set(snapshots) == {"0", "31"}
    saved_pairs = boundary_failures = 0
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
        assert check == row["checks"]
        boundary_failures += sum(not item["pass"] for item in check)
        saved_pairs += len(check)
    numerical_pass = (
        failed_steps == 0 and all(parity.values()) and boundary_failures == 0
    )
    assert metadata["status"] == ("PASS" if numerical_pass else "FAIL")
    audit = {
        "status": "PASS" if numerical_pass else "FAIL_NUMERICAL",
        "batch": batch,
        "request_groups": groups,
        "complete_model_steps": 64 * groups,
        "token_choices": 64 * groups * batch,
        "runner_logit_cache_checks": checks,
        "saved_boundary_tensor_pairs_recomputed": saved_pairs,
        "saved_boundary_failed_pairs": boundary_failures,
        "failed_complete_model_steps": failed_steps,
        "serial_interleaved_tokens_equal": all(parity.values()),
        "input_sha256": digest(args.inputs),
        "selected_input_sha256": cohort_sha256(selected_groups),
        "script_sha256": digest(args.script),
        "gate_rows_sha256": digest(args.result),
        "gate_metadata_sha256": digest(args.result.with_suffix(".meta.json")),
        "snapshot_sha256": digest(args.snapshot),
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))
    raise SystemExit(0 if numerical_pass else 1)


if __name__ == "__main__":
    main()
