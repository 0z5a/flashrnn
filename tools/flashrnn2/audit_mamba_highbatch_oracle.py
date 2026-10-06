"""Audit pinned Mamba high-batch native oracles against both trace schedules."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare(actual: torch.Tensor, reference: torch.Tensor, tolerance: float) -> tuple[float, int]:
    if torch.equal(actual, reference):
        return 0.0, 0
    delta = (actual - reference).abs()
    failed = ~torch.isfinite(actual) | ~torch.isfinite(reference)
    failed |= delta > tolerance * (1 + reference.abs())
    return float(delta.max()), int(failed.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    metadata = json.loads(args.oracle.with_suffix(".meta.json").read_text())
    trace_meta = json.loads(args.trace.with_suffix(".meta.json").read_text())
    inputs = json.loads(args.inputs.read_text())
    batch = metadata["batch"]
    assert metadata["status"] == "NATIVE_CPU_ORACLES_COMPLETE"
    assert metadata["model_family"] == "mamba" and batch in (16, 32, 64)
    assert trace_meta["status"] == "PASS" and trace_meta["batch"] == batch
    assert metadata["input_manifest_sha256"] == digest(args.inputs)
    trace_input_sha256 = next(
        value
        for path, value in trace_meta["source_sha256"].items()
        if Path(path).name == args.inputs.name
    )
    assert trace_input_sha256 == digest(args.inputs)
    assert metadata["artifact_sha256"] == digest(args.oracle)
    assert trace_meta["tensor_sha256"] == digest(args.snapshot)
    implementation = next(
        value
        for path, value in trace_meta["source_sha256"].items()
        if Path(path).name == "modeling_mamba.py"
    )
    assert metadata["implementation_sha256"] == implementation
    cases = torch.load(args.oracle, map_location="cpu", weights_only=True)
    snapshots = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    assert len(cases) == 3 and len(rows) == 192
    assert set(snapshots) == {"0", "31"}
    assert all(row["pass"] for row in rows)
    group = inputs["batches"][str(batch)]
    max_abs = 0.0
    tensor_pairs = 0
    for index, case in enumerate(cases):
        official = metadata["cases"][index]
        assert official["dataset_rows"] == group["dataset_rows"][index]
        assert torch.equal(case["input_ids"], torch.tensor(group["input_ids"][index]))
        assert hashlib.sha256(case["input_ids"].numpy().tobytes()).hexdigest() == official[
            "input_ids_sha256"
        ]
        generated = case["generated_ids"]
        assert generated.shape == (32, batch)
        assert torch.equal(generated, case["logits"].argmax(-1))
        assert hashlib.sha256(generated.numpy().tobytes()).hexdigest() == official[
            "generated_ids_sha256"
        ]
        for schedule in ("serial", "interleaved"):
            selected = [
                row for row in rows if row["schedule"] == schedule and row["case"] == index
            ]
            assert [row["step"] for row in selected] == list(range(32))
            assert torch.equal(
                generated, torch.tensor([row["traced_ids"] for row in selected])
            )
            assert torch.equal(
                generated, torch.tensor([row["native_ids"] for row in selected])
            )
    for step, state in ((0, "prefill_cache"), (31, "final_cache")):
        reference = (cases[0]["logits"][step], *cases[0][state])
        for arm in ("native", "traced"):
            for position, (actual, expected) in enumerate(
                zip(snapshots[str(step)][arm], reference, strict=True)
            ):
                error, failures = compare(actual, expected, 1e-3 if position == 0 else 1e-5)
                assert failures == 0
                max_abs = max(max_abs, error)
                tensor_pairs += 1
    audit = {
        "status": "PASS",
        "batch": batch,
        "oracle_full_logit_argmax_steps": 96,
        "oracle_token_choices": 96 * batch,
        "trace_schedule_token_choices": 192 * batch,
        "full_boundary_tensor_pairs": tensor_pairs,
        "boundary_max_abs": max_abs,
        "input_sha256": digest(args.inputs),
        "oracle_sha256": digest(args.oracle),
        "trace_snapshot_sha256": digest(args.snapshot),
        "implementation_sha256": implementation,
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
