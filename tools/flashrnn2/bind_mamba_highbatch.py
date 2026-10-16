"""Bind an audited full-model Mamba gate to its serving inputs and weights."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()

    gate = json.loads(args.result.with_suffix(".meta.json").read_text())
    audit = json.loads(args.audit.read_text())
    model = json.loads(args.model.with_suffix(".meta.json").read_text())
    oracles = json.loads(args.goldens.with_suffix(".meta.json").read_text())
    rows = [json.loads(line) for line in args.result.read_text().splitlines()]
    batch = gate["batch"]
    assert gate["status"] == audit["status"] == "PASS"
    assert gate["device"] in ("cpu", "cuda")
    assert batch == audit["batch"] == oracles["batch"] and batch in (16, 32, 64)
    assert len(rows) == gate["compared_batch_steps"] == 192
    assert all(
        row["pass"] and row["tokens_equal"] and row["native_ids"] == row["traced_ids"]
        for row in rows
    )
    assert gate["failed_steps"] == audit["failed_complete_model_steps"] == 0
    assert all(gate["serial_interleaved_tokens_equal"].values())
    assert audit["serial_interleaved_tokens_equal"]
    assert audit["saved_boundary_failed_pairs"] == 0
    assert audit["saved_tensor_pairs_recomputed"] == 6
    assert audit["token_comparisons"] == 192 * batch
    assert audit["snapshot_sha256"] == gate["tensor_sha256"]
    assert digest(args.result.with_suffix(".pt")) == gate["tensor_sha256"]
    assert gate["parameters"] == model["parameters"] == oracles["parameters"]
    assert gate["weight_transport"]["artifact_sha256"] == model["artifact_sha256"]
    assert digest(args.model) == model["artifact_sha256"]
    assert digest(args.goldens) == oracles["artifact_sha256"]

    inputs_sha = digest(args.inputs)
    assert inputs_sha == audit["input_sha256"] == oracles["input_manifest_sha256"]
    producer = Path(__file__).with_name("mamba_device_generation.py")
    runtime = Path(__file__).with_name("mamba_native_runtime.py")
    sources = (producer, runtime, args.source, args.config, args.inputs)
    reported = {
        Path(path).name: checksum for path, checksum in gate["source_sha256"].items()
    }
    assert len(reported) == len(gate["source_sha256"])
    for source in sources:
        assert digest(source) == reported[source.name]

    qualification = {
        "status": "PASS",
        "backend": "native",
        "device": gate["device"],
        "batch": batch,
        "model": model,
        "oracles": oracles,
        "native_source_sha256": {str(path): digest(path) for path in sources},
        "gate_result_sha256": digest(args.result),
        "gate_metadata_sha256": digest(args.result.with_suffix(".meta.json")),
        "independent_audit_sha256": digest(args.audit),
        "saved_boundary_sha256": gate["tensor_sha256"],
        "input_manifest_sha256": inputs_sha,
        "scope": "Audited 192-step native/trace gate and pinned full-model CPU oracles",
    }
    args.output.write_text(json.dumps(qualification, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "batch": batch, "device": gate["device"]}))


if __name__ == "__main__":
    main()
