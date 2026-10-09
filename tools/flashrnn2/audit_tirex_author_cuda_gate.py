"""Independently audit the original TiRex CUDA cell at either batch scale."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from audit_tirex_gpu_gate import compare, identity, pairs
from tirex_slstm_reference_gate import SOURCE_REVISION, WEIGHT_SHA256

XLSTM_SOURCE_SHA256 = "dc919005b21352a0b65f075c2dc81012303ee23b78d9bc091c2b76e48a871025"
BATCH_SETS = {
    "small": ((1, 16), (2, 32), (4, 64)),
    "large": ((16, 16), (32, 32), (64, 64)),
}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def required_ids(cases):
    keys = {
        f"cell/B{batch}/T{steps}/{name}"
        for batch, steps in cases
        for name in ("output", "hidden", "cell", "normalizer", "stabilizer")
    }
    keys.update(
        f"forecast/B{batch}/{name}"
        for batch, _ in cases
        for name in ("quantiles", "median")
    )
    keys.update(
        f"forecast_cell/B{batch}/L{layer}/P{patch}/{name}"
        for batch, _ in cases
        for layer in range(12)
        for patch in range(2)
        for name in ("output", "hidden", "cell", "normalizer", "stabilizer")
    )
    assert len(keys) == 381
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output already exists")
    gate = json.loads(args.gate.read_text())
    cases = BATCH_SETS[gate["batch_set"]]
    required = required_ids(cases)
    raw_path = args.gate.with_suffix(".pt")
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    observed = {}
    for key, reference, candidate in pairs(raw):
        if key in observed:
            raise ValueError(f"duplicate tensor pair: {key}")
        observed[key] = compare(reference, candidate)
    reported = {identity(row): row for row in gate["comparisons"]}
    missing = sorted(required - observed.keys())
    unexpected = sorted(observed.keys() - required)
    disagreements = sorted(
        key
        for key in required & observed.keys() & reported.keys()
        if observed[key]["pass"] != reported[key]["pass"]
    )
    structure_complete = (
        gate["baseline"] == "original TiRex xLSTM CUDA cell"
        and gate["budget"] == 1e-4
        and gate["checkpoint_sha256"] == WEIGHT_SHA256
        and gate["source_revision"] == SOURCE_REVISION
        and gate["xlstm_source_sha256"] == XLSTM_SOURCE_SHA256
        and gate["raw_sha256"] == sha256(raw_path)
        and len(reported) == len(gate["comparisons"]) == 381
        and set(reported) == required
        and not missing
        and not unexpected
    )
    passed = (
        structure_complete
        and gate["status"] == "PASS"
        and all(row["pass"] for row in reported.values())
        and all(row["pass"] for row in observed.values())
        and not disagreements
    )
    report = {
        "status": (
            "PASS"
            if passed
            else "FAIL_NUMERICAL"
            if structure_complete and gate["status"] in ("PASS", "FAIL_NUMERICAL")
            else "INCOMPLETE"
        ),
        "gate_sha256": sha256(args.gate),
        "raw_sha256": sha256(raw_path),
        "batch_set": gate["batch_set"],
        "required_pairs": len(required),
        "observed_pairs": len(observed),
        "reported_pairs": len(gate["comparisons"]),
        "missing": missing,
        "unexpected": unexpected,
        "reported_independent_disagreements": disagreements,
        "failed_pairs": sorted(key for key, row in observed.items() if not row["pass"]),
        "comparisons": observed,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
