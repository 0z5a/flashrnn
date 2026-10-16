"""Replay complete author-CUDA TiRex forecast outputs with NumPy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from audit_tirex_gpu_gate import compare
from tirex_slstm_reference_gate import SOURCE_REVISION, WEIGHT_SHA256

XLSTM_SOURCE_SHA256 = "dc919005b21352a0b65f075c2dc81012303ee23b78d9bc091c2b76e48a871025"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def expected_context(index, batch):
    t = np.arange(128, dtype=np.float64)
    return np.stack(
        [
            (1 + 0.03 * series) * np.sin(t / (7 + series % 5) + series / 3)
            + 0.004 * (series + 1) * t
            + 0.2 * np.cos(t / (17 + series % 4))
            for series in range(index * batch, (index + 1) * batch)
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=Path)
    parser.add_argument("--qualification-gate", type=Path, required=True)
    parser.add_argument("--qualification-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output already exists")
    gate = json.loads(args.gate.read_text())
    qualification = json.loads(args.qualification_gate.read_text())
    qualification_audit = json.loads(args.qualification_audit.read_text())
    raw_path = args.gate.with_suffix(".pt")
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    batch = gate["batch"]
    concurrency = gate["concurrency"]
    groups = concurrency // batch
    required = {
        (group, name) for group in range(groups) for name in ("quantiles", "median")
    }
    observed = {}
    contexts = []
    fixture_pass = True
    for index, entry in enumerate(raw["groups"]):
        if entry["group"] != index:
            raise ValueError("missing or reordered forecast group")
        context = entry["context"].numpy().astype(np.float64)
        contexts.append(entry["context"])
        target = expected_context(index, batch)
        fixture_pass &= context.shape == target.shape and bool(
            np.all(
                np.isfinite(context)
                & (np.abs(context - target) <= 1e-5 * (1 + np.abs(target)))
            )
        )
        if entry["reference"] is None or entry["candidate"] is None:
            continue
        for position, name in enumerate(("quantiles", "median")):
            key = (index, name)
            if key in observed:
                raise ValueError("duplicate forecast tensor pair")
            reference = entry["reference"][position]
            candidate = entry["candidate"][position]
            shape = (batch, 64, 9) if name == "quantiles" else (batch, 64)
            observed[key] = (
                compare(reference, candidate)
                if reference.shape == candidate.shape == shape
                else {
                    "pass": False,
                    "reference_shape": list(reference.shape),
                    "candidate_shape": list(candidate.shape),
                }
            )
    raw_fixture_sha = (
        hashlib.sha256(torch.cat(contexts).numpy().tobytes()).hexdigest()
        if contexts
        else None
    )
    reported = {(row["group"], row["tensor"]): row for row in gate["comparisons"]}
    missing = sorted(required - observed.keys())
    unexpected = sorted(observed.keys() - required)
    disagreements = sorted(
        key
        for key in required & observed.keys() & reported.keys()
        if observed[key]["pass"] != reported[key]["pass"]
    )
    structure_complete = (
        batch in (1, 2, 4, 16, 32, 64)
        and concurrency in (1, 8, 32, 64, 128)
        and concurrency >= batch
        and concurrency % batch == 0
        and gate["groups"] == groups == len(raw["groups"])
        and gate["budget"] == 1e-4
        and gate["baseline"] == "original TiRex xLSTM CUDA cell"
        and gate["checkpoint_sha256"] == WEIGHT_SHA256
        and gate["source_revision"] == SOURCE_REVISION
        and gate["xlstm_source_sha256"] == XLSTM_SOURCE_SHA256
        and qualification["status"] == qualification_audit["status"] == "PASS"
        and qualification["baseline"] == "original TiRex xLSTM CUDA cell"
        and qualification["checkpoint_sha256"] == WEIGHT_SHA256
        and qualification["source_revision"] == SOURCE_REVISION
        and qualification["xlstm_source_sha256"] == XLSTM_SOURCE_SHA256
        and qualification["budget"] == 1e-4
        and qualification_audit["required_pairs"] == 381
        and qualification_audit["observed_pairs"] == 381
        and qualification_audit["raw_sha256"] == qualification["raw_sha256"]
        and qualification_audit["gate_sha256"] == sha256(args.qualification_gate)
        and gate["qualification_gate_sha256"] == sha256(args.qualification_gate)
        and gate["qualification_audit_sha256"] == sha256(args.qualification_audit)
        and gate["raw_sha256"] == sha256(raw_path)
        and gate["fixture_sha256"] == raw_fixture_sha
        and fixture_pass
        and len(reported) == len(gate["comparisons"]) == len(required)
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
        "batch": batch,
        "concurrency": concurrency,
        "required_pairs": len(required),
        "observed_pairs": len(observed),
        "missing": [list(key) for key in missing],
        "unexpected": [list(key) for key in unexpected],
        "reported_independent_disagreements": [list(key) for key in disagreements],
        "failed_pairs": [list(key) for key, row in observed.items() if not row["pass"]],
        "fixture_pass": fixture_pass,
        "comparisons": {
            f"G{group}/{name}": row for (group, name), row in observed.items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
