"""Replay saved TiRex GPU qualification tensors with NumPy arithmetic."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tirex_slstm_reference_gate import SOURCE_REVISION, WEIGHT_SHA256

BUDGET = 1e-4
STATES = ("hidden", "cell", "normalizer", "stabilizer")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(row: dict) -> str:
    scope = row["scope"]
    batch = row["batch"]
    if scope == "cell":
        return f"cell/B{batch}/T{row['steps']}/{row['tensor']}"
    if scope == "forecast":
        return f"forecast/B{batch}/{row['tensor']}"
    if scope == "forecast_cell":
        return f"forecast_cell/B{batch}/L{row['block']}/P{row['patch']}/{row['tensor']}"
    raise ValueError(f"unknown scope: {scope}")


def required_ids() -> set[str]:
    keys = {
        f"cell/B{batch}/T{steps}/{name}"
        for batch, steps in ((1, 16), (2, 32), (4, 64))
        for name in ("output", *STATES)
    }
    keys.update(
        f"forecast/B{batch}/{name}"
        for batch in (1, 2, 4)
        for name in ("quantiles", "median")
    )
    keys.update(
        f"forecast_cell/B{batch}/L{layer}/P{patch}/{name}"
        for batch in (1, 2, 4)
        for layer in range(12)
        for patch in range(2)
        for name in ("output", *STATES)
    )
    assert len(keys) == 381
    return keys


def pairs(raw: dict):
    for entry in raw["cell"]:
        if entry["reference"] is None or entry["candidate"] is None:
            continue
        prefix = f"cell/B{entry['batch']}/T{entry['steps']}"
        reference, candidate = entry["reference"], entry["candidate"]
        yield f"{prefix}/output", reference[0], candidate[0]
        for index, name in enumerate(STATES):
            yield f"{prefix}/{name}", reference[1][index], candidate[1][index]
    for entry in raw["forecast"]:
        prefix = f"forecast/B{entry['batch']}"
        reference, candidate = entry["reference"], entry["candidate"]
        if reference["forecast"] is not None and candidate["forecast"] is not None:
            for index, name in enumerate(("quantiles", "median")):
                yield (
                    f"{prefix}/{name}",
                    reference["forecast"][index],
                    candidate["forecast"][index],
                )
        for layer, (reference_calls, candidate_calls) in enumerate(
            zip(reference["cells"], candidate["cells"])
        ):
            for patch, (reference_pair, candidate_pair) in enumerate(
                zip(reference_calls, candidate_calls)
            ):
                cell_prefix = f"forecast_cell/B{entry['batch']}/L{layer}/P{patch}"
                yield f"{cell_prefix}/output", reference_pair[0], candidate_pair[0]
                for index, name in enumerate(STATES):
                    yield (
                        f"{cell_prefix}/{name}",
                        reference_pair[1][index],
                        candidate_pair[1][index],
                    )


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference.shape != candidate.shape:
        return {
            "pass": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "failed_elements": None,
        }
    expected = reference.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    actual = candidate.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    difference = np.abs(actual - expected)
    allowed = BUDGET * (1 + np.abs(expected))
    failed = ~np.isfinite(actual) | ~np.isfinite(expected) | (difference > allowed)
    finite = np.isfinite(difference) & np.isfinite(allowed)
    return {
        "pass": not bool(np.any(failed)),
        "shape": list(expected.shape),
        "elements": int(expected.size),
        "failed_elements": int(np.count_nonzero(failed)),
        "max_abs": float(np.max(difference[finite])) if np.any(finite) else None,
        "worst_normalized": (
            float(np.max(difference[finite] / allowed[finite]))
            if np.any(finite)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    raw_path = args.gate.with_suffix(".pt")
    raw_sha = digest(raw_path)
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    required = required_ids()
    reported = {identity(row): row for row in gate["comparisons"]}
    observed = {}
    for key, reference, candidate in pairs(raw):
        if key in observed:
            raise ValueError(f"duplicate raw tensor pair: {key}")
        observed[key] = compare(reference, candidate)
    reported_complete = len(reported) == len(gate["comparisons"]) == 381
    missing = sorted(required - observed.keys())
    unexpected = sorted(observed.keys() - required)
    report_mismatch = sorted(
        key
        for key in required & observed.keys() & reported.keys()
        if observed[key]["pass"] != reported[key]["pass"]
    )
    complete = (
        gate["status"] == "PASS"
        and gate["budget"] == BUDGET
        and gate["checkpoint_sha256"] == WEIGHT_SHA256
        and gate["source_revision"] == SOURCE_REVISION
        and gate["raw_sha256"] == raw_sha
        and reported_complete
        and set(reported) == required
        and not missing
        and not unexpected
    )
    passed = (
        complete
        and not report_mismatch
        and all(row["pass"] for row in observed.values())
    )
    result = {
        "status": "PASS" if passed else "FAIL_NUMERICAL" if complete else "INCOMPLETE",
        "gate_status": gate["status"],
        "gate_sha256": digest(args.gate),
        "budget": BUDGET,
        "raw_sha256": raw_sha,
        "raw_sha_verified": gate["raw_sha256"] == raw_sha,
        "required_pairs": len(required),
        "observed_pairs": len(observed),
        "reported_pairs": len(gate["comparisons"]),
        "missing": missing,
        "unexpected": unexpected,
        "reported_independent_disagreements": report_mismatch,
        "failed_pairs": sorted(key for key, row in observed.items() if not row["pass"]),
        "comparisons": observed,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
