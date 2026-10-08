"""Independently replay exact TiRex B2/C32 diagnostic tensors with NumPy."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from audit_tirex_gpu_gate import compare
from tirex_slstm_reference_gate import SOURCE_REVISION, WEIGHT_SHA256

FIXTURE_SHA256 = "1b7dbd6eafc2ab94e59c2df5c613cdfe93a9d12cc9e9d620c7c8905304bc83c8"
STATES = ("hidden", "cell", "normalizer", "stabilizer")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(row):
    group = row["group"]
    if row["scope"] in ("forecast_unhooked", "forecast_hooked"):
        return f"G{group}/{row['scope']}/{row['tensor']}"
    if row["scope"] == "hook_effect":
        return f"G{group}/hook_effect/{row['arm']}/{row['tensor']}"
    if row["scope"] == "forecast_cell":
        return f"G{group}/L{row['layer']}/P{row['patch']}/{row['tensor']}"
    raise ValueError(f"unknown scope: {row['scope']}")


def required_ids():
    return {
        *(
            f"G{group}/{scope}/{name}"
            for group in range(16)
            for scope in ("forecast_unhooked", "forecast_hooked")
            for name in ("quantiles", "median")
        ),
        *(
            f"G{group}/hook_effect/{arm}/{name}"
            for group in range(16)
            for arm in ("reference", "candidate")
            for name in ("quantiles", "median")
        ),
        *(
            f"G{group}/L{layer}/P{patch}/{name}"
            for group in range(16)
            for layer in range(12)
            for patch in range(2)
            for name in ("output", *STATES)
        ),
    }


def pairs(raw):
    for group, entry in enumerate(raw["groups"]):
        assert entry["group"] == group
        unhooked, hooked = entry["unhooked"], entry["hooked"]
        for scope, outputs in (
            ("forecast_unhooked", unhooked),
            ("forecast_hooked", {arm: hooked[arm]["forecast"] for arm in hooked}),
        ):
            if outputs["reference"] is not None and outputs["candidate"] is not None:
                assert len(outputs["reference"]) == len(outputs["candidate"]) == 2
                for index, name in enumerate(("quantiles", "median")):
                    yield (
                        f"G{group}/{scope}/{name}",
                        outputs["reference"][index],
                        outputs["candidate"][index],
                    )
        for arm in ("reference", "candidate"):
            if unhooked[arm] is not None and hooked[arm]["forecast"] is not None:
                for index, name in enumerate(("quantiles", "median")):
                    yield (
                        f"G{group}/hook_effect/{arm}/{name}",
                        unhooked[arm][index],
                        hooked[arm]["forecast"][index],
                    )
        reference, candidate = hooked["reference"], hooked["candidate"]
        assert len(reference["cells"]) == len(candidate["cells"]) == 12
        for layer, (expected_calls, actual_calls) in enumerate(
            zip(reference["cells"], candidate["cells"])
        ):
            for patch, (expected, actual) in enumerate(
                zip(expected_calls, actual_calls)
            ):
                yield f"G{group}/L{layer}/P{patch}/output", expected[0], actual[0]
                for index, name in enumerate(STATES):
                    yield (
                        f"G{group}/L{layer}/P{patch}/{name}",
                        expected[1][index],
                        actual[1][index],
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.diagnostic.read_text())
    raw_path = args.diagnostic.with_suffix(".pt")
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    required = required_ids()
    assert len(required) == 2048

    reported = {identity(row): row for row in report["comparisons"]}
    observed = {}
    for key, expected, actual in pairs(raw):
        if key in observed:
            raise ValueError(f"duplicate tensor pair: {key}")
        observed[key] = compare(expected, actual)
    missing = sorted(required - observed.keys())
    unexpected = sorted(observed.keys() - required)
    disagreements = sorted(
        key
        for key in required & observed.keys() & reported.keys()
        if observed[key]["pass"] != reported[key]["pass"]
    )
    complete = (
        report["status"] in ("PASS", "FAIL_NUMERICAL")
        and report["budget"] == 1e-4
        and report["checkpoint_sha256"] == WEIGHT_SHA256
        and report["source_revision"] == SOURCE_REVISION
        and report["raw_sha256"] == sha256(raw_path)
        and report["groups"] == len(raw["groups"]) == 16
        and report["batch"] == 2
        and report["queued_series"] == 32
        and report["fixture_sha256"] == FIXTURE_SHA256
        and report["fixture_sha256"]
        == hashlib.sha256(
            torch.cat([entry["context"] for entry in raw["groups"]]).numpy().tobytes()
        ).hexdigest()
        and len(reported) == len(report["comparisons"]) == 2048
        and set(reported) == required
        and not missing
        and not unexpected
        and not disagreements
    )
    failed = sorted(key for key, row in observed.items() if not row["pass"])
    status = (
        "FAIL_NUMERICAL"
        if complete and failed and report["status"] == "FAIL_NUMERICAL"
        else "PASS"
        if complete and not failed and report["status"] == "PASS"
        else "INCOMPLETE"
    )
    args.output.write_text(
        json.dumps(
            {
                "status": status,
                "structure_complete": complete,
                "diagnostic_sha256": sha256(args.diagnostic),
                "raw_sha256": sha256(raw_path),
                "budget": 1e-4,
                "required_pairs": len(required),
                "observed_pairs": len(observed),
                "reported_pairs": len(report["comparisons"]),
                "missing": missing,
                "unexpected": unexpected,
                "reported_independent_disagreements": disagreements,
                "failed_pairs": failed,
                "comparisons": observed,
            },
            indent=2,
        )
        + "\n"
    )
    if status == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
