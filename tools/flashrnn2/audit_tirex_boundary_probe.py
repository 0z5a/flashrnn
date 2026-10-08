"""Independently audit the six TiRex recurrence boundary probes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from audit_tirex_gpu_gate import compare, digest
from tirex_slstm_reference_gate import SOURCE_REVISION, WEIGHT_SHA256

DIAGNOSTIC_SHA256 = "0de5167a9a6f3ee89a468370e2d0f40886957195dc75c8145786d19fa6fd64aa"
FIXTURE_SHA256 = "1b7dbd6eafc2ab94e59c2df5c613cdfe93a9d12cc9e9d620c7c8905304bc83c8"
STATE_NAMES = ("hidden", "cell", "normalizer", "stabilizer")
TARGETS = (
    (1, 3, 1, 61),
    (4, 3, 1, 59),
    (6, 10, 1, 61),
    (7, 9, 1, 61),
    (8, 9, 0, 61),
    (11, 3, 0, 61),
)


def contexts(concurrency: int, batch: int) -> list[torch.Tensor]:
    t = torch.arange(128, dtype=torch.float32)
    series = torch.stack(
        [
            (1 + 0.03 * index) * torch.sin(t / (7 + index % 5) + index / 3)
            + 0.004 * (index + 1) * t
            + 0.2 * torch.cos(t / (17 + index % 4))
            for index in range(concurrency)
        ]
    )
    return list(series.split(batch))


def pairs(raw):
    for entry in raw["targets"]:
        group = entry["group"]
        forecast = entry["forecast"]
        for index, name in enumerate(("quantiles", "median")):
            yield (
                f"G{group}/forecast/{name}",
                forecast["reference"][index],
                forecast["candidate"][index],
            )
        for scope in ("prefix", "isolated_step"):
            reference = entry[scope]["reference"]
            candidate = entry[scope]["candidate"]
            yield f"G{group}/{scope}/output", reference[0], candidate[0]
            for index, name in enumerate(STATE_NAMES):
                yield (
                    f"G{group}/{scope}/{name}",
                    reference[1][index],
                    candidate[1][index],
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", type=Path)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.probe.read_text())
    prior = json.loads(args.prior.read_text())
    raw_path = args.probe.with_suffix(".pt")
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    required = {
        f"G{group}/{scope}/{name}"
        for group, _, _, _ in TARGETS
        for scope, names in (
            ("forecast", ("quantiles", "median")),
            ("prefix", ("output", *STATE_NAMES)),
            ("isolated_step", ("output", *STATE_NAMES)),
        )
        for name in names
    }
    assert len(required) == 72

    reported = {
        f"G{row['group']}/{row['scope']}/{row['tensor']}": row
        for row in report["comparisons"]
    }
    observed = {
        key: compare(reference, candidate) for key, reference, candidate in pairs(raw)
    }
    boundary = {}
    fixtures = contexts(32, 2)
    fixture_sha = hashlib.sha256(torch.cat(fixtures).numpy().tobytes()).hexdigest()
    for entry in raw["targets"]:
        group = entry["group"]
        reference_gates, reference_state = entry["boundary"]["reference"]
        candidate_gates, candidate_state = entry["boundary"]["candidate"]
        state_equal = reference_state is None and candidate_state is None
        if reference_state is not None and candidate_state is not None:
            state_equal = torch.equal(reference_state, candidate_state)
        boundary[group] = {
            "context_equal": torch.equal(entry["context"], fixtures[group]),
            "gates_equal": torch.equal(reference_gates, candidate_gates),
            "state_equal": state_equal,
        }

    missing = sorted(required - observed.keys())
    unexpected = sorted(observed.keys() - required)
    disagreements = sorted(
        key
        for key in required & observed.keys() & reported.keys()
        if observed[key]["pass"] != reported[key]["pass"]
    )
    prior_flags = {
        (row["group"], row["tensor"]): row["pass"]
        for row in prior["comparisons"]
        if row["scope"] == "forecast_unhooked"
    }
    prior_reproduction_changes = sorted(
        key
        for key in required & observed.keys()
        if "/forecast/" in key
        and observed[key]["pass"]
        != prior_flags[int(key.split("/")[0][1:]), key.split("/")[-1]]
    )
    prior_flag_disagreements = sorted(
        key
        for key in required & reported.keys()
        if "/forecast/" in key
        and reported[key]["matches_prior_unhooked_flag"]
        != (key not in prior_reproduction_changes)
    )
    complete = (
        report["status"] == "DIAGNOSTIC_COMPLETE"
        and report["budget"] == 1e-4
        and report["fixture_sha256"] == FIXTURE_SHA256
        and fixture_sha == FIXTURE_SHA256
        and report["checkpoint_sha256"] == WEIGHT_SHA256
        and report["source_revision"] == SOURCE_REVISION
        and report["previous_diagnostic_sha256"]
        == digest(args.prior)
        == DIAGNOSTIC_SHA256
        and report["raw_sha256"] == digest(raw_path)
        and len(raw["targets"]) == len(report["targets"]) == len(TARGETS)
        and [
            tuple(entry[k] for k in ("group", "layer", "patch", "step"))
            for entry in raw["targets"]
        ]
        == list(TARGETS)
        and [
            tuple(entry[k] for k in ("group", "layer", "patch", "step"))
            for entry in report["targets"]
        ]
        == list(TARGETS)
        and len(reported) == len(report["comparisons"]) == 72
        and set(reported) == required
        and not missing
        and not unexpected
        and not disagreements
        and not prior_flag_disagreements
        and all(row["context_equal"] for row in boundary.values())
        and all(
            report_entry["boundary_equal"]
            == {
                "gates": boundary[report_entry["group"]]["gates_equal"],
                "state": boundary[report_entry["group"]]["state_equal"],
            }
            for report_entry in report["targets"]
        )
    )
    args.output.write_text(
        json.dumps(
            {
                "status": "COMPLETE" if complete else "INCOMPLETE",
                "probe_sha256": digest(args.probe),
                "raw_sha256": digest(raw_path),
                "required_pairs": len(required),
                "observed_pairs": len(observed),
                "missing": missing,
                "unexpected": unexpected,
                "reported_independent_disagreements": disagreements,
                "prior_unhooked_reproduction_changes": prior_reproduction_changes,
                "prior_flag_disagreements": prior_flag_disagreements,
                "boundary": boundary,
                "comparisons": observed,
            },
            indent=2,
        )
        + "\n"
    )
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
