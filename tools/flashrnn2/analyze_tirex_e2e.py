"""Emit an audited Markdown speed row for one TiRex forecast E2E case."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

from analyze import summarize
from tirex_slstm_reference_gate import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads(args.input.with_suffix(".meta.json").read_text())
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    if meta["status"] != "PASS" or len(rows) != meta["blocks"] or len(rows) < 20:
        raise ValueError("complete qualified paired timing required")
    if meta["gate_budget"] != 1e-4:
        raise ValueError("TiRex numerical gate used a different budget")
    if (
        meta["dispatch_policy"]
        != "FlashRNN2 B2; original TiRex Torch cell for other batches"
    ):
        raise ValueError("unrecognized TiRex candidate dispatch")
    if sha256(args.input) != meta["raw_sha256"]:
        raise ValueError("timing raw SHA mismatch")
    for block, row in enumerate(rows):
        order = (
            ["baseline", "candidate"] if block % 2 == 0 else ["candidate", "baseline"]
        )
        if row["block"] != block or row["order"] != order or row["status"] != "PASS":
            raise ValueError("missing block, order mismatch or failed qualification")
        for arm in order:
            sample = row[arm]
            completions = sample["completion_ms"]
            if len(completions) != meta["concurrency"] // meta["batch"]:
                raise ValueError("missing forecast completions")
            if any(not math.isfinite(t) or t <= 0 for t in completions):
                raise ValueError("invalid completion time")
            if (
                completions != sorted(completions)
                or completions[-1] != sample["elapsed_ms"]
            ):
                raise ValueError("completion timestamps do not match elapsed time")
            expected_throughput = 1000 * meta["concurrency"] / sample["elapsed_ms"]
            if not math.isclose(sample["series_per_second"], expected_throughput):
                raise ValueError("throughput does not match completed series")
    summary = summarize(
        [
            (row["baseline"]["elapsed_ms"], row["candidate"]["elapsed_ms"])
            for row in rows
        ]
    )
    summary.update(
        model=meta["model"],
        batch=meta["batch"],
        concurrency=meta["concurrency"],
        candidate_path=(
            "FlashRNN2"
            if meta["batch"] in meta["accelerated_batches"]
            else "TiRex Torch fallback"
        ),
        dispatch_policy=meta["dispatch_policy"],
        baseline_series_per_second=median(
            row["baseline"]["series_per_second"] for row in rows
        ),
        candidate_series_per_second=median(
            row["candidate"]["series_per_second"] for row in rows
        ),
        gate_sha256=meta["gate_sha256"],
        raw_sha256=meta["raw_sha256"],
        arrival_policy=meta["arrival_policy"],
    )
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    low, high = summary["ci95"]
    lines = [
        "# TiRex 35M full-checkpoint GPU forecast E2E",
        "",
        "The public 12-block model forecasts 64 time points in two patches. Throughput counts completed series/s; each batch processes queued requests from one arrival burst. The 95% interval bootstraps 20 within-process paired AB/BA blocks. Model loading, compilation, numerical qualification and warmup are excluded.",
        "",
        f"Candidate dispatch: {meta['dispatch_policy']}. Fallback rows measure dispatch overhead, not FlashRNN2 kernel speedup.",
        "",
        "| Baseline | Candidate path | Batch | Concurrent series | Baseline series/s | Candidate series/s | Paired ratio [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| Official Torch | {summary['candidate_path']} | {meta['batch']} | {meta['concurrency']} | {summary['baseline_series_per_second']:.3f} | {summary['candidate_series_per_second']:.3f} | {summary['paired_speedup']:.3f}× [{low:.3f}, {high:.3f}] |",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
