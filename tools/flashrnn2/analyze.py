"""Analyze complete paired blocks; missing or failed results are not speedups."""

import argparse
import json
import math
import random
from pathlib import Path
from statistics import median


def summarize(
    pairs: list[tuple[float, float]], resamples: int = 10000
) -> dict[str, float | list[float] | int]:
    if len(pairs) < 20:
        raise ValueError("at least 20 independent paired blocks are required")
    if any(not math.isfinite(value) or value <= 0 for pair in pairs for value in pair):
        raise ValueError("timings must be finite and positive")
    logs = [math.log(baseline / candidate) for baseline, candidate in pairs]
    rng = random.Random(20261005)
    means = sorted(
        math.exp(sum(rng.choices(logs, k=len(logs))) / len(logs))
        for _ in range(resamples)
    )
    speedup = math.exp(sum(logs) / len(logs))
    return {
        "blocks": len(pairs),
        "baseline_median_ms": median(a for a, _ in pairs),
        "candidate_median_ms": median(b for _, b in pairs),
        "paired_speedup": speedup,
        "throughput_gain_percent": 100 * (speedup - 1),
        "latency_reduction_percent": 100 * (1 - 1 / speedup),
        "ci95": [means[int(resamples * 0.025)], means[int(resamples * 0.975)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    signatures = {
        (
            row["case"],
            row["cell"],
            row["mode"],
            row["device_uuid"],
            row["dtype_contract_id"],
            row["baseline_sha"],
            row["candidate_sha"],
            row["session_id"],
        )
        for row in rows
    }
    if len(signatures) != 1:
        raise ValueError(
            "analyze one matched case, device, arithmetic contract, source pair and session at a time"
        )
    if any(row["status"] != "PASS" for row in rows):
        raise ValueError("failed/unaudited blocks cannot support a speed comparison")
    if len({row["block_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate paired block")
    result = summarize([(row["baseline_ms"], row["candidate_ms"]) for row in rows])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
