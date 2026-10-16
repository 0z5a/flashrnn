"""Recalculate synthetic full-model throughput from privacy-cleared paired blocks."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

from analyze import summarize


def analyze(path: Path, expected_sha: str, expected_blocks: int) -> dict:
    match = re.fullmatch(
        r"(?:lstm|slstm|gru|elman)-b[0-9]+-c([0-9]+)\.jsonl", path.name
    )
    if match is None:
        raise ValueError(f"unrecognized workload: {path}")
    concurrency = int(match.group(1))
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f"timing hash mismatch: {path}")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(rows) != expected_blocks or expected_blocks < 20:
        raise ValueError(f"incomplete paired blocks: {path}")
    pairs = []
    for block, row in enumerate(rows):
        order = (
            ["baseline", "candidate"] if block % 2 == 0 else ["candidate", "baseline"]
        )
        if row["block"] != block or row["order"] != order or row["status"] != "PASS":
            raise ValueError(f"paired order or status mismatch: {path}")
        baseline, candidate = row["baseline"], row["candidate"]
        for arm in (baseline, candidate):
            if any(
                not math.isfinite(arm[key]) or arm[key] <= 0
                for key in ("elapsed_ms", "responses_per_second")
            ):
                raise ValueError(f"invalid timing: {path}")
            if not math.isclose(
                arm["responses_per_second"],
                concurrency * 1000 / arm["elapsed_ms"],
                rel_tol=1e-12,
            ):
                raise ValueError(f"inconsistent throughput: {path}")
        if not math.isclose(
            row["speedup"],
            baseline["elapsed_ms"] / candidate["elapsed_ms"],
            rel_tol=1e-12,
        ):
            raise ValueError(f"inconsistent speedup: {path}")
        pairs.append((baseline["elapsed_ms"], candidate["elapsed_ms"]))
    result = summarize(pairs)
    result["baseline_median_responses_per_second"] = median(
        row["baseline"]["responses_per_second"] for row in rows
    )
    result["candidate_median_responses_per_second"] = median(
        row["candidate"]["responses_per_second"] for row in rows
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parents[1]
    results = []
    for entry in manifest["files"]:
        result = analyze(root / entry["path"], entry["sha256"], entry["blocks"])
        results.append({"path": entry["path"], **result})
    args.output.write_text(json.dumps({"cases": results}, indent=2) + "\n")
    print(f"analyzed {len(results)} complete paired cases")


if __name__ == "__main__":
    main()
