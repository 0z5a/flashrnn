"""Compare matched full-model cohorts; correctness-only runs are ineligible."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from analyze import summarize


def check_oracles(
    row: dict, groups: int, steps: int, cache_checks: int, cache_names: list[str] | None
) -> None:
    assert len(row["logit_checks"]) == groups * steps
    assert {(check["group"], check["step"]) for check in row["logit_checks"]} == {
        (group, step) for group in range(groups) for step in range(steps)
    }
    assert all(len(check["errors"]) == 1 for check in row["logit_checks"])
    assert all(
        error["pass"] for check in row["logit_checks"] for error in check["errors"]
    )
    assert len(row["cache_checks"]) == groups
    assert {check["group"] for check in row["cache_checks"]} == set(range(groups))
    assert all(len(check["errors"]) == cache_checks for check in row["cache_checks"])
    if cache_names is not None:
        assert all(
            [error["tensor"] for error in check["errors"]] == cache_names
            for check in row["cache_checks"]
        )
    assert all(
        error["pass"] for check in row["cache_checks"] for error in check["errors"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.input.with_suffix(".meta.json").read_text())
    if metadata["oracles"].get("measurement_oracle_checks", False) and (
        "rows_sha256" not in metadata
    ):
        raise ValueError("timed full-output audit requires a completed raw SHA")
    if "rows_sha256" in metadata:
        with args.input.open("rb") as handle:
            assert (
                hashlib.file_digest(handle, "sha256").hexdigest()
                == metadata["rows_sha256"]
            )
    if metadata["verification_only"]:
        raise ValueError("verification-only runs cannot support a speed comparison")
    assert metadata["status"] == "PASS" and metadata["device"] == "cuda"
    baseline_name = metadata.get("baseline_name", "torchscript")
    candidate_name = metadata.get("candidate_name", "decode_graph")
    arm_order = [baseline_name, candidate_name]
    assert baseline_name != candidate_name
    assert metadata.get("arm_order", arm_order) == arm_order
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    assert all(row["status"] == "PASS" and row["token_match"] for row in rows)
    groups = defaultdict(dict)
    warmups = defaultdict(dict)
    for row in rows:
        if row["phase"] == "qualification_and_warmup":
            key = row["batch"], row["concurrency"]
            assert row["arm"] not in warmups[key]
            warmups[key][row["arm"]] = row
            continue
        if row["phase"] != "measurement":
            raise ValueError(f"unexpected serving phase: {row['phase']}")
        key = row["batch"], row["concurrency"]
        block = groups[key].setdefault(row["block"], {})
        assert row["arm"] not in block
        block[row["arm"]] = row
    assert groups, "no measured paired cohorts"
    expected = {(metadata["oracles"]["batch"], c) for c in metadata["concurrency"]}
    assert set(groups) == set(warmups) == expected
    assert metadata["paired_blocks"] >= 20
    results = []
    for (batch, concurrency), blocks in sorted(groups.items()):
        assert set(warmups[batch, concurrency]) == set(arm_order)
        steps = metadata["oracles"]["generated_tokens"]
        cache_checks = metadata["oracles"].get("cache_checks", 2)
        prefill_calls = metadata["oracles"].get("prefill_model_calls", 1)
        cache_names = metadata["oracles"].get("cache_check_names")
        measurement_oracles = metadata["oracles"].get(
            "measurement_oracle_checks", False
        )
        qualification_cache_step = metadata["oracles"].get(
            "qualification_cache_check_step", steps - 1
        )
        measurement_cache_step = metadata["oracles"].get(
            "measurement_cache_check_step", steps - 2
        )
        assert cache_checks >= 1 and prefill_calls >= 1
        assert cache_names is None or len(cache_names) == cache_checks
        group_count = concurrency // batch
        for row in warmups[batch, concurrency].values():
            assert row["request_count"] == row["peak_active_requests"] == concurrency
            check_oracles(row, group_count, steps, cache_checks, cache_names)
            if measurement_oracles:
                assert row["cache_check_step"] == qualification_cache_step
        assert set(blocks) == set(range(metadata["paired_blocks"]))
        pairs = []
        for index, block in sorted(blocks.items()):
            assert set(block) == set(arm_order)
            baseline, candidate = block[baseline_name], block[candidate_name]
            order = arm_order if index % 2 == 0 else list(reversed(arm_order))
            assert baseline["order"] == candidate["order"] == order
            left, right = baseline["requests"], candidate["requests"]
            assert len(left) == len(right) == concurrency
            assert len({r["request_id"] for r in left}) == concurrency
            assert [r["request_id"] for r in left] == [r["request_id"] for r in right]
            assert [r["token_ids"] for r in left] == [r["token_ids"] for r in right]
            assert baseline["model_batches"] == candidate["model_batches"]
            assert all(len(r["token_ids"]) == steps for r in left)
            assert baseline["model_batches"] == {
                "prefill": [batch] * (concurrency // batch * prefill_calls),
                "decode": [batch] * (concurrency // batch * (steps - 1)),
            }
            assert (
                baseline["output_tokens"]
                == candidate["output_tokens"]
                == concurrency * steps
            )
            for row in (baseline, candidate):
                assert row["batch"] == batch and row["concurrency"] == concurrency
                assert (
                    row["request_count"] == row["peak_active_requests"] == concurrency
                )
                assert row["admission_batches"] == [batch] * (concurrency // batch)
                assert math.isfinite(row["elapsed_ms"]) and row["elapsed_ms"] > 0
                assert math.isclose(
                    row["tokens_per_second"],
                    concurrency * steps * 1000 / row["elapsed_ms"],
                    rel_tol=1e-12,
                )
                assert math.isclose(
                    row["requests_per_second"],
                    concurrency * 1000 / row["elapsed_ms"],
                    rel_tol=1e-12,
                )
                if measurement_oracles:
                    check_oracles(row, group_count, steps, cache_checks, cache_names)
                    assert row["cache_check_step"] == measurement_cache_step
            pairs.append((baseline["elapsed_ms"], candidate["elapsed_ms"]))
        summary = summarize(pairs)
        summary.update(
            batch=batch,
            concurrency=concurrency,
            baseline=baseline_name,
            candidate=candidate_name,
        )
        for arm in arm_order:
            arm_rows = [block[arm] for block in blocks.values()]
            summary[arm] = {
                "median_tokens_per_second": median(
                    row["tokens_per_second"] for row in arm_rows
                ),
                "median_cohort_p50_ttft_ms": median(
                    row["ttft_ms"]["p50"] for row in arm_rows
                ),
                "median_cohort_p99_request_ms": median(
                    row["request_latency_ms"]["p99"] for row in arm_rows
                ),
            }
        results.append(summary)
    args.output.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# Full-model queued-request comparison",
        "",
        "Within-process paired cohorts; model loading and candidate setup are excluded. HTTP and tokenization are excluded. Cohort P99 is not a steady-state SLO result.",
        "",
        f"| B | C | {baseline_name} tok/s | {candidate_name} tok/s | Paired speedup [95% CI] | Throughput change |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        low, high = row["ci95"]
        lines.append(
            f"| {row['batch']} | {row['concurrency']} | {row[baseline_name]['median_tokens_per_second']:.3f} | {row[candidate_name]['median_tokens_per_second']:.3f} | {row['paired_speedup']:.3f}× [{low:.3f}, {high:.3f}] | {row['throughput_gain_percent']:+.2f}% |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
