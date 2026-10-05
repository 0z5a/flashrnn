"""Compare matched full-model cohorts; correctness-only runs are ineligible."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from analyze import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.input.with_suffix(".meta.json").read_text())
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
    for row in rows:
        if row["phase"] != "measurement":
            continue
        key = row["batch"], row["concurrency"]
        block = groups[key].setdefault(row["block"], {})
        assert row["arm"] not in block
        block[row["arm"]] = row
    assert groups, "no measured paired cohorts"
    assert set(groups) == {
        (metadata["oracles"]["batch"], c) for c in metadata["concurrency"]
    }
    assert metadata["paired_blocks"] >= 20
    results = []
    for (batch, concurrency), blocks in sorted(groups.items()):
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
            steps = metadata["oracles"]["generated_tokens"]
            assert all(len(r["token_ids"]) == steps for r in left)
            assert baseline["model_batches"] == {
                "prefill": [batch] * (concurrency // batch),
                "decode": [batch] * (concurrency // batch * (steps - 1)),
            }
            assert (
                baseline["output_tokens"]
                == candidate["output_tokens"]
                == concurrency * steps
            )
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
        "Within-process paired cohorts; initialization and decode-graph setup are excluded. HTTP and tokenization are excluded. Cohort P99 is not a steady-state SLO result.",
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
