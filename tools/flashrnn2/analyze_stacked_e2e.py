"""Turn one qualified stacked-model AB/BA run into a speedup Markdown row."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from analyze import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    meta = json.loads(args.input.with_suffix(".meta.json").read_text())
    qualification = args.input.with_suffix(".qualification.pt")
    with qualification.open("rb") as handle:
        assert (
            hashlib.file_digest(handle, "sha256").hexdigest()
            == meta["qualification_sha256"]
        )
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    if meta["status"] != "PASS" or len(rows) < 20:
        raise ValueError("complete qualified timing required")
    if meta["baseline"] == "cudnn":
        flags = meta["cudnn_training_flags"]
        if len(flags) != meta["model"]["layers"] or any(
            item["module"] or any(item["heads"]) for item in flags
        ):
            raise ValueError("cuDNN baseline did not run in eval mode")
        packed = False
        if "packed_cudnn_required" in meta:
            packed = (
                meta["packed_cudnn_required"]
                and len(meta["cudnn_weight_storage_counts"]) == meta["model"]["layers"]
                and all(count == 1 for count in meta["cudnn_weight_storage_counts"])
                and all(meta["cudnn_weights_acceptable"])
            )
    else:
        packed = None
    if [row["block"] for row in rows] != list(range(len(rows))):
        raise ValueError("missing or duplicate paired block")
    for row in rows:
        expected_order = (
            ["baseline", "candidate"]
            if row["block"] % 2 == 0
            else ["candidate", "baseline"]
        )
        if row["status"] != "PASS" or row["order"] != expected_order:
            raise ValueError("paired order or correctness failure")
    pairs = [
        (row["baseline"]["elapsed_ms"], row["candidate"]["elapsed_ms"]) for row in rows
    ]
    summary = summarize(pairs)
    batch = meta["workload"]["batch"]
    concurrency = meta["workload"]["concurrency"]
    base_rate = median(row["baseline"]["responses_per_second"] for row in rows)
    candidate_rate = median(row["candidate"]["responses_per_second"] for row in rows)
    summary.update(
        model=meta["model"],
        workload=meta["workload"],
        baseline=meta["baseline"],
        candidate=meta["candidate"],
        baseline_responses_per_second=base_rate,
        candidate_responses_per_second=candidate_rate,
        device_uuid=meta["device_uuid"],
        qualification_max_abs=meta["qualification_max_abs"],
        cudnn_fair_baseline_qualified=packed,
    )
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    low, high = summary["ci95"]
    lines = [
        "# Complete synthetic stacked-model sequence inference",
        "",
        f"Fixed random weights and request IDs in {meta['dtype']}; this is a full {meta['model']['layers']}-layer model workload, not a pretrained checkpoint or autoregressive generation result. Throughput is completed sequence responses/s. Timed scope includes host-to-device IDs, embedding, recurrence, readout, device-to-host logits and queued groups. Compilation, model construction and qualification are excluded.",
        "",
        "| Cell | Baseline | Batch | Concurrent requests | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] | Qualification |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| {meta['model']['cell']} | {meta['baseline']} | {batch} | {concurrency} | {base_rate:.3f} | {candidate_rate:.3f} | {summary['paired_speedup']:.3f}× [{low:.3f}, {high:.3f}] | {'Diagnostic: cuDNN weights not proven packed' if packed is False else 'Qualified'} |",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
