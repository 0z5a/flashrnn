"""Report only complete, independently qualified paired DEER training blocks."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from analyze import summarize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--native-gate-raw", type=Path, required=True)
    parser.add_argument("--native-gate-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_json = args.output.with_suffix(".json")
    if args.output.exists() or result_json.exists():
        parser.error("analysis output already exists")
    meta = json.loads(args.input.with_suffix(".meta.json").read_text())
    qualification = args.input.with_suffix(".qualification.npz")
    qualification_meta = args.input.with_suffix(".qualification.meta.json")
    qualification_record = json.loads(qualification_meta.read_text())
    audit = json.loads(args.input.with_suffix(".qualification.audit.json").read_text())
    native_raw = json.loads(args.native_gate_raw.read_text())
    native_audit = json.loads(args.native_gate_audit.read_text())
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    groups = meta["groups"]
    if (
        meta["status"] != "PASS"
        or meta["scope"] != "SYNTHETIC_FOUR_LAYER_GRU_COMPLETE_TRAINING_STEP"
        or meta["gpu_model"] != "H100"
        or meta["gpu_count"] != 1
        or meta["blocks"] != len(rows)
        or len(rows) < 20
        or meta["raw_sha256"] != sha256(args.input)
        or meta["qualification_sha256"] != sha256(qualification)
        or meta["qualification_meta_sha256"] != sha256(qualification_meta)
        or meta["qualification_audit_sha256"]
        != sha256(args.input.with_suffix(".qualification.audit.json"))
        or meta["runner_sha256"] != qualification_record["runner_sha256"]
        or meta["source_revision"] != qualification_record["source_revision"]
        or meta["source_manifest_sha256"]
        != qualification_record["source_manifest_sha256"]
        or meta["batch"] != qualification_record["batch"]
        or meta["concurrency"] != qualification_record["concurrency"]
        or meta["groups"] != qualification_record["groups"]
        or qualification_record["budget"] != 1e-4
        or qualification_record["learning_rate"] != 1e-3
        or audit["status"] != "PASS"
        or audit["failed_elements"] != 0
        or audit["raw_sha256"] != meta["qualification_sha256"]
        or audit["qualification_meta_sha256"] != meta["qualification_meta_sha256"]
        or audit["source_manifest_sha256"] != meta["source_manifest_sha256"]
        or audit["native_gate_raw_sha256"] != meta["native_gate_raw_sha256"]
        or audit["native_gate_audit_sha256"] != meta["native_gate_audit_sha256"]
        or meta["native_gate_raw_sha256"] != sha256(args.native_gate_raw)
        or meta["native_gate_audit_sha256"] != sha256(args.native_gate_audit)
        or native_audit["status"] != "PASS"
        or native_audit["failed_elements"] != 0
        or native_audit["paired_tensor_count"] != 21
        or native_audit["raw_sha256"] != meta["native_gate_raw_sha256"]
        or native_audit["source_manifest_sha256"] != meta["source_manifest_sha256"]
        or native_raw["source_manifest_sha256"] != meta["source_manifest_sha256"]
        or hashlib.sha256(native_raw["device_uuid"].encode()).hexdigest()
        != meta["device_identity_sha256"]
        or native_raw["jax_version"] != meta["jax_version"]
        or native_raw["jaxlib_version"] != meta["jaxlib_version"]
        or audit["paired_tensor_count"] != groups * 77
        or audit["independent_forward_count"] != groups * 8
        or audit["independent_loss_count"] != groups * 2
        or audit["optimizer_update_count"] != groups * 72
        or groups * meta["batch"] != meta["concurrency"]
    ):
        raise ValueError("complete independent training qualification required")
    for block, row in enumerate(rows):
        order = (
            ["baseline", "candidate"] if block % 2 == 0 else ["candidate", "baseline"]
        )
        if (
            set(row) != {"block", "status", "order", "baseline", "candidate"}
            or row["status"] != "PASS"
            or row["block"] != block
            or row["order"] != order
        ):
            raise ValueError("missing, duplicated or incorrectly ordered paired block")
        for arm in order:
            timing = row[arm]
            if set(timing) != {
                "elapsed_ms",
                "requests_per_second",
                "sequences_per_second",
            }:
                raise ValueError("incomplete training timing")
            elapsed = timing["elapsed_ms"]
            requests = timing["requests_per_second"]
            sequences = timing["sequences_per_second"]
            if (
                not all(
                    math.isfinite(value) for value in (elapsed, requests, sequences)
                )
                or elapsed <= 0
                or not math.isclose(requests, groups * 1000 / elapsed, rel_tol=1e-12)
                or not math.isclose(
                    sequences, meta["concurrency"] * 1000 / elapsed, rel_tol=1e-12
                )
            ):
                raise ValueError("training throughput differs from completed requests")
    summary = summarize(
        [
            (row["baseline"]["elapsed_ms"], row["candidate"]["elapsed_ms"])
            for row in rows
        ]
    )
    summary.update(
        {
            "batch": meta["batch"],
            "concurrency": meta["concurrency"],
            "model": meta["model"],
            "baseline_requests_per_second": median(
                row["baseline"]["requests_per_second"] for row in rows
            ),
            "candidate_requests_per_second": median(
                row["candidate"]["requests_per_second"] for row in rows
            ),
            "baseline_sequences_per_second": median(
                row["baseline"]["sequences_per_second"] for row in rows
            ),
            "candidate_sequences_per_second": median(
                row["candidate"]["sequences_per_second"] for row in rows
            ),
            "qualification_tensor_pairs": audit["paired_tensor_count"],
        }
    )
    result_json.write_text(json.dumps(summary, indent=2) + "\n")
    low, high = summary["ci95"]
    args.output.write_text(
        "\n".join(
            [
                "# DEER complete synthetic GRU training step",
                "",
                "Fixed FP32 weights and inputs; four recurrent layers, full-sequence forward, loss, backward and SGD update. Each request is one independent microbatch of B sequences. C is the total queued sequence count. The measured path includes host-to-device input, the compiled training step, synchronization and host-visible loss. Compilation, qualification and correctness checks are excluded. This is neither pretrained token generation nor an inference serving speedup.",
                "",
                "| Batch | Queued sequences | Sequential requests/s | DEER requests/s | Sequential sequences/s | DEER sequences/s | Paired speedup [95% CI] | Numerical audit |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                f"| {meta['batch']} | {meta['concurrency']} | {summary['baseline_requests_per_second']:.3f} | {summary['candidate_requests_per_second']:.3f} | {summary['baseline_sequences_per_second']:.3f} | {summary['candidate_sequences_per_second']:.3f} | {summary['paired_speedup']:.3f}× [{low:.3f}, {high:.3f}] | {audit['paired_tensor_count']} pairs, zero failures |",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
