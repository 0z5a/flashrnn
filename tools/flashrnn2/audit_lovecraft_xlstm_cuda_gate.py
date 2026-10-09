"""Independently compare saved complete-model xLSTM CUDA and vanilla tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

MODEL_REVISION = "34d473516f5391ffb59df94313b4acb6fd0745b6"
WEIGHT_SHA256 = "1ed83701a3164eab20a2abb09278e8a2635d7ea8000253d4faec9cd6b6b397d1"
SOURCE_ARCHIVE_SHA256 = (
    "dc919005b21352a0b65f075c2dc81012303ee23b78d9bc091c2b76e48a871025"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--batch-set", choices=("small", "large"), default="small")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batches = (1, 2, 4) if args.batch_set == "small" else (16, 32, 64)
    raw_sha = hashlib.sha256(args.raw.read_bytes()).hexdigest()
    rows_sha = hashlib.sha256(args.rows.read_bytes()).hexdigest()
    meta = json.loads(args.raw.with_suffix(".meta.json").read_text())
    if (meta["raw_sha256"], meta["rows_sha256"]) != (raw_sha, rows_sha):
        raise ValueError("saved GPU result differs from producer metadata")
    if meta["budget"] != {"atol": 1e-4, "rtol": 1e-4}:
        raise ValueError("original numerical budget changed")
    if (
        meta["model_revision"] != MODEL_REVISION
        or meta["weight_sha256"] != WEIGHT_SHA256
        or meta["source_archive_sha256"] != SOURCE_ARCHIVE_SHA256
        or meta.get("batch_set", "small") != args.batch_set
        or meta.get("batches", [1, 2, 4]) != list(batches)
    ):
        raise ValueError("checkpoint identity changed")
    data = torch.load(args.raw, map_location="cpu", weights_only=True)
    pairs, tokens = data["pairs"], data["tokens"]
    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    if len(pairs) != 675 or len(tokens) != 36 or len(rows) != 9:
        raise ValueError("incomplete full-model checkpoint gate")
    if meta["tensor_pairs"] != len(pairs) or meta["token_rows"] != len(tokens):
        raise ValueError("producer case counts differ from saved tensors")
    identities = set()
    results = {}
    for batch in batches:
        for case in range(3):
            results[(batch, case)] = {
                "batch": batch,
                "case": case,
                "tensor_pairs": 0,
                "elements": 0,
                "failed_elements": 0,
                "shape_mismatches": 0,
                "producer_disagreements": 0,
                "token_rows": 0,
                "token_mismatches": 0,
            }
    state_names = (
        "mlstm_0",
        "mlstm_1",
        "mlstm_2",
        "mlstm_conv",
        "slstm_conv",
        "slstm_hcnm",
    )
    expected_pairs = {
        (batch, case, phase, step, name)
        for batch in batches
        for case in range(3)
        for phase, steps, names in (
            ("prefill", range(5), ("step_logits", *state_names)),
            (
                "forecast",
                range(4),
                ("full_logits", "vanilla_cached_vs_full", "cuda_cached_vs_full"),
            ),
            ("decode", range(4), ("step_logits", *state_names)),
        )
        for step in steps
        for name in names
    }
    for pair in pairs:
        key = (pair["batch"], pair["case"])
        identity = (*key, pair["phase"], pair["step"], pair["name"])
        if key not in results or identity in identities:
            raise ValueError("unknown or repeated tensor comparison")
        identities.add(identity)
        reference = pair["reference"].numpy()
        native = pair["native"].numpy()
        shape_mismatch = reference.shape != native.shape
        count = (
            max(reference.size, native.size)
            if shape_mismatch
            else int(
                (
                    ~np.isfinite(reference)
                    | ~np.isfinite(native)
                    | ~np.isclose(native, reference, atol=1e-4, rtol=1e-4)
                ).sum()
            )
        )
        result = results[key]
        result["tensor_pairs"] += 1
        result["elements"] += int(reference.size)
        result["failed_elements"] += count
        result["shape_mismatches"] += int(shape_mismatch)
        result["producer_disagreements"] += int(count != pair["producer_failed"])
    if identities != expected_pairs:
        raise ValueError("missing or unexpected numerical boundary")
    token_identities = set()
    for token in tokens:
        key = (token["batch"], token["case"])
        identity = (*key, token["generation"])
        if key not in results or identity in token_identities:
            raise ValueError("unknown or repeated token comparison")
        token_identities.add(identity)
        result = results[key]
        result["token_rows"] += 1
        result["token_mismatches"] += int(
            not np.array_equal(token["reference"].numpy(), token["native"].numpy())
        )
    if token_identities != {
        (batch, case, generation)
        for batch in batches
        for case in range(3)
        for generation in range(4)
    }:
        raise ValueError("missing or unexpected generated token")
    for row in rows:
        key = (row["batch"], row["case"])
        if key not in results:
            raise ValueError("unknown full-model case")
        result = results[key]
        for field in (
            "tensor_pairs",
            "failed_elements",
            "token_rows",
            "token_mismatches",
        ):
            result["producer_disagreements"] += int(row[field] != result[field])
    if {(row["batch"], row["case"]) for row in rows} != set(results):
        raise ValueError("missing full-model case")
    if any(
        row["tensor_pairs"] != 75 or row["token_rows"] != 4 for row in results.values()
    ):
        raise ValueError("incomplete per-case full-model trace")
    report = {
        "status": "PASS"
        if all(
            row["failed_elements"] == 0
            and row["shape_mismatches"] == 0
            and row["token_mismatches"] == 0
            and row["producer_disagreements"] == 0
            for row in results.values()
        )
        else "FAIL_NUMERICAL",
        "raw_sha256": raw_sha,
        "rows_sha256": rows_sha,
        "tensor_pairs": len(pairs),
        "token_rows": len(tokens),
        "results": [results[key] for key in sorted(results)],
    }
    producer_status = "PASS" if report["status"] == "PASS" else "NUMERICAL_FAILED"
    if meta["status"] != producer_status:
        report["status"] = "FAIL_REPORT_DISAGREEMENT"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "tensor_pairs": len(pairs)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
