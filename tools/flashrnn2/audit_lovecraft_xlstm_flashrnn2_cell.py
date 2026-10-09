"""Independently audit saved xLSTM author-CUDA/FlashRNN2 cell tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

WEIGHT_SHA256 = "1ed83701a3164eab20a2abb09278e8a2635d7ea8000253d4faec9cd6b6b397d1"
SOURCE_ARCHIVE_SHA256 = (
    "dc919005b21352a0b65f075c2dc81012303ee23b78d9bc091c2b76e48a871025"
)
CASES = (
    (1, 1, "zero"),
    (2, 16, "positive"),
    (4, 64, "positive"),
    (4, 16, "mixed_zero"),
)
NAMES = ("output", "hidden", "cell", "normalizer", "stabilizer")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads(args.raw.with_suffix(".meta.json").read_text())
    if (
        meta["raw_sha256"] != digest(args.raw)
        or meta["rows_sha256"] != digest(args.rows)
        or meta["weight_sha256"] != WEIGHT_SHA256
        or meta["source_archive_sha256"] != SOURCE_ARCHIVE_SHA256
        or meta["scope"] != "PINNED_XLSTM_FP32_FLASHRNN2_CELL_VS_AUTHOR_CUDA_NO_TIMING"
        or meta["budget"] != {"atol": 1e-4, "rtol": 1e-4}
    ):
        raise ValueError("checkpoint, original budget or raw evidence changed")
    records = torch.load(args.raw, map_location="cpu", weights_only=True)
    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    expected = {
        (batch, steps, kind, name) for batch, steps, kind in CASES for name in NAMES
    }
    if len(records) != 20 or len(rows) != 4 or meta["tensor_pairs"] != 20:
        raise ValueError("incomplete cell comparison")
    seen = set()
    counts = {(batch, steps, kind): 0 for batch, steps, kind in CASES}
    disagreements = 0
    elements = 0
    for record in records:
        key = (record["batch"], record["steps"], record["state_kind"], record["name"])
        if key not in expected or key in seen:
            raise ValueError("unknown or repeated cell boundary")
        seen.add(key)
        reference = record["reference"].numpy()
        candidate = record["candidate"].numpy()
        expected_shape = (
            (record["batch"], 1, record["steps"], 64)
            if record["name"] == "output"
            else (record["batch"], 64)
        )
        if reference.shape != expected_shape:
            raise ValueError("author CUDA cell boundary has unexpected shape")
        elements += reference.size
        if reference.shape != candidate.shape:
            failed = max(reference.size, candidate.size)
        else:
            failed = int(
                (
                    ~np.isfinite(reference)
                    | ~np.isfinite(candidate)
                    | ~np.isclose(candidate, reference, atol=1e-4, rtol=1e-4)
                ).sum()
            )
        counts[key[:3]] += failed
        disagreements += int(failed != record["producer_failed"])
    if seen != expected:
        raise ValueError("missing cell boundary")
    row_keys = set()
    for row in rows:
        key = (row["batch"], row["steps"], row["state_kind"])
        if key not in counts or key in row_keys or row["tensor_pairs"] != 5:
            raise ValueError("unknown, repeated or incomplete cell case")
        row_keys.add(key)
        disagreements += int(row["failed_elements"] != counts[key])
    if row_keys != set(counts):
        raise ValueError("missing cell case")
    passed = not any(counts.values()) and disagreements == 0
    report = {
        "status": "PASS" if passed else "FAIL_NUMERICAL",
        "raw_sha256": digest(args.raw),
        "rows_sha256": digest(args.rows),
        "tensor_pairs": len(records),
        "elements": elements,
        "failed_elements": sum(counts.values()),
        "producer_disagreements": disagreements,
        "cases": [
            {
                "batch": b,
                "steps": t,
                "state_kind": s,
                "failed_elements": counts[b, t, s],
            }
            for b, t, s in CASES
        ],
    }
    producer_status = "PASS" if passed else "NUMERICAL_FAILED"
    if meta["status"] != producer_status:
        report["status"] = "FAIL_REPORT_DISAGREEMENT"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
