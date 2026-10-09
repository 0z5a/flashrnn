"""Independently audit the pinned full-model FlashRNN2 xLSTM comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCOPE = "PINNED_XLSTM_FLASHRNN2_FP32_FULL_MODEL_VS_AUTHOR_CUDA_NO_TIMING"
REVISION = "34d473516f5391ffb59df94313b4acb6fd0745b6"
WEIGHT_SHA256 = "1ed83701a3164eab20a2abb09278e8a2635d7ea8000253d4faec9cd6b6b397d1"
SOURCE_ARCHIVE_SHA256 = (
    "dc919005b21352a0b65f075c2dc81012303ee23b78d9bc091c2b76e48a871025"
)
STATES = (
    "mlstm_0",
    "mlstm_1",
    "mlstm_2",
    "mlstm_conv",
    "slstm_conv",
    "slstm_hcnm",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--author-raw", type=Path, required=True)
    parser.add_argument("--author-audit", type=Path, required=True)
    parser.add_argument("--cell-raw", type=Path, required=True)
    parser.add_argument("--cell-audit", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--layout-source", type=Path, required=True)
    parser.add_argument("--small-full-raw", type=Path)
    parser.add_argument("--small-full-audit", type=Path)
    parser.add_argument("--batch-set", choices=("small", "large"), default="small")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batches = (1, 2, 4) if args.batch_set == "small" else (16, 32, 64)
    meta = json.loads(args.raw.with_suffix(".meta.json").read_text())
    raw_sha, rows_sha = digest(args.raw), digest(args.rows)
    author_audit = json.loads(args.author_audit.read_text())
    cell_audit = json.loads(args.cell_audit.read_text())
    author_meta = json.loads(args.author_raw.with_suffix(".meta.json").read_text())
    cell_meta = json.loads(args.cell_raw.with_suffix(".meta.json").read_text())
    if (
        meta["scope"] != SCOPE
        or meta["status"] not in ("PASS", "NUMERICAL_FAILED")
        or meta["model_revision"] != REVISION
        or meta["weight_sha256"] != WEIGHT_SHA256
        or meta["source_archive_sha256"] != SOURCE_ARCHIVE_SHA256
        or meta["budget"] != {"atol": 1e-4, "rtol": 1e-4}
        or meta["raw_sha256"] != raw_sha
        or meta["rows_sha256"] != rows_sha
        or meta["author_audit_sha256"] != digest(args.author_audit)
        or meta["cell_audit_sha256"] != digest(args.cell_audit)
        or meta["candidate_source_sha256"] != digest(args.candidate_source)
        or meta["kernel_source_sha256"] != digest(args.kernel_source)
        or meta["layout_source_sha256"] != digest(args.layout_source)
        or meta.get("batch_set", "small") != args.batch_set
        or meta.get("batches", [1, 2, 4]) != list(batches)
        or author_audit["status"] != "PASS"
        or cell_audit["status"] != "PASS"
        or author_meta["status"] != "PASS"
        or cell_meta["status"] != "PASS"
        or author_meta.get("batch_set", "small") != args.batch_set
        or author_audit["raw_sha256"] != digest(args.author_raw)
        or cell_audit["raw_sha256"] != digest(args.cell_raw)
        or cell_meta["layout_source_sha256"] != meta["layout_source_sha256"]
    ):
        raise ValueError("checkpoint, prerequisites or saved GPU result changed")
    if args.batch_set == "large":
        if args.small_full_raw is None or args.small_full_audit is None:
            parser.error("large batches require the small full-model audit")
        small_audit = json.loads(args.small_full_audit.read_text())
        small_meta = json.loads(
            args.small_full_raw.with_suffix(".meta.json").read_text()
        )
        if (
            meta["small_full_audit_sha256"] != digest(args.small_full_audit)
            or small_audit["status"] != "PASS"
            or small_meta["status"] != "PASS"
            or small_meta.get("batch_set", "small") != "small"
            or small_meta["candidate_source_sha256"] != meta["candidate_source_sha256"]
            or small_meta["kernel_source_sha256"] != meta["kernel_source_sha256"]
            or small_meta["layout_source_sha256"] != meta["layout_source_sha256"]
            or small_audit["raw_sha256"] != digest(args.small_full_raw)
        ):
            raise ValueError("small full-model qualification changed")
    data = torch.load(args.raw, map_location="cpu", weights_only=True)
    pairs, tokens = data["pairs"], data["tokens"]
    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    if len(pairs) != 675 or len(tokens) != 36 or len(rows) != 9:
        raise ValueError("incomplete full-model evidence")
    if meta["tensor_pairs"] != len(pairs) or meta["token_rows"] != len(tokens):
        raise ValueError("producer case counts differ from saved tensors")
    expected_pairs = {
        (batch, case, phase, step, name)
        for batch in batches
        for case in range(3)
        for phase, steps, names in (
            ("prefill", range(5), ("step_logits", *STATES)),
            (
                "forecast",
                range(4),
                (
                    "full_logits",
                    "reference_cached_vs_full",
                    "candidate_cached_vs_full",
                ),
            ),
            ("decode", range(4), ("step_logits", *STATES)),
        )
        for step in steps
        for name in names
    }
    results = {
        (batch, case): {
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
        for batch in batches
        for case in range(3)
    }
    seen = set()
    for pair in pairs:
        key = (pair["batch"], pair["case"])
        identity = (*key, pair["phase"], pair["step"], pair["name"])
        if identity not in expected_pairs or identity in seen:
            raise ValueError("unknown or repeated tensor boundary")
        seen.add(identity)
        reference = pair["reference"].numpy()
        candidate = pair["candidate"].numpy()
        shape_mismatch = reference.shape != candidate.shape
        failed = (
            max(reference.size, candidate.size)
            if shape_mismatch
            else int(
                (
                    ~np.isfinite(reference)
                    | ~np.isfinite(candidate)
                    | ~np.isclose(candidate, reference, atol=1e-4, rtol=1e-4)
                ).sum()
            )
        )
        result = results[key]
        result["tensor_pairs"] += 1
        result["elements"] += int(reference.size)
        result["failed_elements"] += failed
        result["shape_mismatches"] += int(shape_mismatch)
        result["producer_disagreements"] += int(failed != pair["producer_failed"])
    if seen != expected_pairs:
        raise ValueError("missing tensor boundary")
    expected_tokens = {
        (batch, case, step)
        for batch in batches
        for case in range(3)
        for step in range(4)
    }
    seen_tokens = set()
    for token in tokens:
        key = (token["batch"], token["case"])
        identity = (*key, token["generation"])
        if identity not in expected_tokens or identity in seen_tokens:
            raise ValueError("unknown or repeated generated token")
        seen_tokens.add(identity)
        result = results[key]
        result["token_rows"] += 1
        result["token_mismatches"] += int(
            not np.array_equal(token["reference"].numpy(), token["candidate"].numpy())
        )
    if seen_tokens != expected_tokens:
        raise ValueError("missing generated token")
    row_keys = set()
    for row in rows:
        key = (row["batch"], row["case"])
        if key not in results or key in row_keys:
            raise ValueError("unknown or repeated full-model case")
        row_keys.add(key)
        result = results[key]
        for field in (
            "tensor_pairs",
            "failed_elements",
            "token_rows",
            "token_mismatches",
        ):
            result["producer_disagreements"] += int(row[field] != result[field])
    if row_keys != set(results):
        raise ValueError("missing full-model case")
    passed = all(
        result["tensor_pairs"] == 75
        and result["token_rows"] == 4
        and result["failed_elements"] == 0
        and result["shape_mismatches"] == 0
        and result["token_mismatches"] == 0
        and result["producer_disagreements"] == 0
        for result in results.values()
    )
    report = {
        "status": "PASS" if passed else "FAIL_NUMERICAL",
        "raw_sha256": raw_sha,
        "rows_sha256": rows_sha,
        "layout_source_sha256": meta["layout_source_sha256"],
        "tensor_pairs": len(pairs),
        "token_rows": len(tokens),
        "results": [results[key] for key in sorted(results)],
    }
    if meta["status"] != ("PASS" if passed else "NUMERICAL_FAILED"):
        report["status"] = "FAIL_REPORT_DISAGREEMENT"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
