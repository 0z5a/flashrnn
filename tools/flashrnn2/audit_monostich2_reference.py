"""Recompute saved Monostich-2 logits and token comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    meta = json.loads(args.rows.with_suffix(".meta.json").read_text())
    tensor_path = args.rows.with_suffix(".pt")
    with tensor_path.open("rb") as handle:
        assert (
            hashlib.file_digest(handle, "sha256").hexdigest() == meta["tensor_sha256"]
        )
    snapshots = torch.load(tensor_path, weights_only=True, map_location="cpu")
    assert len(rows) == len(snapshots) == meta["batch_steps"] == 36
    expected = {
        (batch, case, step)
        for batch in (1, 2, 4)
        for case in range(3)
        for step in range(4)
    }
    assert {(row["batch"], row["case"], row["step"]) for row in rows} == expected
    for row, saved in zip(rows, snapshots, strict=True):
        assert (row["batch"], row["case"], row["step"]) == (
            saved["batch"],
            saved["case"],
            saved["step"],
        )
        left, right = saved["batched_logits"], saved["serial_logits"]
        assert left.shape == right.shape == (row["batch"], 49152)
        difference = (left - right).abs()
        allowed = 1e-4 + 1e-4 * right.abs()
        failed = ~torch.isfinite(left) | ~torch.isfinite(right) | (difference > allowed)
        assert row["logits"]["pass"] == (not failed.any().item())
        assert row["logits"]["max_abs"] == difference.max().item()
        assert row["logits"]["failed_elements"] == int(failed.sum())
        assert row["logits"]["worst_normalized"] == (difference / allowed).max().item()
        assert torch.equal(saved["tokens"], left.argmax(-1))
        assert row["token_equal"] == torch.equal(left.argmax(-1), right.argmax(-1))
        assert row["pass"] == (row["logits"]["pass"] and row["token_equal"])
    assert sum(row["batch"] for row in rows) == meta["token_choices"] == 84
    for path, digest in meta["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    audit = {
        "status": "PASS",
        "scope": "SAVED_FULL_LOGITS_AND_TOKEN_INDEPENDENT_AUDIT",
        "rows": len(rows),
        "logits_pass": sum(row["logits"]["pass"] for row in rows),
        "token_choices": meta["token_choices"],
        "snapshot_sha256": meta["tensor_sha256"],
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "source_files_verified": len(meta["source_sha256"]),
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
