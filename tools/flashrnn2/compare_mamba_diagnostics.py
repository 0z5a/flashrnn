"""Compare saved CPU/CUDA snapshots without rerunning the model or changing gates."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mamba_script_diagnostic import details


def load(path: Path) -> tuple[list[dict], dict]:
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    assert digest == metadata["tensor_sha256"]
    return torch.load(path, map_location="cpu", weights_only=True), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    actual, actual_meta = load(args.actual)
    reference, reference_meta = load(args.reference)
    assert len(actual) == len(reference) == 3
    for key in ("model", "oracles"):
        assert (
            actual_meta[key]["artifact_sha256"]
            == reference_meta[key]["artifact_sha256"]
        )
    contracts = actual_meta["oracles"]
    assert contracts["logits_contract"] == reference_meta["oracles"]["logits_contract"]
    assert contracts["cache_contract"] == reference_meta["oracles"]["cache_contract"]
    rows = []
    for index, (left, right) in enumerate(zip(actual, reference)):
        row = {"case": index}
        for phase in ("prefill", "decode"):
            row[phase] = details(left[phase], right[phase], contracts)
        row["within_original_budgets"] = all(
            item["pass"] for phase in ("prefill", "decode") for item in row[phase]
        )
        rows.append(row)
    result = {
        "scope": "SAME_ARTIFACT_SNAPSHOT_COMPARISON_DIAGNOSIS_ONLY",
        "actual": {
            key: actual_meta[key] for key in ("device", "torch", "tensor_sha256")
        },
        "reference": {
            key: reference_meta[key] for key in ("device", "torch", "tensor_sha256")
        },
        "contracts": {
            key: contracts[key] for key in ("logits_contract", "cache_contract")
        },
        "cases": rows,
        "original_native_oracle_failures_replaced": False,
        "performance_claim": False,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(details.__code__.co_filename))
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "compared_cases": len(rows),
                "cases_within_original_budgets": sum(
                    row["within_original_budgets"] for row in rows
                ),
                "diagnosis_only": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
