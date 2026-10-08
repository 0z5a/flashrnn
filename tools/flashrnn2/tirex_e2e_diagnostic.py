"""Capture the exact B2/C32 TiRex E2E fixtures at every recurrent boundary."""

import argparse
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from tirex_e2e import contexts
from tirex_gpu_gate import forecast_with_cells
from tirex_slstm_reference_gate import (
    SOURCE_REVISION,
    WEIGHT_BYTES,
    WEIGHT_SHA256,
    add_training_import_shim,
    compare,
    git_blob_sha,
    sha256,
)

from flashrnn.flashrnn2.tirex_adapter import DISPATCH_POLICY, install_tirex_cells

BUDGET = 1e-4
FIXTURE_SHA256 = "1b7dbd6eafc2ab94e59c2df5c613cdfe93a9d12cc9e9d620c7c8905304bc83c8"
STATES = ("hidden", "cell", "normalizer", "stabilizer")


def record_pairs(records, group, reference, candidate):
    assert len(reference["forecast"]) == len(candidate["forecast"]) == 2
    assert len(reference["cells"]) == len(candidate["cells"]) == 12
    for name, expected, actual in zip(
        ("quantiles", "median"), reference["forecast"], candidate["forecast"]
    ):
        records.append(
            {"scope": "forecast", "group": group, "tensor": name}
            | compare(actual, expected, BUDGET)
        )
    for layer, (expected_calls, actual_calls) in enumerate(
        zip(reference["cells"], candidate["cells"])
    ):
        assert len(expected_calls) == len(actual_calls) == 2
        for patch, (expected_pair, actual_pair) in enumerate(
            zip(expected_calls, actual_calls)
        ):
            for name, expected, actual in (
                ("output", expected_pair[0], actual_pair[0]),
                *(
                    (name, expected_pair[1][index], actual_pair[1][index])
                    for index, name in enumerate(STATES)
                ),
            ):
                records.append(
                    {
                        "scope": "forecast_cell",
                        "group": group,
                        "layer": layer,
                        "patch": patch,
                        "tensor": name,
                    }
                    | compare(actual, expected, BUDGET)
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.output.with_suffix(".pt")
    assert not args.output.exists() and not raw_path.exists()

    gate = json.loads(args.gate.read_text())
    audit = json.loads(args.audit.read_text())
    adapter_sha = sha256(Path(inspect.getfile(install_tirex_cells)))
    assert gate["status"] == audit["status"] == "PASS"
    assert gate["budget"] == BUDGET and len(gate["comparisons"]) == 381
    assert all(row["pass"] for row in gate["comparisons"])
    assert gate["dispatch_policy"] == DISPATCH_POLICY
    assert gate["adapter_sha256"] == adapter_sha
    assert gate["raw_sha256"] == sha256(args.gate.with_suffix(".pt"))
    assert audit["gate_sha256"] == sha256(args.gate)
    assert audit["raw_sha256"] == gate["raw_sha256"]
    assert audit["required_pairs"] == audit["observed_pairs"] == 381

    weight = args.model / "model.ckpt"
    assert weight.stat().st_size == WEIGHT_BYTES and sha256(weight) == WEIGHT_SHA256
    pins = json.loads((args.source / "source-pins.json").read_text())
    assert pins["revision"] == SOURCE_REVISION
    for entry in pins["files"]:
        source = args.source / entry["path"]
        assert source.stat().st_size == entry["bytes"]
        assert git_blob_sha(source) == entry["sha"]

    shim = add_training_import_shim()
    sys.path.insert(0, str(args.source / "src"))
    from tirex.models.tirex import TiRexZero

    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    model = TiRexZero(backend="torch", **checkpoint["hyper_parameters"])
    model.load_state_dict(
        {
            name.replace("block_stack.", ""): value
            for name, value in checkpoint["state_dict"].items()
        },
        strict=True,
    )
    model.to("cuda").eval()
    candidate = copy.deepcopy(model)
    install_tirex_cells(candidate)
    del checkpoint

    fixtures = contexts(32, 2)
    fixture_sha = hashlib.sha256(torch.cat(fixtures).numpy().tobytes()).hexdigest()
    assert fixture_sha == FIXTURE_SHA256
    raw = {"groups": []}
    records = []
    status = "INCOMPLETE"
    try:
        with torch.inference_mode():
            for group, context in enumerate(fixtures):
                entry = {
                    "group": group,
                    "context": context.clone(),
                    "reference": {
                        "forecast": None,
                        "cells": [[] for _ in model.blocks],
                    },
                    "candidate": {
                        "forecast": None,
                        "cells": [[] for _ in model.blocks],
                    },
                }
                raw["groups"].append(entry)
                for arm, selected in (("reference", model), ("candidate", candidate)):
                    outputs = forecast_with_cells(
                        selected, context, entry[arm]["cells"]
                    )
                    entry[arm]["forecast"] = tuple(
                        output.detach().cpu().clone() for output in outputs
                    )
                record_pairs(records, group, entry["reference"], entry["candidate"])
        assert len(raw["groups"]) == 16 and len(records) == 16 * (2 + 12 * 2 * 5)
        status = "PASS" if all(row["pass"] for row in records) else "FAIL_NUMERICAL"
    finally:
        torch.save(raw, raw_path)
        args.output.write_text(
            json.dumps(
                {
                    "status": status,
                    "scope": "Exact B2/C32 E2E fixtures; correctness only, no timing",
                    "fixture_sha256": fixture_sha,
                    "groups": len(raw["groups"]),
                    "batch": 2,
                    "queued_series": 32,
                    "budget": BUDGET,
                    "checkpoint_sha256": WEIGHT_SHA256,
                    "source_revision": SOURCE_REVISION,
                    "adapter_sha256": adapter_sha,
                    "dispatch_policy": DISPATCH_POLICY,
                    "qualification_gate_sha256": sha256(args.gate),
                    "qualification_audit_sha256": sha256(args.audit),
                    "raw_sha256": sha256(raw_path),
                    "training_only_sklearn_import_shim": shim,
                    "comparisons": records,
                },
                indent=2,
            )
            + "\n"
        )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
