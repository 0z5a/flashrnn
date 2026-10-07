"""Qualify the pinned TiRex checkpoint before timing forecast E2E."""

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

import torch
import triton
from tirex_slstm_reference_gate import (
    SOURCE_REVISION,
    WEIGHT_BYTES,
    WEIGHT_SHA256,
    add_training_import_shim,
    compare,
    git_blob_sha,
    series_inputs,
    sha256,
)

from flashrnn.flashrnn2.tirex_adapter import install_tirex_cells, slstm_cell

BUDGET = 1e-4


def compare_cell(records, scope, reference, candidate, budget):
    names = ("hidden", "cell", "normalizer", "stabilizer")
    tensors = [("output", reference[0], candidate[0])]
    tensors.extend(
        (name, reference[1][index], candidate[1][index])
        for index, name in enumerate(names)
    )
    for name, expected, actual in tensors:
        records.append(scope | {"tensor": name} | compare(actual, expected, budget))


def forecast_with_cells(model, context, cells):
    def capture(index):
        def hook(module, inputs, result):
            cells[index].append(tuple(value.detach().cpu().clone() for value in result))

        return hook

    hooks = [
        block.slstm_layer.slstm_cell.register_forward_hook(capture(index))
        for index, block in enumerate(model.blocks)
    ]
    try:
        return model.forecast(
            context,
            prediction_length=64,
            output_type="torch",
            batch_size=context.shape[0],
            full_rollout=False,
            dynamic_padding=False,
        )
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    raw_path = args.output.with_suffix(".pt")
    assert not raw_path.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)

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

    records = []
    raw = {"cell": [], "forecast": []}
    status = "INCOMPLETE"
    failure = None
    num_heads = head_dim = device_name = None
    try:
        checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
        model = TiRexZero(backend="torch", **checkpoint["hyper_parameters"])
        state = {
            name.replace("block_stack.", ""): value
            for name, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state, strict=True)
        model.to("cuda").eval()
        candidate_model = copy.deepcopy(model)
        install_tirex_cells(candidate_model)
        del checkpoint, state
        cell = model.blocks[0].slstm_layer.slstm_cell
        head_dim = cell.config.head_dim
        num_heads = cell.config.num_heads
        device_name = torch.cuda.get_device_name()
        assert 16 <= head_dim <= 128
        hidden = cell.config.embedding_dim

        with torch.inference_mode():
            for batch, steps in ((1, 16), (2, 32), (4, 64)):
                torch.manual_seed(1000 + batch)
                gates = torch.randn(batch, steps, 4 * hidden, device="cuda") * 0.25
                initial = (
                    None
                    if batch == 1
                    else torch.randn(4, batch, hidden, device="cuda") * 0.1
                )
                entry = {
                    "batch": batch,
                    "steps": steps,
                    "reference": None,
                    "candidate": None,
                }
                raw["cell"].append(entry)
                entry["reference"] = tuple(
                    value.detach().cpu().clone() for value in cell(gates, initial)
                )
                entry["candidate"] = tuple(
                    value.detach().cpu().clone()
                    for value in slstm_cell(
                        gates, initial, cell._recurrent_kernel_, cell._bias_
                    )
                )
                compare_cell(
                    records,
                    {"scope": "cell", "batch": batch, "steps": steps},
                    entry["reference"],
                    entry["candidate"],
                    BUDGET,
                )

            inputs = series_inputs()
            for batch in (1, 2, 4):
                context = inputs[:batch]
                entry = {
                    "batch": batch,
                    "reference": {
                        "forecast": None,
                        "cells": [[] for _ in model.blocks],
                    },
                    "candidate": {
                        "forecast": None,
                        "cells": [[] for _ in model.blocks],
                    },
                }
                raw["forecast"].append(entry)
                entry["reference"]["forecast"] = forecast_with_cells(
                    model, context, entry["reference"]["cells"]
                )
                entry["candidate"]["forecast"] = forecast_with_cells(
                    candidate_model, context, entry["candidate"]["cells"]
                )
                for name, expected, actual in zip(
                    ("quantiles", "median"),
                    entry["reference"]["forecast"],
                    entry["candidate"]["forecast"],
                ):
                    records.append(
                        {"scope": "forecast", "batch": batch, "tensor": name}
                        | compare(actual, expected, BUDGET)
                    )
                for index, (reference_calls, candidate_calls) in enumerate(
                    zip(entry["reference"]["cells"], entry["candidate"]["cells"])
                ):
                    assert len(reference_calls) == len(candidate_calls) == 2
                    for patch, (reference_pair, candidate_pair) in enumerate(
                        zip(reference_calls, candidate_calls)
                    ):
                        compare_cell(
                            records,
                            {
                                "scope": "forecast_cell",
                                "batch": batch,
                                "block": index,
                                "patch": patch,
                            },
                            reference_pair,
                            candidate_pair,
                            BUDGET,
                        )
        status = "PASS" if all(row["pass"] for row in records) else "FAIL_NUMERICAL"
    except Exception:
        status = "ERROR"
        failure = traceback.format_exc()
        raise
    finally:
        torch.save(raw, raw_path)
        result = {
            "status": status,
            "failure_traceback": failure,
            "checkpoint_sha256": WEIGHT_SHA256,
            "source_revision": SOURCE_REVISION,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "device": device_name,
            "training_only_sklearn_import_shim": shim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "budget": BUDGET,
            "raw_sha256": sha256(raw_path),
            "comparisons": records,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
