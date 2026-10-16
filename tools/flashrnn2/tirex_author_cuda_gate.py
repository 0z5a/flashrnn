"""Qualify TiRex's original CUDA cell against its Torch checkpoint path."""

import argparse
import json
import sys
import tarfile
import traceback
from pathlib import Path

import torch
from audit_tirex_gpu_gate import pairs
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

XLSTM_SOURCE_SHA256 = "dc919005b21352a0b65f075c2dc81012303ee23b78d9bc091c2b76e48a871025"
BUDGET = 1e-4
BATCH_SETS = {
    "small": ((1, 16), (2, 32), (4, 64)),
    "large": ((16, 16), (32, 32), (64, 64)),
}


def large_series_inputs():
    t = torch.arange(128, dtype=torch.float32)
    return torch.stack(
        [
            (1 + 0.03 * index) * torch.sin(t / (7 + index % 5) + index / 3)
            + 0.004 * (index + 1) * t
            + 0.2 * torch.cos(t / (17 + index % 4))
            for index in range(64)
        ]
    )


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


def comparison(key, reference, candidate):
    parts = key.split("/")
    row = {"scope": parts[0], "batch": int(parts[1][1:]), "tensor": parts[-1]}
    if row["scope"] == "cell":
        row["steps"] = int(parts[2][1:])
    elif row["scope"] == "forecast_cell":
        row["block"] = int(parts[2][1:])
        row["patch"] = int(parts[3][1:])
    if reference.shape != candidate.shape:
        return row | {
            "pass": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    return row | compare(candidate.float(), reference.float(), BUDGET)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--xlstm-source", type=Path, required=True)
    parser.add_argument("--xlstm-source-archive", type=Path, required=True)
    parser.add_argument("--batch-set", choices=tuple(BATCH_SETS), default="small")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = BATCH_SETS[args.batch_set]
    raw_path = args.output.with_suffix(".pt")
    assert not args.output.exists() and not raw_path.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    weight = args.model / "model.ckpt"
    assert weight.stat().st_size == WEIGHT_BYTES and sha256(weight) == WEIGHT_SHA256
    pins = json.loads((args.source / "source-pins.json").read_text())
    assert pins["revision"] == SOURCE_REVISION
    for entry in pins["files"]:
        path = args.source / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert git_blob_sha(path) == entry["sha"]
    assert sha256(args.xlstm_source_archive) == XLSTM_SOURCE_SHA256
    root = "xlstm-ab22eadbd245f293dd8dec38ed29963d73758a12/"
    with tarfile.open(args.xlstm_source_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                assert member.name.startswith(root)
                source = args.xlstm_source / member.name.removeprefix(root)
                assert source.read_bytes() == archive.extractfile(member).read()
    assert torch.cuda.device_count() == 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    shim = add_training_import_shim()
    sys.path.insert(0, str(args.xlstm_source))
    sys.path.insert(0, str(args.source / "src"))
    from tirex.models.tirex import TiRexZero

    raw = {"cell": [], "forecast": []}
    rows = []
    status = "INCOMPLETE"
    failure = None
    try:
        checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
        state = {
            name.removeprefix("block_stack."): value
            for name, value in checkpoint["state_dict"].items()
        }
        reference = TiRexZero(backend="torch", **checkpoint["hyper_parameters"])
        reference.load_state_dict(state, strict=True)
        candidate = TiRexZero(backend="cuda", **checkpoint["hyper_parameters"])
        candidate.load_state_dict(state, strict=True)
        reference.to("cuda").eval()
        candidate.to("cuda").eval()
        del checkpoint, state
        cell = reference.blocks[0].slstm_layer.slstm_cell
        cuda_cell = candidate.blocks[0].slstm_layer.slstm_cell
        hidden = cell.config.embedding_dim

        with torch.inference_mode():
            for batch, steps in cases:
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
                    value.detach().cpu().clone() for value in cuda_cell(gates, initial)
                )

            inputs = (
                series_inputs() if args.batch_set == "small" else large_series_inputs()
            )
            for batch, _ in cases:
                context = inputs[:batch]
                entry = {
                    "batch": batch,
                    "reference": {
                        "forecast": None,
                        "cells": [[] for _ in reference.blocks],
                    },
                    "candidate": {
                        "forecast": None,
                        "cells": [[] for _ in candidate.blocks],
                    },
                }
                raw["forecast"].append(entry)
                entry["reference"]["forecast"] = forecast_with_cells(
                    reference, context, entry["reference"]["cells"]
                )
                entry["candidate"]["forecast"] = forecast_with_cells(
                    candidate, context, entry["candidate"]["cells"]
                )
        rows = [
            comparison(key, expected, actual) for key, expected, actual in pairs(raw)
        ]
        assert len(rows) == 381
        status = "PASS" if all(row["pass"] for row in rows) else "FAIL_NUMERICAL"
    except Exception:
        status = "ERROR"
        failure = traceback.format_exc()
        raise
    finally:
        torch.save(raw, raw_path)
        args.output.write_text(
            json.dumps(
                {
                    "status": status,
                    "failure_traceback": failure,
                    "checkpoint_sha256": WEIGHT_SHA256,
                    "source_revision": SOURCE_REVISION,
                    "xlstm_source_sha256": XLSTM_SOURCE_SHA256,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "training_only_sklearn_import_shim": shim,
                    "budget": BUDGET,
                    "batch_set": args.batch_set,
                    "baseline": "original TiRex xLSTM CUDA cell",
                    "raw_sha256": sha256(raw_path),
                    "comparisons": rows,
                },
                indent=2,
            )
            + "\n"
        )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
