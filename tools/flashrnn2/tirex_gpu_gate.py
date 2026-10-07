"""Qualify the pinned TiRex checkpoint before timing forecast E2E."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import triton
from flashrnn.flashrnn2.tirex_adapter import install_tirex_cells, slstm_cell
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


def compare_cell(records, scope, reference, candidate, budget):
    names = ("hidden", "cell", "normalizer", "stabilizer")
    tensors = [("output", reference[0], candidate[0])]
    tensors.extend(
        (name, reference[1][index], candidate[1][index])
        for index, name in enumerate(names)
    )
    for name, expected, actual in tensors:
        records.append(scope | {"tensor": name} | compare(actual, expected, budget))


def forecast_with_cells(model, context):
    cells = [[] for _ in model.blocks]

    def capture(index):
        def hook(module, inputs, result):
            cells[index].append(tuple(value.detach().cpu().clone() for value in result))

        return hook

    hooks = [
        block.slstm_layer.slstm_cell.register_forward_hook(capture(index))
        for index, block in enumerate(model.blocks)
    ]
    forecast = model.forecast(
        context,
        prediction_length=64,
        output_type="torch",
        batch_size=context.shape[0],
        full_rollout=False,
        dynamic_padding=False,
    )
    for hook in hooks:
        hook.remove()
    return forecast, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.01)
    args = parser.parse_args()
    assert args.budget > 0 and not args.output.exists()
    raw_path = args.output.with_suffix(".pt")
    assert not raw_path.exists()

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
    assert 16 <= head_dim <= 128
    hidden = cell.config.embedding_dim

    records = []
    raw = {"cell": [], "forecast": []}
    with torch.inference_mode():
        for batch, steps in ((1, 16), (2, 32), (4, 64)):
            torch.manual_seed(1000 + batch)
            gates = torch.randn(batch, steps, 4 * hidden, device="cuda") * 0.25
            initial = (
                None
                if batch == 1
                else torch.randn(4, batch, hidden, device="cuda") * 0.1
            )
            reference = cell(gates, initial)
            candidate = slstm_cell(gates, initial, cell._recurrent_kernel_, cell._bias_)
            pair = [
                (a.detach().cpu().clone(), b.detach().cpu().clone())
                for a, b in zip(reference, candidate)
            ]
            raw["cell"].append(pair)
            compare_cell(
                records,
                {"scope": "cell", "batch": batch, "steps": steps},
                reference,
                candidate,
                args.budget,
            )

        inputs = series_inputs()
        for batch in (1, 2, 4):
            context = inputs[:batch]
            reference = forecast_with_cells(model, context)
            candidate = forecast_with_cells(candidate_model, context)
            raw["forecast"].append((reference, candidate))
            for name, expected, actual in zip(
                ("quantiles", "median"), reference[0], candidate[0]
            ):
                records.append(
                    {"scope": "forecast", "batch": batch, "tensor": name}
                    | compare(actual, expected, args.budget)
                )
            for index, (reference_calls, candidate_calls) in enumerate(
                zip(reference[1], candidate[1])
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
                        args.budget,
                    )

    torch.save(raw, raw_path)
    result = {
        "status": "PASS" if all(row["pass"] for row in records) else "FAIL",
        "checkpoint_sha256": WEIGHT_SHA256,
        "source_revision": SOURCE_REVISION,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "training_only_sklearn_import_shim": shim,
        "num_heads": cell.config.num_heads,
        "head_dim": head_dim,
        "budget": args.budget,
        "raw_sha256": sha256(raw_path),
        "comparisons": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
