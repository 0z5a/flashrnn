"""Full TiRex sLSTM CPU forecast: batched, serial and two-patch references."""

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

REVISION = "63c740922493f5fbe60b277609ec62babfba2762"
SOURCE_REVISION = "91b67bc6e5d4d5d1e68a302dd848dd1d612f4c97"
WEIGHT_BYTES = 141230262
WEIGHT_SHA256 = "b8c3f5a036c63272ce4b91c00187e26922a394cb6cb49d4e16db070ad0422314"
STATE_NAMES = ("hidden", "cell", "normalizer", "stabilizer")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def compare(actual: torch.Tensor, expected: torch.Tensor, budget: float) -> dict:
    assert actual.shape == expected.shape
    difference = (actual - expected).abs()
    allowed = budget * (1 + expected.abs())
    failed = (
        ~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)
    )
    return {
        "pass": not failed.any().item(),
        "bitwise": torch.equal(actual, expected),
        "max_abs": difference.max().item(),
        "worst_normalized": (difference / allowed).max().item(),
        "failed_elements": failed.sum().item(),
    }


def series_inputs() -> torch.Tensor:
    t = torch.arange(128, dtype=torch.float32)
    return torch.stack(
        [
            (1 + 0.03 * index) * torch.sin(t / (7 + index % 5) + index / 3)
            + 0.004 * (index + 1) * t
            + 0.2 * torch.cos(t / (17 + index % 4))
            for index in range(12)
        ]
    )


def add_training_import_shim() -> bool:
    if importlib.util.find_spec("sklearn") is not None:
        return False
    sklearn = types.ModuleType("sklearn")
    model_selection = types.ModuleType("sklearn.model_selection")

    def unavailable(*args, **kwargs):
        raise AssertionError("training-only sklearn path invoked during inference")

    model_selection.train_test_split = unavailable
    sklearn.model_selection = model_selection
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.model_selection"] = model_selection
    return True


def forecast_with_states(model, context: torch.Tensor, batch_size: int, horizon: int):
    states = [[] for _ in range(12)]

    def state_hook(index):
        def hook(module, inputs, output):
            states[index].append(output[1].detach().cpu().clone())

        return hook

    hooks = [
        block.slstm_layer.slstm_cell.register_forward_hook(state_hook(index))
        for index, block in enumerate(model.blocks)
    ]
    quantiles, median = model.forecast(
        context,
        prediction_length=horizon,
        output_type="torch",
        batch_size=batch_size,
        full_rollout=False,
        dynamic_padding=False,
    )
    for hook in hooks:
        hook.remove()
    assert quantiles.shape == (context.shape[0], horizon, 9)
    assert median.shape == (context.shape[0], horizon)
    return quantiles, median, states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=1e-4)
    args = parser.parse_args()
    assert args.budget > 0 and not args.output.exists()
    assert not list(args.output.parent.glob(args.output.stem + "-b*-c*.pt"))
    torch.set_num_threads(1)

    weight = args.model / "model.ckpt"
    assert weight.stat().st_size == WEIGHT_BYTES and sha256(weight) == WEIGHT_SHA256
    pins = json.loads((args.source / "source-pins.json").read_text())
    assert pins["revision"] == SOURCE_REVISION
    for entry in pins["files"]:
        path = args.source / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert git_blob_sha(path) == entry["sha"]

    shim = add_training_import_shim()
    sys.path.insert(0, str(args.source / "src"))
    from tirex.models.tirex import TiRexZero

    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    assert checkpoint["hyper_parameters"]["train_ctx_len"] == 2048
    state = {
        name.replace("block_stack.", ""): tensor
        for name, tensor in checkpoint["state_dict"].items()
    }
    model = TiRexZero(backend="torch", **checkpoint["hyper_parameters"])
    assert len(state) == len(model.state_dict()) == 157
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    assert len(model.blocks) == 12
    assert sum(parameter.numel() for parameter in model.parameters()) == 35291200
    model.eval()

    metadata = {
        "status": "RUNNING",
        "model": "NX-AI/TiRex",
        "revision": REVISION,
        "source_revision": SOURCE_REVISION,
        "source_pins": pins,
        "weight_bytes": WEIGHT_BYTES,
        "weight_sha256": WEIGHT_SHA256,
        "torch": torch.__version__,
        "device": "cpu",
        "backend": "torch",
        "training_only_sklearn_import_shim": shim,
        "blocks": 12,
        "parameters": 35291200,
        "checkpoint_tensors": 157,
        "context_length": 128,
        "model_context_length": 2048,
        "forecast_horizon": 64,
        "quantiles": 9,
        "batches": [1, 2, 4],
        "cases_per_batch": 3,
        "budget": args.budget,
        "gate_sha256": sha256(Path(__file__)),
        "snapshots": [],
        "gpu_executed": False,
        "performance_claim": False,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    inputs = series_inputs()
    rows = []
    with torch.inference_mode(), args.output.open("x") as handle:
        for batch in (1, 2, 4):
            for case in range(3):
                context = inputs[4 * case : 4 * case + batch]
                quantiles, median, batch_states = forecast_with_states(
                    model, context, batch, 64
                )
                serial, serial_median, serial_states = forecast_with_states(
                    model, context, 1, 64
                )
                first, first_median, _ = forecast_with_states(model, context, batch, 32)
                padded = torch.cat(
                    (context, torch.full((batch, 32), float("nan"))), dim=1
                )
                second, second_median, _ = forecast_with_states(
                    model, padded, batch, 32
                )
                composed = torch.cat((first, second), dim=1)
                composed_median = torch.cat((first_median, second_median), dim=1)
                checks = {
                    "quantiles_batch_vs_serial": compare(
                        quantiles, serial, args.budget
                    ),
                    "median_batch_vs_serial": compare(
                        median, serial_median, args.budget
                    ),
                    "quantiles_two_patch_composition": compare(
                        quantiles, composed, args.budget
                    ),
                    "median_two_patch_composition": compare(
                        median, composed_median, args.budget
                    ),
                    "median_matches_q50": compare(
                        median, quantiles[:, :, 4], args.budget
                    ),
                }
                state_checks = []
                for layer in range(12):
                    assert len(batch_states[layer]) == 2
                    assert len(serial_states[layer]) == 2 * batch
                    for phase in range(2):
                        serialized = torch.cat(
                            [
                                serial_states[layer][2 * index + phase]
                                for index in range(batch)
                            ],
                            dim=1,
                        )
                        for state_index, name in enumerate(STATE_NAMES):
                            state_checks.append(
                                {
                                    "layer": layer,
                                    "phase": phase,
                                    "state": name,
                                    **compare(
                                        batch_states[layer][phase][state_index],
                                        serialized[state_index],
                                        args.budget,
                                    ),
                                }
                            )
                row = {
                    "batch": batch,
                    "case": case,
                    "checks": checks,
                    "state_checks": state_checks,
                    "state_pairs": len(state_checks),
                    "pass": all(check["pass"] for check in checks.values())
                    and all(check["pass"] for check in state_checks),
                    "bitwise": all(check["bitwise"] for check in checks.values())
                    and all(check["bitwise"] for check in state_checks),
                }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                snapshot = {
                    "context": context,
                    "quantiles": quantiles,
                    "median": median,
                    "serial_quantiles": serial,
                    "serial_median": serial_median,
                    "composed_quantiles": composed,
                    "composed_median": composed_median,
                    "batched_states": batch_states,
                    "serial_states": serial_states,
                }
                path = args.output.with_name(f"{args.output.stem}-b{batch}-c{case}.pt")
                partial = path.with_suffix(".pt.partial")
                assert not partial.exists()
                torch.save(snapshot, partial)
                partial.replace(path)
                metadata["snapshots"].append(
                    {
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "batch": batch,
                        "case": case,
                    }
                )
                meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "batch": batch,
                            "case": case,
                            "pass": row["pass"],
                            "bitwise": row["bitwise"],
                            "state_pairs": len(state_checks),
                        }
                    ),
                    flush=True,
                )
    metadata["status"] = (
        "PASS" if all(row["pass"] for row in rows) else "NUMERICAL_FAILED"
    )
    metadata["case_count"] = len(rows)
    metadata["state_pairs"] = sum(row["state_pairs"] for row in rows)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(0 if metadata["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
