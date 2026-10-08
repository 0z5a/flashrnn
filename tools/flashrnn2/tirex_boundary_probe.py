"""Isolate the first TiRex C32 recurrence differences without timing."""

import argparse
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from tirex_e2e import contexts
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
DIAGNOSTIC_SHA256 = "0de5167a9a6f3ee89a468370e2d0f40886957195dc75c8145786d19fa6fd64aa"
TARGETS = (
    (1, 3, 1, 61),
    (4, 3, 1, 59),
    (6, 10, 1, 61),
    (7, 9, 1, 61),
    (8, 9, 0, 61),
    (11, 3, 0, 61),
)
STATE_NAMES = ("hidden", "cell", "normalizer", "stabilizer")


def capture_boundary(model, context, layer):
    inputs = []

    def capture(module, args):
        gates, state = args
        inputs.append(
            (
                gates.detach().cpu().clone(),
                None if state is None else state.detach().cpu().clone(),
            )
        )

    cell = model.blocks[layer].slstm_layer.slstm_cell
    hook = cell.register_forward_pre_hook(capture)
    try:
        output = model.forecast(
            context,
            prediction_length=64,
            output_type="torch",
            batch_size=2,
            full_rollout=False,
            dynamic_padding=False,
        )
    finally:
        hook.remove()
    assert len(inputs) == 2
    return tuple(value.detach().cpu().clone() for value in output), inputs


def compare_cell(records, scope, group, reference, candidate):
    for name, expected, actual in (
        ("output", reference[0], candidate[0]),
        *(
            (name, reference[1][i], candidate[1][i])
            for i, name in enumerate(STATE_NAMES)
        ),
    ):
        records.append(
            {"group": group, "scope": scope, "tensor": name}
            | compare(actual, expected, BUDGET)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.output.with_suffix(".pt")
    assert not args.output.exists() and not raw_path.exists()

    gate = json.loads(args.gate.read_text())
    audit = json.loads(args.audit.read_text())
    prior = json.loads(args.diagnostic.read_text())
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
    assert sha256(args.diagnostic) == DIAGNOSTIC_SHA256
    assert prior["status"] == "FAIL_NUMERICAL" and len(prior["comparisons"]) == 2048
    assert prior["budget"] == BUDGET and prior["fixture_sha256"] == FIXTURE_SHA256
    prior_flags = {
        (row["group"], row["tensor"]): row["pass"]
        for row in prior["comparisons"]
        if row["scope"] == "forecast_unhooked"
    }

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
    reference = TiRexZero(backend="torch", **checkpoint["hyper_parameters"])
    reference.load_state_dict(
        {
            name.replace("block_stack.", ""): value
            for name, value in checkpoint["state_dict"].items()
        },
        strict=True,
    )
    reference.to("cuda").eval()
    candidate = copy.deepcopy(reference)
    install_tirex_cells(candidate)
    del checkpoint

    fixtures = contexts(32, 2)
    fixture_sha = hashlib.sha256(torch.cat(fixtures).numpy().tobytes()).hexdigest()
    assert fixture_sha == FIXTURE_SHA256
    raw = {"targets": []}
    records = []
    with torch.inference_mode():
        for group, layer, patch, step in TARGETS:
            entry = {
                "group": group,
                "layer": layer,
                "patch": patch,
                "step": step,
                "context": fixtures[group].clone(),
            }
            reference_forecast, reference_inputs = capture_boundary(
                reference, fixtures[group], layer
            )
            candidate_forecast, candidate_inputs = capture_boundary(
                candidate, fixtures[group], layer
            )
            entry["forecast"] = {
                "reference": reference_forecast,
                "candidate": candidate_forecast,
            }
            entry["boundary"] = {
                "reference": reference_inputs[patch],
                "candidate": candidate_inputs[patch],
            }
            for name, expected, actual in zip(
                ("quantiles", "median"), reference_forecast, candidate_forecast
            ):
                row = {"group": group, "scope": "forecast", "tensor": name} | compare(
                    actual, expected, BUDGET
                )
                row["matches_prior_unhooked_flag"] = (
                    row["pass"] == prior_flags[group, name]
                )
                records.append(row)

            reference_gates, reference_initial = reference_inputs[patch]
            candidate_gates, candidate_initial = candidate_inputs[patch]
            assert reference_gates.shape == candidate_gates.shape
            state_equal = reference_initial is None and candidate_initial is None
            if reference_initial is not None and candidate_initial is not None:
                state_equal = torch.equal(reference_initial, candidate_initial)
            entry["boundary_equal"] = {
                "gates": torch.equal(reference_gates, candidate_gates),
                "state": state_equal,
            }
            reference_cell = reference.blocks[layer].slstm_layer.slstm_cell
            candidate_cell = candidate.blocks[layer].slstm_layer.slstm_cell
            gates = reference_gates.to("cuda")
            initial = (
                None if reference_initial is None else reference_initial.to("cuda")
            )
            prefix_reference = reference_cell(gates[:, :step], initial)
            prefix_candidate = candidate_cell(gates[:, :step], initial)
            entry["prefix"] = {
                "reference": tuple(
                    value.detach().cpu().clone() for value in prefix_reference
                ),
                "candidate": tuple(
                    value.detach().cpu().clone() for value in prefix_candidate
                ),
            }
            compare_cell(records, "prefix", group, prefix_reference, prefix_candidate)
            isolated_reference = reference_cell(
                gates[:, step : step + 1], prefix_reference[1]
            )
            isolated_candidate = candidate_cell(
                gates[:, step : step + 1], prefix_reference[1]
            )
            entry["isolated_step"] = {
                "reference": tuple(
                    value.detach().cpu().clone() for value in isolated_reference
                ),
                "candidate": tuple(
                    value.detach().cpu().clone() for value in isolated_candidate
                ),
            }
            compare_cell(
                records, "isolated_step", group, isolated_reference, isolated_candidate
            )
            raw["targets"].append(entry)
    assert len(raw["targets"]) == 6 and len(records) == 6 * 12
    torch.save(raw, raw_path)
    args.output.write_text(
        json.dumps(
            {
                "status": "DIAGNOSTIC_COMPLETE",
                "scope": "Six C32 boundary captures and shared-state one-step controls; no timing",
                "budget": BUDGET,
                "fixture_sha256": fixture_sha,
                "checkpoint_sha256": WEIGHT_SHA256,
                "source_revision": SOURCE_REVISION,
                "adapter_sha256": adapter_sha,
                "qualification_gate_sha256": sha256(args.gate),
                "qualification_audit_sha256": sha256(args.audit),
                "previous_diagnostic_sha256": DIAGNOSTIC_SHA256,
                "raw_sha256": sha256(raw_path),
                "training_only_sklearn_import_shim": shim,
                "targets": [
                    {
                        k: v
                        for k, v in entry.items()
                        if k in ("group", "layer", "patch", "step", "boundary_equal")
                    }
                    for entry in raw["targets"]
                ],
                "comparisons": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
