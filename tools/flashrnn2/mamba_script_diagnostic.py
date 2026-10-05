"""Locate full-model numerical failures without changing their error budgets."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mamba_script_gate import errors, verified_metadata
from target_mamba_artifact import nodes

Output = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def details(actual: Output, expected: Output, contracts: dict) -> list[dict]:
    records = errors(actual, expected, contracts)
    for index, (record, value, target) in enumerate(zip(records, actual, expected)):
        budget = contracts["logits_contract" if index == 0 else "cache_contract"]
        ratio = (value - target).abs() / (
            budget["atol"] + budget["rtol"] * target.abs()
        )
        worst = tuple(int(i) for i in torch.unravel_index(ratio.argmax(), ratio.shape))
        failed = ~torch.isclose(value, target, **budget)
        record.update(
            failed_elements=int(failed.sum()),
            worst_normalized_error=float(ratio[worst]),
            worst_coordinate=list(worst),
            actual_at_worst=float(value[worst]),
            expected_at_worst=float(target[worst]),
        )
        if index > 0:
            record["failed_by_layer"] = failed.flatten(1).sum(1).tolist()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, contracts = verified_metadata(args.model), verified_metadata(args.goldens)
    assert source["batch"] == contracts["batch"] == 1
    assert source["prompt_length"] == contracts["prompt_length"] == 5
    model = torch.jit.load(str(args.model), map_location=args.device).eval()
    assert sum(p.numel() for p in model.parameters()) == source["parameters"]
    changes = 0
    for part in model.modules():
        for name in part._c._method_names():
            for node in nodes(part._c._get_method(name).graph):
                if (
                    node.kind() == "prim::Constant"
                    and str(node.output().type()) == "Device"
                    and str(node.output().toIValue()) != args.device
                ):
                    node.s_("value", args.device)
                    changes += 1
    cases = torch.load(args.goldens, map_location="cpu", weights_only=True)
    snapshots, rows = [], []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            conv = torch.zeros_like(case["prefill"][1], device=args.device)
            ssm = torch.zeros_like(case["prefill"][2], device=args.device)
            prefill = model.prefill(case["input_ids"].to(args.device), conv, ssm)
            prefix_copy = tuple(t.cpu().clone() for t in prefill)
            ids = prefill[0][:, -1].argmax(-1, keepdim=True)
            decode = model.decode(ids, prefill[1], prefill[2])
            decode_copy = tuple(t.cpu().clone() for t in decode)
            prefix_errors = details(prefix_copy, case["prefill"], contracts)
            decode_errors = details(decode_copy, case["decode"], contracts)
            token_match = torch.equal(ids.cpu(), case["next_ids"]) and torch.equal(
                decode_copy[0][:, -1].argmax(-1), case["decode"][0][:, -1].argmax(-1)
            )
            passed = token_match and all(
                r["pass"] for r in prefix_errors + decode_errors
            )
            rows.append(
                {
                    "case": index,
                    "prefill": prefix_errors,
                    "decode": decode_errors,
                    "token_match": token_match,
                    "status": "PASS" if passed else "CORRECTNESS_FAILED",
                }
            )
            snapshots.append({"prefill": prefix_copy, "decode": decode_copy})
            print(json.dumps(rows[-1]), flush=True)
    tensor_path = args.output.with_suffix(".pt")
    torch.save(snapshots, tensor_path)
    metadata = {
        "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
        "scope": "FULL_MODEL_NUMERICAL_DIAGNOSIS_NO_TIMING",
        "device": args.device,
        "torch": torch.__version__,
        "retargeted_device_literals": changes,
        "model": source,
        "oracles": contracts,
        "parameters": sum(p.numel() for p in model.parameters()),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(errors.__code__.co_filename),
                Path(nodes.__code__.co_filename),
            )
        },
        "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
    }
    if args.device == "cuda":
        metadata["device_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    raise SystemExit(0 if metadata["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
