"""Full-model CUDA qualification using only the existing Torch runtime."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch


def verified_metadata(path: Path) -> dict:
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    with path.open("rb") as handle:
        assert (
            hashlib.file_digest(handle, "sha256").hexdigest()
            == metadata["artifact_sha256"]
        )
    return metadata


def errors(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    contracts: dict,
) -> list[dict]:
    result = []
    for name, output, target in zip(
        ("logits", "conv_cache", "ssm_cache"), actual, expected
    ):
        output = output.cpu()
        contract = contracts[
            "logits_contract" if name == "logits" else "cache_contract"
        ]
        difference = (output - target).abs()
        result.append(
            {
                "tensor": name,
                "max_abs": difference.max().item(),
                "relative_l2": (
                    difference.norm() / target.norm().clamp_min(1e-30)
                ).item(),
                "pass": torch.isclose(output, target, **contract).all().item(),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--goldens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = verified_metadata(args.model)
    contracts = verified_metadata(args.goldens)
    assert source["batch"] == contracts["batch"] == 1
    assert source["prompt_length"] == contracts["prompt_length"] == 5
    model = torch.jit.load(str(args.model), map_location="cuda").eval()
    cases = torch.load(args.goldens, map_location="cpu", weights_only=True)
    metadata = {
        "scope": "FULL_MAMBA130M_TORCHSCRIPT_CUDA_PREFILL_DECODE_QUALIFICATION",
        "pid": os.getpid(),
        "started": time.time(),
        "status": "RUNNING",
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": source,
        "oracles": contracts,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matmul_tf32": False,
        "cudnn_tf32": False,
    }
    assert metadata["parameters"] == source["parameters"]
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    failed = 0
    with args.output.open("x") as handle, torch.inference_mode():
        for index, case in enumerate(cases):
            ids = case["input_ids"].cuda()
            conv = torch.zeros_like(case["prefill"][1], device="cuda")
            ssm = torch.zeros_like(case["prefill"][2], device="cuda")
            prefill = model.prefill(ids, conv, ssm)
            prefill_errors = errors(prefill, case["prefill"], contracts)
            next_ids = prefill[0][:, -1].argmax(-1, keepdim=True)
            token_match = torch.equal(next_ids.cpu(), case["next_ids"])
            decode = model.decode(next_ids, prefill[1], prefill[2])
            decode_errors = errors(decode, case["decode"], contracts)
            decoded_match = torch.equal(
                decode[0][:, -1].argmax(-1).cpu(),
                case["decode"][0][:, -1].argmax(-1),
            )
            passed = (
                token_match
                and decoded_match
                and all(item["pass"] for item in prefill_errors + decode_errors)
            )
            failed += int(not passed)
            row = {
                "case": index,
                "prefill": prefill_errors,
                "decode": decode_errors,
                "greedy_token_match": token_match and decoded_match,
                "status": "PASS" if passed else "CORRECTNESS_FAILED",
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    metadata.update(
        status="FAIL" if failed else "PASS", failed_cases=failed, finished=time.time()
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
