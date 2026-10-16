"""Verify that frozen model-logit budgets reject cross-request and stale caches."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from mamba_script_gate import errors, verified_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = verified_metadata(args.model)
    oracle = verified_metadata(args.goldens)
    assert source["batch"] == oracle["batch"]
    assert source["prompt_length"] == oracle["prompt_length"]
    cases = torch.load(args.goldens, map_location="cpu", weights_only=True)
    assert len(cases) >= 2 and oracle["generated_tokens"] >= 2
    model = torch.jit.load(str(args.model), map_location=args.device).eval()
    started = time.time()
    records = []
    with torch.inference_mode():
        prefixes = []
        for case in cases[:2]:
            states = [
                torch.zeros_like(x, device=args.device) for x in case["prefill_cache"]
            ]
            output = model.prefill(case["input_ids"].to(args.device), *states)
            prefixes.append((output[1], output[2]))
        conv, ssm = prefixes[0]
        variants = {
            "unmodified": (conv, ssm),
            "other_request_conv": (prefixes[1][0], ssm),
            "other_request_ssm": (conv, prefixes[1][1]),
            "stale_zero_cache": (torch.zeros_like(conv), torch.zeros_like(ssm)),
        }
        ids = cases[0]["generated_ids"][0, :, None].to(args.device)
        for name, states in variants.items():
            output = model.decode(ids, *(x.clone() for x in states))
            comparison = errors((output[0][:, -1],), (cases[0]["logits"][1],), oracle)[
                0
            ]
            expected_pass = name == "unmodified"
            records.append(
                {
                    "control": name,
                    "logits": comparison,
                    "expected_acceptance": expected_pass,
                    "control_pass": comparison["pass"] == expected_pass,
                }
            )
    result = {
        "scope": "COMPLETE_MAMBA_CACHE_FAULT_DETECTION_AT_FIRST_DECODE",
        "status": "PASS" if all(row["control_pass"] for row in records) else "FAIL",
        "device": args.device,
        "torch": torch.__version__,
        "started": started,
        "finished": time.time(),
        "records": records,
        "model_sha256": source["artifact_sha256"],
        "oracles": oracle,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_helper_sha256": hashlib.sha256(
            Path(errors.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "performance_claim": False,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "records": records}), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
