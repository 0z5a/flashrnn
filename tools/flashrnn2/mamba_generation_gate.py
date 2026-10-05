"""Complete-model generation and request-isolation gate, without timing claims."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from mamba_native_runtime import NativeMamba
from mamba_script_gate import errors, verified_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--allow-batch-change", action="store_true")
    parser.add_argument("--backend", choices=("script", "native"), default="script")
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--native-config", type=Path)
    args = parser.parse_args()
    if args.backend == "native" and (
        args.native_source is None or args.native_config is None
    ):
        parser.error("native execution requires --native-source and --native-config")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = verified_metadata(args.model)
    oracle = verified_metadata(args.goldens)
    cases = torch.load(args.goldens, map_location="cpu", weights_only=True)
    if args.backend == "native":
        assert oracle.get("model_family", "mamba") == "mamba"
        model = NativeMamba(
            args.model, args.native_source, args.native_config, args.device
        )
        batch_contract = "NATIVE_DYNAMIC_INPUTS_ARTIFACT_IS_WEIGHT_TRANSPORT_ONLY"
    else:
        assert source["batch"] == oracle["batch"] or args.allow_batch_change
        assert source["prompt_length"] == oracle["prompt_length"]
        model = torch.jit.load(str(args.model), map_location=args.device).eval()
        batch_contract = (
            "MATCHES_TRACE"
            if source["batch"] == oracle["batch"]
            else "CROSS_BATCH_PROBE"
        )
    assert sum(p.numel() for p in model.parameters()) == oracle["parameters"]
    metadata = {
        "scope": "COMPLETE_MAMBA_FIXED_LENGTH_GENERATION_AND_REQUEST_ISOLATION",
        "model_family": oracle.get("model_family", "mamba"),
        "backend": args.backend,
        "batch_contract": batch_contract,
        "pid": os.getpid(),
        "started": time.time(),
        "status": "RUNNING",
        "torch": torch.__version__,
        "device": args.device,
        "model": source,
        "oracles": oracle,
        "schedule_contract": "Serial and round-robin interleaved request groups on one execution stream. This checks cache isolation, not parallel stream execution or serving throughput.",
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_helper_sha256": hashlib.sha256(
            Path(errors.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "performance_claim": False,
    }
    if args.backend == "native":
        metadata["native_source_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.native_source,
                args.native_config,
                Path(NativeMamba.run.__code__.co_filename),
            )
        }
    if args.device == "cuda":
        metadata["device_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    failed = 0
    with torch.inference_mode(), args.output.open("x") as handle:
        for schedule in ("serial", "interleaved"):
            states = [
                (
                    torch.zeros_like(c["prefill_cache"][0], device=args.device),
                    torch.zeros_like(c["prefill_cache"][1], device=args.device),
                )
                for c in cases
            ]
            next_ids = [c["input_ids"].to(args.device) for c in cases]
            order = [
                (i, s)
                for i in range(len(cases))
                for s in range(oracle["generated_tokens"])
            ]
            if schedule == "interleaved":
                order.sort(key=lambda pair: (pair[1], pair[0]))
            for index, step in order:
                case = cases[index]
                method = model.prefill if step == 0 else model.decode
                output = method(next_ids[index], *states[index])
                states[index] = (output[1], output[2])
                actual_logits = output[0][:, -1]
                next_ids[index] = actual_logits.argmax(-1, keepdim=True)
                token_match = torch.equal(
                    next_ids[index][:, 0].cpu(), case["generated_ids"][step]
                )
                comparisons = errors((actual_logits,), (case["logits"][step],), oracle)
                if step in (0, oracle["generated_tokens"] - 1):
                    key = "prefill_cache" if step == 0 else "final_cache"
                    comparisons = errors(
                        (actual_logits, *states[index]),
                        (case["logits"][step], *case[key]),
                        oracle,
                    )
                passed = token_match and all(item["pass"] for item in comparisons)
                failed += int(not passed)
                row = {
                    "schedule": schedule,
                    "case": index,
                    "step": step,
                    "errors": comparisons,
                    "greedy_token_match": token_match,
                    "status": "PASS" if passed else "CORRECTNESS_FAILED",
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            print(
                json.dumps({"schedule": schedule, "failed_steps_so_far": failed}),
                flush=True,
            )
    metadata.update(
        status="FAIL" if failed else "PASS",
        failed_steps=failed,
        compared_steps=2 * len(cases) * oracle["generated_tokens"],
        finished=time.time(),
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
