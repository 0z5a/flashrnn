"""Paired complete-model Mamba2 cohorts after a same-device logit/cache gate."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from export_mamba2 import Mamba2Execution
from mamba2_checkpoint import load_mamba2
from mamba_generation_goldens import verified_checkpoint
from mamba_script_gate import verified_metadata
from mamba_serving import burst
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert args.verify_only or (args.device == "cuda" and args.blocks >= 20)
    assert len(set(args.concurrency)) == len(args.concurrency)
    assert all(c >= args.batch and c % args.batch == 0 for c in args.concurrency)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = verified_metadata(args.script)
    qualification = json.loads(args.qualification.read_text())
    manifest = verified_checkpoint(args.checkpoint)
    assert qualification["status"] == "PASS"
    assert qualification["device"] == args.device
    assert qualification["batch"] == args.batch
    assert args.verify_only or qualification.get("request_groups", 3) == 3
    assert qualification["compared_steps"] == 64 * qualification.get(
        "request_groups", 3
    )
    assert qualification["failed_steps"] == 0
    assert qualification["schedule_parity"] == {"native": True, "traced": True}
    assert qualification["script_artifact_sha256"] == source["artifact_sha256"]
    assert qualification["input_sha256"] == digest(args.inputs)
    assert qualification["checkpoint_manifest"] == manifest
    assert source["checkpoint_manifest"] == manifest
    inputs = json.loads(args.inputs.read_text())
    assert inputs["prompt_length"] == 128 and inputs["generated_tokens"] == 32
    groups = inputs["batches"][str(args.batch)]["input_ids"]
    assert len(groups) == 3 and all(len(group) == args.batch for group in groups)
    model = load_mamba2(args.checkpoint).to(args.device).eval()
    native = Mamba2Execution(model).eval()
    script = torch.jit.load(str(args.script), map_location=args.device).eval()
    assert sum(p.numel() for p in native.parameters()) == source["parameters"]
    assert sum(p.numel() for p in script.parameters()) == source["parameters"]
    cases = []
    with torch.inference_mode():
        for group in groups:
            blank = Mamba2Cache(model.config, args.batch, torch.float32, "cpu")
            initial = (blank.conv_states, blank.ssm_states)
            state = tuple(value.to(args.device) for value in initial)
            ids = torch.tensor(group, device=args.device)
            tokens = []
            for step in range(32):
                output = (native.prefill if step == 0 else native.decode)(ids, *state)
                state = output[1:]
                ids = output[0][:, -1].argmax(-1, keepdim=True)
                tokens.append(ids[:, 0].cpu())
            cases.append(
                {
                    "input_ids": torch.tensor(group),
                    "prefill_cache": initial,
                    "generated_ids": torch.stack(tokens),
                    "final_cache": tuple(value.cpu() for value in state),
                }
            )
    arms = {
        "native_transformers": (native.prefill, native.decode),
        "torchscript": (script.prefill, script.decode),
    }
    metadata = {
        "status": "RUNNING",
        "started": time.time(),
        "scope": "QUEUED_TOKEN_IDS_TO_HOST_VISIBLE_TOKENS",
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint_manifest": manifest,
        "script_artifact_sha256": source["artifact_sha256"],
        "input_sha256": digest(args.inputs),
        "qualification_sha256": digest(args.qualification),
        "source_sha256": {
            Path(path).name: digest(Path(path))
            for path in (
                Path(__file__),
                Path(burst.__code__.co_filename),
                Path(Mamba2Execution._run.__code__.co_filename),
                Path(load_mamba2.__code__.co_filename),
            )
        },
        "baseline_name": "native_transformers",
        "candidate_name": "torchscript",
        "arm_order": list(arms),
        "oracles": {
            "batch": args.batch,
            "generated_tokens": 32,
            "cache_contract": {"atol": 1e-5, "rtol": 1e-5},
        },
        "concurrency": args.concurrency,
        "verification_only": args.verify_only,
        "paired_blocks": 0 if args.verify_only else args.blocks,
        "fast_kernel_qualified": False,
        "performance_claim": not args.verify_only,
        "timed": "Queue wait, state allocation, prompt copy, full prefill, 31 decode steps, argmax, host-visible token delivery",
        "excluded": "Weight loading, golden generation, tokenization, network, correctness comparisons and JSON output",
        "arrival_policy": "All C requests at time zero; B-sized static groups, single worker, round-robin decode",
        "statistics": "Within-process AB/BA pairs; finite-cohort p99 is not a steady-state SLO",
    }
    if args.device == "cuda":
        metadata["gpu_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
        assert metadata["gpu_uuid"] == qualification["gpu_uuid"]
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    with torch.inference_mode(), args.output.open("x") as handle:
        for concurrency in args.concurrency:
            for label, (prefill, decode) in arms.items():
                row = burst(
                    prefill,
                    decode,
                    cases,
                    metadata["oracles"],
                    concurrency,
                    args.device,
                    True,
                )
                row.update(arm=label, phase="qualification_and_warmup")
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                assert row["status"] == "PASS"
            if args.verify_only:
                continue
            for block in range(args.blocks):
                order = list(arms) if block % 2 == 0 else list(reversed(arms))
                for label in order:
                    row = burst(
                        *arms[label],
                        cases,
                        metadata["oracles"],
                        concurrency,
                        args.device,
                        False,
                    )
                    row.update(arm=label, phase="measurement", block=block, order=order)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    assert row["status"] == "PASS"
            print(
                json.dumps(
                    {
                        "batch": args.batch,
                        "concurrency": concurrency,
                        "paired_blocks": args.blocks,
                    }
                ),
                flush=True,
            )
    metadata.update(status="PASS", finished=time.time())
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
