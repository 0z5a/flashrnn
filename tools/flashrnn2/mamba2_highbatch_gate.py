"""Mamba2 full-model high-batch generation parity on fixed WikiText requests."""

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
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> dict:
    actual, expected = actual.cpu(), expected.cpu()
    delta = (actual - expected).abs()
    failures = ~torch.isfinite(actual) | ~torch.isfinite(expected)
    failures |= delta > tolerance * (1 + expected.abs())
    return {
        "shape": list(actual.shape),
        "max_abs": float(delta.max()),
        "failed_elements": int(failures.sum()),
        "pass": not bool(failures.any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--groups", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = verified_metadata(args.script)
    assert source["model_family"] == "mamba2" and source["prompt_length"] == 128
    checkpoint = verified_checkpoint(args.checkpoint)
    native_model = load_mamba2(args.checkpoint).to(args.device).eval()
    native = Mamba2Execution(native_model).eval()
    traced = torch.jit.load(str(args.script), map_location=args.device).eval()
    assert sum(p.numel() for p in native.parameters()) == source["parameters"]
    inputs = json.loads(args.inputs.read_text())
    assert inputs["prompt_length"] == 128 and inputs["generated_tokens"] == 32
    cases = inputs["batches"][str(args.batch)]["input_ids"]
    assert len(cases) == 3 and all(len(group) == args.batch for group in cases)
    cases = cases[: args.groups]
    started = time.time()
    metadata = {
        "status": "RUNNING",
        "batch": args.batch,
        "request_groups": args.groups,
        "device": args.device,
        "prompt_length": 128,
        "generated_tokens": 32,
        "started": started,
        "torch": torch.__version__,
        "parameters": source["parameters"],
        "layers": len(native_model.backbone.layers),
        "script_artifact_sha256": source["artifact_sha256"],
        "script_trace_batch": source["batch"],
        "checkpoint_manifest": checkpoint,
        "input_sha256": digest(args.inputs),
        "source_sha256": {
            path.name: digest(path)
            for path in (
                Path(__file__),
                Path(Mamba2Execution._run.__code__.co_filename),
                Path(load_mamba2.__code__.co_filename),
            )
        },
        "logits_atol_rtol": 1e-3,
        "cache_atol_rtol": 1e-5,
        "performance_claim": False,
    }
    if args.device == "cuda":
        metadata["gpu_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def blank() -> tuple[torch.Tensor, torch.Tensor]:
        cache = Mamba2Cache(
            native_model.config, args.batch, torch.float32, args.device
        )
        return cache.conv_states, cache.ssm_states

    rows, snapshots, generated = [], {}, {}
    with torch.inference_mode(), args.output.open("x") as handle:
        for schedule in ("serial", "interleaved"):
            states = {arm: [blank() for _ in cases] for arm in ("native", "traced")}
            next_ids = {
                arm: [torch.tensor(group, device=args.device) for group in cases]
                for arm in states
            }
            order = [(case, step) for case in range(args.groups) for step in range(32)]
            if schedule == "interleaved":
                order.sort(key=lambda item: (item[1], item[0]))
            generated[schedule] = {arm: [[] for _ in cases] for arm in states}
            for index, step in order:
                outputs = {}
                for arm, model in (("native", native), ("traced", traced)):
                    method = model.prefill if step == 0 else model.decode
                    output = method(next_ids[arm][index], *states[arm][index])
                    states[arm][index] = output[1:]
                    outputs[arm] = (output[0][:, -1], *output[1:])
                    next_ids[arm][index] = outputs[arm][0].argmax(-1, keepdim=True)
                    generated[schedule][arm][index].append(
                        next_ids[arm][index][:, 0].cpu().tolist()
                    )
                count = 3 if step in (0, 31) else 1
                checks = [
                    compare(actual, expected, 1e-3 if position == 0 else 1e-5)
                    for position, (actual, expected) in enumerate(
                        zip(outputs["traced"][:count], outputs["native"][:count], strict=True)
                    )
                ]
                row = {
                    "schedule": schedule,
                    "case": index,
                    "step": step,
                    "checks": checks,
                    "native_ids": generated[schedule]["native"][index][-1],
                    "traced_ids": generated[schedule]["traced"][index][-1],
                }
                row["pass"] = row["native_ids"] == row["traced_ids"] and all(
                    check["pass"] for check in checks
                )
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if schedule == "serial" and index == 0 and step in (0, 31):
                    snapshots[str(step)] = {
                        arm: tuple(t.cpu().clone() for t in values)
                        for arm, values in outputs.items()
                    }
            print(json.dumps({"schedule": schedule, "rows": len(rows)}), flush=True)
    parity = {
        arm: generated["serial"][arm] == generated["interleaved"][arm]
        for arm in ("native", "traced")
    }
    snapshot_path = args.output.with_suffix(".pt")
    torch.save(snapshots, snapshot_path)
    failed = sum(not row["pass"] for row in rows)
    metadata.update(
        status="PASS" if failed == 0 and all(parity.values()) else "FAIL",
        finished=time.time(),
        compared_steps=len(rows),
        token_choices=len(rows) * args.batch,
        failed_steps=failed,
        schedule_parity=parity,
        snapshot_sha256=digest(snapshot_path),
    )
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": metadata["status"], "failed_steps": failed}), flush=True)
    raise SystemExit(0 if metadata["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
