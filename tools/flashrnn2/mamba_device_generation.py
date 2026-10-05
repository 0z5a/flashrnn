"""Full Mamba generation parity with a trace built on the execution device."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from mamba_native_runtime import NativeMamba


def compare(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> dict:
    actual, expected = actual.cpu(), expected.cpu()
    delta = (actual - expected).abs()
    allowed = tolerance * (1 + expected.abs())
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    failed = ~finite | (delta > allowed)
    return {
        "shape": list(actual.shape),
        "finite": bool(finite.all()),
        "max_abs": float(delta.max()),
        "failed_elements": int(failed.sum()),
        "pass": not bool(failed.any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = json.loads(args.inputs.read_text())
    assert inputs["prompt_length"] == 128 and inputs["generated_tokens"] == 32
    cases = inputs["batches"][str(args.batch)]["input_ids"]
    assert len(cases) == 3
    assert all(
        len(c) == args.batch and all(len(row) == 128 for row in c) for c in cases
    )
    started = time.time()
    native = NativeMamba(args.model, args.source, args.config, args.device)
    config = native.config

    def blank() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(
                config.num_hidden_layers,
                args.batch,
                config.intermediate_size,
                config.conv_kernel,
                device=args.device,
            ),
            torch.zeros(
                config.num_hidden_layers,
                args.batch,
                config.intermediate_size,
                config.state_size,
                device=args.device,
            ),
        )

    metadata = {
        "scope": "FULL_NATIVE_MAMBA_DEVICE_TRACE_P128_G32_SERIAL_INTERLEAVED",
        "pid": os.getpid(),
        "started": started,
        "status": "RUNNING",
        "batch": args.batch,
        "device": args.device,
        "torch": torch.__version__,
        "parameters": sum(p.numel() for p in native.parameters()),
        "layers": len(native.backbone.layers),
        "weight_transport": native.metadata,
        "inputs": inputs,
        "logits_atol_rtol": 1e-3,
        "cache_atol_rtol": 1e-5,
        "trace_seed": 20261008,
        "trace_check": "Disabled built-in repetition; independently generated text cases below",
        "schedule": "Three request groups, one stream; serial and round-robin interleaved",
        "performance_claim": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(NativeMamba.run.__code__.co_filename),
                args.source,
                args.config,
                args.inputs,
            )
        },
    }
    if args.device == "cuda":
        metadata["gpu_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    snapshots, rows, generated = {}, [], {}
    with torch.inference_mode(), args.output.open("x") as handle:
        samples = {
            "prefill": (
                torch.randint(config.vocab_size, (args.batch, 128), device=args.device),
                *blank(),
            ),
            "decode": (
                torch.randint(config.vocab_size, (args.batch, 1), device=args.device),
                *blank(),
            ),
        }
        traced = torch.jit.trace_module(native, samples, check_trace=False)
        metadata["trace_finished"] = time.time()
        for schedule in ("serial", "interleaved"):
            states = {name: [blank() for _ in cases] for name in ("native", "traced")}
            next_ids = {
                name: [torch.tensor(c, device=args.device) for c in cases]
                for name in states
            }
            order = [(index, step) for index in range(3) for step in range(32)]
            if schedule == "interleaved":
                order.sort(key=lambda item: (item[1], item[0]))
            generated[schedule] = {name: [[] for _ in cases] for name in states}
            for index, step in order:
                outputs = {}
                for name, model in (("native", native), ("traced", traced)):
                    method = model.prefill if step == 0 else model.decode
                    output = method(next_ids[name][index], *states[name][index])
                    states[name][index] = output[1:]
                    outputs[name] = (output[0][:, -1], *output[1:])
                    next_ids[name][index] = output[0][:, -1].argmax(-1, keepdim=True)
                    generated[schedule][name][index].append(
                        next_ids[name][index][:, 0].cpu().tolist()
                    )
                checks = [
                    compare(a, b, 1e-3 if i == 0 else 1e-5)
                    for i, (a, b) in enumerate(
                        zip(outputs["traced"], outputs["native"], strict=True)
                    )
                ]
                tokens_equal = torch.equal(
                    next_ids["traced"][index], next_ids["native"][index]
                )
                row = {
                    "schedule": schedule,
                    "case": index,
                    "step": step,
                    "checks": checks,
                    "tokens_equal": tokens_equal,
                    "native_ids": generated[schedule]["native"][index][-1],
                    "traced_ids": generated[schedule]["traced"][index][-1],
                    "pass": tokens_equal and all(c["pass"] for c in checks),
                }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                # Complete first-group boundary tensors allow an independent witness audit.
                if schedule == "serial" and index == 0 and step in (0, 31):
                    snapshots[str(step)] = {
                        name: tuple(t.cpu().clone() for t in values)
                        for name, values in outputs.items()
                    }
            print(
                json.dumps(
                    {
                        "schedule": schedule,
                        "failed_steps": sum(not r["pass"] for r in rows),
                    }
                ),
                flush=True,
            )
    schedule_equal = {
        name: generated["serial"][name] == generated["interleaved"][name]
        for name in ("native", "traced")
    }
    tensor_path = args.output.with_suffix(".pt")
    torch.save(snapshots, tensor_path)
    with tensor_path.open("rb") as handle:
        tensor_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    failed = sum(not r["pass"] for r in rows)
    passed = not failed and all(schedule_equal.values())
    metadata.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        compared_batch_steps=len(rows),
        token_choices=len(rows) * args.batch,
        failed_steps=failed,
        serial_interleaved_tokens_equal=schedule_equal,
        tensor_sha256=tensor_sha,
        snapshot_scope="First serial group prefill/final last logits and complete caches, both arms",
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps({"status": metadata["status"], "failed_steps": failed}), flush=True
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
