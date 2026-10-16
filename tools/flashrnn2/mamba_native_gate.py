"""Compare the native Torch fallback and traced model on the same device/runtime."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from mamba_native_runtime import NativeMamba
from mamba_script_diagnostic import details
from mamba_script_gate import verified_metadata
from target_mamba_artifact import nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    native = NativeMamba(args.model, args.source, args.config, args.device)
    original = verified_metadata(args.goldens)
    cases = torch.load(args.goldens, map_location="cpu", weights_only=True)
    assert len(cases) == 3 and original["batch"] == 1 and original["prompt_length"] == 5
    checkpoint_verified = False
    if args.checkpoint is not None:
        from safetensors.torch import load_file

        assert args.device == "cpu"
        with args.checkpoint.open("rb") as handle:
            assert (
                hashlib.file_digest(handle, "sha256").hexdigest()
                == "1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea"
            )
        weights = load_file(args.checkpoint)
        state = native.state_dict()
        assert set(state) - set(weights) == {"lm_head.weight"}
        assert not set(weights) - set(state)
        assert all(torch.equal(value, state[name]) for name, value in weights.items())
        checkpoint_verified = True
    traced = torch.jit.load(str(args.model), map_location=args.device).eval()
    retargeted = 0
    for part in traced.modules():
        for name in part._c._method_names():
            for node in nodes(part._c._get_method(name).graph):
                if (
                    node.kind() == "prim::Constant"
                    and str(node.output().type()) == "Device"
                    and str(node.output().toIValue()) != args.device
                ):
                    node.s_("value", args.device)
                    retargeted += 1
    rows, snapshots = [], []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            outputs = {}
            for name, model in (("native", native), ("traced", traced)):
                conv = torch.zeros_like(case["prefill"][1], device=args.device)
                ssm = torch.zeros_like(case["prefill"][2], device=args.device)
                prefill = model.prefill(case["input_ids"].to(args.device), conv, ssm)
                prefix = tuple(t.cpu().clone() for t in prefill)
                ids = prefill[0][:, -1].argmax(-1, keepdim=True)
                decode = model.decode(ids, prefill[1], prefill[2])
                outputs[name] = {
                    "prefill": prefix,
                    "decode": tuple(t.cpu().clone() for t in decode),
                    "next_ids": ids.cpu().clone(),
                }
            comparisons = {}
            for label, left, right in (
                ("traced_vs_same_device_native", outputs["traced"], outputs["native"]),
                ("native_vs_original_oracle", outputs["native"], case),
            ):
                checks = {
                    phase: details(left[phase], right[phase], original)
                    for phase in ("prefill", "decode")
                }
                tokens = torch.equal(
                    left["next_ids"], right["next_ids"]
                ) and torch.equal(
                    left["decode"][0][:, -1].argmax(-1),
                    right["decode"][0][:, -1].argmax(-1),
                )
                comparisons[label] = {
                    "checks": checks,
                    "tokens_equal": tokens,
                    "pass": tokens
                    and all(e["pass"] for values in checks.values() for e in values),
                }
            row = {"case": index, **comparisons}
            rows.append(row)
            snapshots.append(outputs["native"])
            print(json.dumps(row), flush=True)
    tensor_path = args.output.with_suffix(".pt")
    torch.save(snapshots, tensor_path)
    with tensor_path.open("rb") as handle:
        tensor_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "scope": "FULL_MODEL_NATIVE_TORCH_FALLBACK_AND_TRACED_SAME_DEVICE",
        "started": started,
        "finished": time.time(),
        "torch": torch.__version__,
        "device": args.device,
        "model": native.metadata,
        "oracles": original,
        "checkpoint_tensor_identity_verified": checkpoint_verified,
        "native_parameters": sum(p.numel() for p in native.parameters()),
        "native_layers": len(native.backbone.layers),
        "native_method_execution": "Unchanged selected Transformers eager methods; serialized model used only to transport weights",
        "retargeted_traced_device_constants": retargeted,
        "same_device_native_passed_cases": sum(
            r["traced_vs_same_device_native"]["pass"] for r in rows
        ),
        "native_vs_original_oracle_passed_cases": sum(
            r["native_vs_original_oracle"]["pass"] for r in rows
        ),
        "original_oracle_failures_replaced": False,
        "tensor_sha256": tensor_sha,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(NativeMamba.run.__code__.co_filename),
                args.source,
                args.config,
            )
        },
        "performance_claim": False,
    }
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    raise SystemExit(0 if metadata["same_device_native_passed_cases"] == 3 else 1)


if __name__ == "__main__":
    main()
