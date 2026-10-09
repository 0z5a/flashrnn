"""Finite alternating-CUDA qualification with explicit BF16 cast boundaries."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from build_without_ninja import TaskBuilder

from flashrnn.flashrnn import cuda_init
from flashrnn.flashrnn.flashrnn import FlashRNNConfig, flashrnn


def reference(
    wx: torch.Tensor, r: torch.Tensor, b: torch.Tensor, initial: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float pointwise operations; stored recurrent products and states use input dtype."""
    state = initial[:, :, 0]
    history = []
    for x in wx.unbind(1):
        ry = torch.einsum("bhp,ghdp->bghd", state[0].float(), r.float()).to(wx.dtype)
        i, f, z, o = (x.float() + (ry.float() + b.float())).unbind(1)
        c = f.sigmoid() * state[1].float() + i.sigmoid() * z.tanh()
        h = o.sigmoid() * c.tanh()
        state = torch.stack((h, c)).to(initial.dtype)
        history.append(state)
    return torch.stack(history, 2), state.unsqueeze(2)


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    delta = actual - expected
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-30)).item(),
        "finite": torch.isfinite(actual).all().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(cuda_init.__file__).parent
    cuda_init._load = TaskBuilder(args.build_root, root).load
    os.environ["CUDA_LIB"] = str(Path(cuda_init.CUDA_HOME) / "lib64")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    config = FlashRNNConfig(
        function="lstm",
        backend="cuda",
        dtype=args.dtype,
        batch_size=16,
        hidden_dim=64,
        num_heads=1,
    )
    metadata = {
        "scope": "ALTERNATING_CUDA_LSTM_FORWARD_AND_GRADIENT_CHARACTERIZATION",
        "pid": os.getpid(),
        "started": time.time(),
        "status": "RUNNING",
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "matmul_tf32": False,
        "pointwise": "FP32; recurrent GEMM and stored states rounded to input dtype",
        "reference_limit": "Mathematical activations; CUDA fast-math is not emulated",
        "gradient_status": "ERROR_CHARACTERIZATION_ONLY_NO_ACCEPTANCE_BUDGET",
        "forward_contract": {"atol": 0.002, "rtol": 0.02, "max_abs": 0.003},
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    failed = 0
    with args.output.open("x") as handle:
        for steps in (1, 17, 128):
            shapes = (
                (16, steps, 4, 1, 64),
                (4, 1, 64, 64),
                (4, 1, 64),
                (2, 16, 1, 1, 64),
            )
            inputs = [(torch.randn(shape) * 0.05).to(dtype) for shape in shapes]
            inputs[1].div_(8)
            for loss_mode in ("final", "history_and_final"):
                cpu = [x.clone().requires_grad_() for x in inputs]
                gpu = [x.cuda().requires_grad_() for x in inputs]
                expected = reference(*cpu)
                actual = flashrnn(gpu[0], gpu[1], gpu[2], gpu[3], config=config)
                forward_errors = [error(a, b) for a, b in zip(actual, expected)]
                passed = all(
                    torch.isclose(a.detach().cpu(), b.detach(), atol=0.002, rtol=0.02)
                    .all()
                    .item()
                    and item["max_abs"] <= 0.003
                    for a, b, item in zip(actual, expected, forward_errors)
                )
                readonly = all(
                    torch.equal(a.detach().cpu(), b) for a, b in zip(gpu, inputs)
                )
                probes = [torch.randn(x.shape).to(dtype) for x in expected]
                losses = []
                for outputs in (expected, actual):
                    loss = (outputs[1] * probes[1].to(outputs[1].device)).float().sum()
                    if loss_mode == "history_and_final":
                        loss = (
                            loss
                            + (outputs[0] * probes[0].to(outputs[0].device))
                            .float()
                            .sum()
                        )
                    losses.append(loss)
                cpu_grads = torch.autograd.grad(losses[0], cpu)
                gpu_grads = torch.autograd.grad(losses[1], gpu)
                gradients = {
                    name: error(a, b)
                    for name, a, b in zip(
                        ("dWx", "dR", "db", "dinitial"), gpu_grads, cpu_grads
                    )
                }
                passed = (
                    passed and readonly and all(x["finite"] for x in gradients.values())
                )
                failed += int(not passed)
                row = {
                    "shape_B_T_H_D": [16, steps, 1, 64],
                    "loss": loss_mode,
                    "forward_errors": forward_errors,
                    "inputs_readonly": readonly,
                    "gradients": gradients,
                    "gradient_accuracy": "UNQUALIFIED",
                    "status": "FORWARD_PASS_GRADIENTS_FINITE" if passed else "FAILED",
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    metadata.update(
        finished=time.time(),
        failed_cases=failed,
        status="FAIL" if failed else "FORWARD_PASS_GRADIENTS_UNQUALIFIED",
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
