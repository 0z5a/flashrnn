"""Device-selectable full-state/gradient qualification of the public Torch API."""

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from flashrnn import flashrnn, flashrnn_torch
from flashrnn.flashrnn2.reference import SIZES, Cell, recurrence
from flashrnn.flashrnn2.torch_backend import Numerics


@dataclass(frozen=True)
class Case:
    name: str
    cell: Cell
    shape: tuple[int, int, int, int]
    numerics: Numerics
    initial: str = "nonzero"
    mathematical_dtype: torch.dtype = torch.float64


def fixtures(case: Case, seed: int) -> tuple[torch.Tensor, ...]:
    batch, steps, heads, width = case.shape
    gates, biases, states = SIZES[case.cell]
    generator = torch.Generator().manual_seed(seed)
    shapes = (
        (batch, steps, gates, heads, width),
        (gates, heads, width, width),
        (biases, heads, width),
        (states, batch, 1, heads, width),
    )
    tensors = tuple(
        torch.randn(shape, generator=generator, dtype=torch.float64) * scale
        for shape, scale in zip(
            shapes, (0.1, 0.1 / width**0.5, 0.05, 0.05), strict=True
        )
    )
    if case.cell == "slstm":
        tensors[3][2].abs_().add_(1)
        if case.initial == "zero":
            tensors[3].zero_()
        elif case.initial == "mixed":
            tensors[3][2, 0].zero_()
        elif case.initial == "saturated":
            tensors[0][:, :, 0].add_(30)
            tensors[0][:, :, 1].sub_(30)
    dtype = (
        case.mathematical_dtype if case.numerics == "mathematical" else torch.float32
    )
    return tuple(t.to(dtype) for t in tensors)


def loss(outputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    history, final = outputs
    coefficients = torch.linspace(-1, 1, history.numel(), dtype=history.dtype)
    return (
        history * coefficients.to(history.device).view_as(history)
    ).sum() + final.square().sum()


def measure(actual: torch.Tensor, expected: torch.Tensor, budget: float) -> dict:
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    assert actual.shape == expected.shape
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = (actual - expected).abs()
    passed = finite and bool((difference <= budget + budget * expected.abs()).all())
    return {
        "pass": passed,
        "finite": finite,
        "max_abs": float(difference.max()) if finite else None,
        "rms": float(difference.square().mean().sqrt()) if finite else None,
        "relative_l2": float(difference.norm() / expected.norm().clamp_min(1e-30))
        if finite
        else None,
        "max_coordinate": [
            int(i) for i in torch.unravel_index(difference.argmax(), difference.shape)
        ]
        if finite
        else [],
        "failed_elements": int((difference > budget + budget * expected.abs()).sum())
        if finite
        else None,
    }


def run_case(case: Case, seed: int, device: torch.device) -> tuple[dict, dict]:
    originals = fixtures(case, seed)
    actual_inputs = tuple(
        t.to(device).detach().clone().requires_grad_() for t in originals
    )
    mathematical = case.numerics == "mathematical"
    oracle_inputs = tuple(
        t.clone().to(torch.float64 if mathematical else torch.float32).requires_grad_()
        for t in originals
    )
    exact_dtype = mathematical and case.mathematical_dtype == torch.float64
    forward_budget, gradient_budget = (1e-12, 1e-11) if exact_dtype else (1e-5, 1e-5)
    if mathematical:
        oracle = flashrnn(
            *oracle_inputs[:3],
            states=oracle_inputs[3],
            function=case.cell,
            backend="vanilla",
            dtype="float32",
        )
    else:
        wx, r, b, initial = oracle_inputs
        # Separate one-step calls place a fresh Torch weight cast at each step.
        history = []
        carry = initial
        for step in range(wx.shape[1]):
            state, carry = recurrence(
                wx[:, step : step + 1],
                r.bfloat16().float(),
                b,
                carry,
                cell=case.cell,
                mma_dtype=torch.bfloat16,
            )
            history.append(state)
        oracle = torch.cat(history, dim=2), carry
    actual = flashrnn_torch(*actual_inputs, cell=case.cell, numerics=case.numerics)
    actual_gradients = torch.autograd.grad(loss(actual), actual_inputs)
    oracle_gradients = torch.autograd.grad(loss(oracle), oracle_inputs)
    wx, r, b, initial = actual_inputs
    split = max(1, wx.shape[1] // 2)
    first, carry = flashrnn_torch(
        wx[:, :split], r, b, initial, cell=case.cell, numerics=case.numerics
    )
    if split < wx.shape[1]:
        last, carry = flashrnn_torch(
            wx[:, split:], r, b, carry, cell=case.cell, numerics=case.numerics
        )
        first = torch.cat((first, last), dim=2)
    chunk = first, carry
    chunk_gradients = torch.autograd.grad(loss(chunk), actual_inputs)
    checks = {}
    for name, a, e in zip(("history", "final"), actual, oracle, strict=True):
        checks[name] = measure(a, e, forward_budget)
        checks[name]["per_state"] = [
            measure(x, y, forward_budget) for x, y in zip(a, e, strict=True)
        ]
    for name, a, e, c in zip(
        ("wx", "recurrent", "bias", "initial"),
        actual_gradients,
        oracle_gradients,
        chunk_gradients,
        strict=True,
    ):
        checks["gradient_" + name] = measure(a, e, gradient_budget)
        checks["chunk_gradient_" + name] = measure(c, a, gradient_budget)
    for name, a, e in zip(("chunk_history", "chunk_final"), chunk, actual, strict=True):
        checks[name] = measure(a, e, forward_budget)
    readonly = all(
        torch.equal(t.detach().cpu(), original)
        for t, original in zip(actual_inputs, originals, strict=True)
    )
    device_dtype_match = all(
        t.device == actual_inputs[0].device and t.dtype == actual_inputs[0].dtype
        for t in (*actual, *actual_gradients, *chunk, *chunk_gradients)
    )
    with torch.no_grad():
        repeated = flashrnn_torch(
            *actual_inputs, cell=case.cell, numerics=case.numerics
        )
        repeat_exact = all(
            torch.equal(a, b) for a, b in zip(actual, repeated, strict=True)
        )
        streams_exact = None
        if device.type == "cuda":
            current = torch.cuda.current_stream(device)
            streams = [torch.cuda.Stream(device=device) for _ in range(2)]
            concurrent = []
            for stream in streams:
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    concurrent.append(
                        flashrnn_torch(
                            *actual_inputs, cell=case.cell, numerics=case.numerics
                        )
                    )
            for stream in streams:
                current.wait_stream(stream)
            streams_exact = all(
                torch.equal(a, b)
                for result in concurrent
                for a, b in zip(actual, result, strict=True)
            )
    passed = (
        readonly
        and device_dtype_match
        and repeat_exact
        and streams_exact is not False
        and all(c["pass"] for c in checks.values())
    )
    row = {
        "case": case.name,
        "cell": case.cell,
        "shape_B_T_H_D": case.shape,
        "initial": case.initial,
        "seed": seed,
        "numerics": case.numerics,
        "dtype": str(actual_inputs[0].dtype),
        "oracle_dtype": str(oracle_inputs[0].dtype),
        "device": str(device),
        "actual_backend": "torch_eager",
        "status": "PASSED" if passed else "FAILED",
        "reference": "CPU_UPSTREAM_VANILLA"
        if mathematical
        else "CPU_SAME_CAST_REFERENCE",
        "forward_atol_rtol": forward_budget,
        "gradient_atol_rtol": gradient_budget,
        "checks": checks,
        "input_readonly": readonly,
        "device_dtype_match": device_dtype_match,
        "repeat_exact": repeat_exact,
        "two_cuda_streams_exact": streams_exact,
        "speed_claim": False,
        "gradient_contract": "per_step_cast_v2" if not mathematical else "input_dtype",
    }
    saved = {"inputs": originals}
    for name, tensors in (
        ("actual", actual),
        ("oracle", oracle),
        ("chunk", chunk),
        ("actual_gradients", actual_gradients),
        ("oracle_gradients", oracle_gradients),
        ("chunk_gradients", chunk_gradients),
        ("repeated", repeated),
        ("inputs_after", actual_inputs),
    ):
        saved[name] = tuple(t.detach().cpu() for t in tensors)
    return row, saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config["schema_version"] == 1
    mathematical_dtype = {"float32": torch.float32, "float64": torch.float64}[
        config.get("mathematical_dtype", "float64")
    ]
    cases = [
        Case(
            row["id"],
            cell,
            tuple(row["shape"]),
            policy,
            mathematical_dtype=mathematical_dtype,
        )
        for row in config["shapes"]
        for cell in config["cells"]
        for policy in config["policies"]
    ]
    cases += [
        Case(mode, "slstm", (3, 3, 2, 7), policy, mode, mathematical_dtype)
        for mode in ("zero", "mixed", "saturated")
        for policy in config["policies"]
    ]
    device = torch.device(args.device)
    assert device.type in ("cpu", "cuda", "mps")
    if device.type == "mps":
        assert torch.backends.mps.is_available()
        assert mathematical_dtype == torch.float32, (
            "MPS requires FP32 mathematical inputs"
        )
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    folder = args.output.with_suffix(".tensors")
    folder.mkdir()
    metadata = {
        "pid": os.getpid(),
        "started": time.time(),
        "status": "RUNNING",
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "hip_build": torch.version.hip,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device": str(device),
        "threads": 1,
        "config": config,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scope": "PUBLIC_TORCH_EAGER_STATES_GRADIENTS_CHUNKS_READONLY_REPEAT",
        "performance_claim": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(flashrnn_torch.__code__.co_filename),
                Path(recurrence.__code__.co_filename),
                Path(flashrnn.__code__.co_filename),
            )
        },
    }
    vanilla = Path(flashrnn.__code__.co_filename).parent / "vanilla"
    metadata["source_sha256"].update(
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in vanilla.glob("*.py")
        }
    )
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        metadata["accelerator"] = {
            "name": props.name,
            "memory_bytes": props.total_memory,
            "backend": "ROCM" if torch.version.hip else "NVIDIA_CUDA",
        }
        if torch.version.hip is None:
            metadata["accelerator"]["compute_capability"] = [props.major, props.minor]
    elif device.type == "mps":
        metadata["accelerator"] = {
            "backend": "APPLE_MPS",
            "macos": platform.mac_ver()[0],
            "cpu_fallback_enabled": False,
        }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with args.output.open("x") as handle:
        for index, case in enumerate(cases):
            row, tensors = run_case(case, config["seed"] + index, device)
            path = folder / f"{index:03d}.pt"
            torch.save(tensors, path)
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            row.update(tensors=path.name, tensor_sha256=digest)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
            print(
                json.dumps(
                    {key: row[key] for key in ("case", "cell", "numerics", "status")}
                ),
                flush=True,
            )
    passed = all(row["status"] == "PASSED" for row in rows)
    metadata.update(
        status="PASSED" if passed else "FAILED",
        finished=time.time(),
        cases=len(rows),
        passed=sum(row["status"] == "PASSED" for row in rows),
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
