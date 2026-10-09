"""Finite CPU Inductor qualification; timings are local probe measurements."""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import time
from functools import partial
from pathlib import Path

import torch
from portable_gate import Case, fixtures, loss, measure
from torch._dynamo.utils import counters

from flashrnn import flashrnn_torch
from flashrnn.flashrnn2.reference import SIZES, recurrence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=tuple(SIZES), required=True)
    parser.add_argument(
        "--numerics", choices=("mathematical", "fp32_state_bf16_mma"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    torch.set_num_threads(1)
    case = Case(
        "compile",
        args.cell,
        (2, 4, 1, 8),
        args.numerics,
        mathematical_dtype=torch.float32,
    )
    inputs = tuple(t.requires_grad_() for t in fixtures(case, 20261006))
    eager = partial(flashrnn_torch, cell=args.cell, numerics=args.numerics)
    source_files = (
        Path(__file__),
        Path(fixtures.__code__.co_filename),
        Path(flashrnn_torch.__code__.co_filename),
        Path(recurrence.__code__.co_filename),
    )
    record = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": time.time(),
        "cell": args.cell,
        "numerics": args.numerics,
        "shape": case.shape,
        "dtype": "torch.float32",
        "device": "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "backend": "inductor",
        "mode": "default",
        "fullgraph": True,
        "dynamic": False,
        "threads": 1,
        "compile_threads": os.environ["TORCHINDUCTOR_COMPILE_THREADS"],
        "cache": os.environ["TORCHINDUCTOR_CACHE_DIR"],
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
        },
        "atol": 1e-5,
        "rtol": 1e-5,
        "timing_scope": "UNQUALIFIED_LOCAL_SMALL_TENSOR_PROBE_NOT_MODEL_E2E",
    }
    progress = args.output / "progress.json"
    progress.write_text(json.dumps(record, indent=2) + "\n")
    expected = eager(*inputs)
    expected_gradients = torch.autograd.grad(loss(expected), inputs)
    start = time.perf_counter_ns()
    compiled = torch.compile(
        eager, backend="inductor", mode="default", fullgraph=True, dynamic=False
    )
    record["compile_wrapper_ms"] = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    actual = compiled(*inputs)
    record["first_forward_including_compile_ms"] = (
        time.perf_counter_ns() - start
    ) / 1e6
    progress.write_text(json.dumps(record, indent=2) + "\n")
    start = time.perf_counter_ns()
    actual_gradients = torch.autograd.grad(loss(actual), inputs)
    record["first_backward_including_compile_ms"] = (
        time.perf_counter_ns() - start
    ) / 1e6
    progress.write_text(json.dumps(record, indent=2) + "\n")
    wx, r, b, initial = inputs
    start = time.perf_counter_ns()
    first, carry = compiled(wx[:, :2], r, b, initial)
    last, final = compiled(wx[:, 2:], r, b, carry)
    chunk = torch.cat((first, last), dim=2), final
    chunk_gradients = torch.autograd.grad(loss(chunk), inputs)
    record["first_chunk_including_compile_ms"] = (time.perf_counter_ns() - start) / 1e6
    checks = {}
    for name, a, e, c in zip(
        ("history", "final"), actual, expected, chunk, strict=True
    ):
        checks[name] = measure(a, e, 1e-5)
        checks["chunk_" + name] = measure(c, e, 1e-5)
    for name, a, e, c in zip(
        ("wx", "recurrent", "bias", "initial"),
        actual_gradients,
        expected_gradients,
        chunk_gradients,
        strict=True,
    ):
        checks["gradient_" + name] = measure(a, e, 1e-5)
        checks["chunk_gradient_" + name] = measure(c, e, 1e-5)
    passed = all(c["pass"] for c in checks.values())
    record["checks"] = checks
    record["graphs_before_steady"] = dict(counters["stats"])
    timings = []
    if passed:
        for block in range(10):
            order = (
                ("eager", "compiled", "compiled", "eager")
                if block % 2 == 0
                else ("compiled", "eager", "eager", "compiled")
            )
            for method in order:
                function = eager if method == "eager" else compiled
                start = time.perf_counter_ns()
                output = function(*inputs)
                after_forward = time.perf_counter_ns()
                torch.autograd.grad(loss(output), inputs)
                finished = time.perf_counter_ns()
                timings.append(
                    {
                        "block": block,
                        "method": method,
                        "forward_ms": (after_forward - start) / 1e6,
                        "backward_and_loss_ms": (finished - after_forward) / 1e6,
                        "total_ms": (finished - start) / 1e6,
                    }
                )
        medians = {
            method: statistics.median(
                x["total_ms"] for x in timings if x["method"] == method
            )
            for method in ("eager", "compiled")
        }
        record["steady_total_median_ms"] = medians
        record["local_probe_speedup"] = medians["eager"] / medians["compiled"]
    record["graphs_after_steady"] = dict(counters["stats"])
    record["steady_recompile"] = record["graphs_after_steady"].get(
        "unique_graphs", 0
    ) - record["graphs_before_steady"].get("unique_graphs", 0)
    record["timings"] = timings
    unit = 1024**2 if platform.system() == "Darwin" else 1024
    record["process_peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / unit
    )
    record["completed_child_peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / unit
    )
    saved = {
        name: tuple(t.detach().cpu() for t in tensors)
        for name, tensors in (
            ("inputs", inputs),
            ("expected", expected),
            ("actual", actual),
            ("chunk", chunk),
            ("expected_gradients", expected_gradients),
            ("actual_gradients", actual_gradients),
            ("chunk_gradients", chunk_gradients),
        )
    }
    tensor_path = args.output / "tensors.pt"
    torch.save(saved, tensor_path)
    record["tensor_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    record.update(
        status="PASSED" if passed else "NUMERICAL_FAILED", finished=time.time()
    )
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "cell",
                    "numerics",
                    "first_forward_including_compile_ms",
                    "first_backward_including_compile_ms",
                    "steady_recompile",
                )
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
