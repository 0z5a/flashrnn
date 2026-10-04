"""Paired L1 measurements including input projection and public output layout.

This is a layer workload, not a pretrained-model or serving E2E benchmark.
Run only in an admitted quiet window. Setup/packing is reported separately.
"""

import argparse
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import torch

from flashrnn.flashrnn2.reference import recurrence as reference
from flashrnn.flashrnn2.torch_layer import TorchLayer
from flashrnn.flashrnn2.triton_persistent import recurrence as persistent
from flashrnn.flashrnn2.triton_step import recurrence as step

Output = tuple[torch.Tensor, torch.Tensor]


def wall_ms(run: Callable[[], Output]) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = run()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter_ns() - started) / 1e6
    # Keep returned tensors alive through completion, outside the measured release.
    del output
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument(
        "--candidate",
        choices=("triton_step", "triton_persistent", "gluon_persistent"),
        default="triton_persistent",
    )
    parser.add_argument("--blocks", type=int, default=20)
    args = parser.parse_args()
    if args.blocks < 20:
        raise ValueError("at least 20 paired blocks are required")
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    batch, steps, heads, width = args.batch, args.steps, args.heads, args.width
    shapes = (
        (batch, steps, width),
        (4, heads, width, width),
        (4, heads, width, width),
        (4, heads, width),
        (2, batch, 1, heads, width),
    )
    cpu_inputs = tuple((torch.randn(shape) * 0.05).bfloat16() for shape in shapes)
    cpu_inputs[1].div_(width**0.5)
    cpu_inputs[2].div_(width**0.5)
    x, w, r, b, s = (tensor.cuda() for tensor in cpu_inputs)
    torch.cuda.synchronize()
    setup_start = time.perf_counter_ns()
    module = TorchLayer(w, r, b, "lstm")
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter_ns() - setup_start) / 1e6
    kernel = step if args.candidate == "triton_step" else persistent

    def baseline() -> Output:
        return module(x, s)

    def candidate() -> Output:
        wx = torch.einsum("bti,ghdi->btghd", x, w)
        if args.candidate == "gluon_persistent":
            history, final = kernel(wx, r, b, s, "lstm", register_layout=True)
        else:
            history, final = kernel(wx, r, b, s, "lstm")
        return history[0], final

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(".meta.json")
    contract = {"atol": 0.002, "rtol": 0.02, "max_error_limit": 0.003}
    source_files = (
        Path(kernel.__code__.co_filename),
        Path(reference.__code__.co_filename),
        Path(__file__),
        Path(TorchLayer.forward.__code__.co_filename),
    )
    if args.candidate == "gluon_persistent":
        source_files += (
            Path(persistent.__code__.co_filename).with_name("gluon_persistent.py"),
        )
    hashes = {
        str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    session = str(uuid.uuid4())
    props = torch.cuda.get_device_properties(0)
    metadata = {
        "session_id": session,
        "pid": os.getpid(),
        "started": time.time(),
        "case": f"B{batch}_T{steps}_H{heads}_D{width}",
        "cell": "lstm",
        "mode": "L1",
        "device_uuid": str(props.uuid),
        "gpu": props.name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "baseline_sha": torch.__version__ + ":" + hashes["torch_layer.py"],
        "candidate_sha": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": hashes,
        "dtype_contract_id": "bf16-layer-input-weights-hidden-final-v1",
        "reference_cast_path": "FP32 projection rounded to BF16; FP32 local states; recurrent h rounded to BF16; BF16 snapshots",
        "candidate_local_states": "FP32",
        "baseline_internal_casts": "cuDNN opaque; not inferred from candidate implementation",
        "contract": contract,
        "baseline": "torch_nn",
        "candidate": args.candidate,
        "scope": "LAYER_FORWARD_ONLY",
        "torch_nn_setup_ms": setup_ms,
        "original_cuda_fused": "NOT_RUN_MISSING_EXISTING_DEPENDENCIES",
        "statistical_scope": "within-process paired blocks; independent startup validation pending",
        "status": "RUNNING",
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    xc, wc, rc, bc, sc = cpu_inputs
    wx = torch.einsum("bti,ghdi->btghd", xc.float(), wc.float()).bfloat16().float()
    history, final = reference(
        wx, rc.float(), bc.float(), sc.float(), "lstm", mma_dtype=torch.bfloat16
    )
    targets = (history[0].bfloat16(), final.bfloat16())
    correctness = {}
    first_call_ms = {}
    with torch.no_grad():
        # Baseline is checked first against the independent cast reference.
        for label, run in (("baseline", baseline), ("candidate", candidate)):
            errors = []
            torch.cuda.synchronize()
            first_start = time.perf_counter_ns()
            outputs = run()
            torch.cuda.synchronize()
            first_call_ms[label] = (time.perf_counter_ns() - first_start) / 1e6
            for actual, target in zip(outputs, targets):
                actual = actual.cpu()
                error = (actual.float() - target.float()).abs().max().item()
                errors.append(error)
                torch.testing.assert_close(
                    actual, target, atol=contract["atol"], rtol=contract["rtol"]
                )
                assert error <= contract["max_error_limit"], (label, error)
            correctness[label] = errors
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            baseline()
            torch.cuda.synchronize()
        operators = sorted(
            {
                event.key
                for event in profile.key_averages()
                if "cudnn" in event.key.lower()
            }
        )
        if not operators:
            raise RuntimeError("torch.nn did not dispatch a cuDNN operator")
        metadata.update(
            correctness_max_errors=correctness,
            cudnn_operators=operators,
            first_call_ms=first_call_ms,
            compilation_cache="existing task cache; not clean-build timing",
        )
        for _ in range(5):
            baseline()
            candidate()
        torch.cuda.synchronize()
        with args.output.open("x") as handle:
            for block in range(args.blocks):
                runs = (("baseline", baseline), ("candidate", candidate))
                if block % 2:
                    runs = tuple(reversed(runs))
                timings = {label: wall_ms(run) for label, run in runs}
                row = {
                    key: metadata[key]
                    for key in (
                        "case",
                        "cell",
                        "mode",
                        "device_uuid",
                        "dtype_contract_id",
                        "baseline_sha",
                        "candidate_sha",
                        "session_id",
                    )
                }
                row.update(
                    block_id=block,
                    order=[label for label, _ in runs],
                    baseline_ms=timings["baseline"],
                    candidate_ms=timings["candidate"],
                    status="PASS",
                )
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps(row), flush=True)
    metadata.update(finished=time.time(), status="PASS")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
