"""Finite correctness diagnostics; this script does not report speedups."""

import argparse
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

import torch

from flashrnn.flashrnn2.reference import recurrence as reference
from flashrnn.flashrnn2.triton_persistent import recurrence as persistent
from flashrnn.flashrnn2.triton_step import recurrence as step

BACKENDS = {
    "triton_step": step,
    "triton_persistent": persistent,
    "gluon_persistent": partial(persistent, register_layout=True),
}

# Frozen before the first candidate GPU execution. Keep failed rows.
CONTRACT = {
    "atol": 0.002,
    "rtol": 0.02,
    "max_error_limit": 0.003,
    "reference": "FP32 local states; BF16 recurrent h; BF16 input/weight values; BF16 snapshots",
}


def cases() -> list[tuple[str, str, tuple[torch.Tensor, ...]]]:
    torch.manual_seed(20261005)
    result = []
    shapes = [("C0", 1, 1, 1, 64), ("C1", 3, 17, 1, 64), ("prototype", 16, 128, 1, 256)]
    # Preserve the original six cases' RNG order for failure reproduction.
    for cell in ("lstm", "slstm"):
        for name, batch, steps, heads, width in shapes:
            count = 4 if cell == "slstm" else 2
            wx = (torch.randn(batch, steps, 4, heads, width) * 0.1).bfloat16()
            r = (torch.randn(4, heads, width, width) * (0.1 / width**0.5)).bfloat16()
            bias = (torch.randn(4, heads, width) * 0.05).bfloat16()
            initial = (torch.randn(count, batch, 1, heads, width) * 0.05).bfloat16()
            if cell == "slstm":
                initial[2].abs_().add_(1)
            result.append((name, cell, (wx, r, bias, initial)))
    for mode in ("zero", "mixed"):
        wx, r, bias, initial = (x.clone() for x in result[4][2])
        initial.zero_()
        if mode == "mixed":
            initial[2, 0].fill_(1)
        result.append(("C1_" + mode, "slstm", (wx, r, bias, initial)))
    for cell in ("lstm", "slstm"):
        count = 4 if cell == "slstm" else 2
        for width in (48, 128):
            wx = (torch.randn(3, 33, 4, 2, width) * 0.1).bfloat16()
            r = (torch.randn(4, 2, width, width) * (0.1 / width**0.5)).bfloat16()
            bias = torch.zeros(4, 2, width, dtype=torch.bfloat16)
            initial = torch.zeros(count, 3, 1, 2, width, dtype=torch.bfloat16)
            result.append((f"heads2_D{width}", cell, (wx, r, bias, initial)))
    return result


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    difference = (actual.float() - expected.float()).abs()
    states = []
    for index, state in enumerate(difference):
        location = tuple(
            int(x) for x in torch.unravel_index(state.argmax(), state.shape)
        )
        full_location = (index, *location)
        states.append(
            {
                "max_error": state.max().item(),
                "location_B_T_H_D": location,
                "actual": actual[full_location].item(),
                "expected": expected[full_location].item(),
            }
        )
    return {
        "max_error": difference.max().item(),
        "max_error_by_timestep": difference.amax(dim=(0, 1, 3, 4)).tolist(),
        "relative_l2": (
            difference.norm() / expected.float().norm().clamp_min(1e-30)
        ).item(),
        "states": states,
    }


def compiler_record(details: dict[str, object], folder: Path) -> dict[str, object]:
    result = {}
    for key, value in details.items():
        if isinstance(value, dict):
            result[key] = compiler_record(value, folder)
        elif key == "ptx":
            assert isinstance(value, str)
            digest = hashlib.sha256(value.encode()).hexdigest()
            (folder / (digest + ".ptx")).write_text(value)
            result["ptx_sha256"] = digest
            result["mma_sync"] = "mma.sync" in value
        else:
            result[key] = value
    return result


def lifecycle(
    backend: str, inputs: tuple[torch.Tensor, ...], cell: str
) -> dict[str, object]:
    run = BACKENDS[backend]
    snapshots = tuple(x.clone() for x in inputs)
    with torch.no_grad():
        first, _ = run(*inputs, cell)
        second, _ = run(*inputs, cell)
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        outputs = []
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs.append(run(*inputs, cell)[0])
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for output in (second, *outputs):
            torch.testing.assert_close(first, output, atol=0, rtol=0)
        for original, current in zip(snapshots, inputs):
            torch.testing.assert_close(original, current, atol=0, rtol=0)
        # A new recurrent tensor must affect the result; no pointer-keyed stale pack.
        changed = (inputs[0], inputs[1] + 0.125, inputs[2], inputs[3])
        changed_output, _ = run(*changed, cell)
        target, _ = reference(
            *(x.cpu().float() for x in changed), cell, mma_dtype=torch.bfloat16
        )
        torch.testing.assert_close(
            changed_output.cpu(),
            target.bfloat16(),
            atol=CONTRACT["atol"],
            rtol=CONTRACT["rtol"],
        )
        assert not torch.equal(first, changed_output)
    return {
        "repeat": "PASS",
        "concurrent_streams": "PASS",
        "input_readonly": "PASS",
        "changed_recurrent": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=tuple(BACKENDS),
        default=list(BACKENDS),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ptx = args.output.parent / (args.output.stem + "-ptx")
    ptx.mkdir(exist_ok=True)
    metadata = {
        "pid": os.getpid(),
        "started": time.time(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cc": torch.cuda.get_device_capability(),
        "source_sha256": {
            Path(fn.__code__.co_filename).name: hashlib.sha256(
                Path(fn.__code__.co_filename).read_bytes()
            ).hexdigest()
            for fn in (step, persistent, reference, main)
        },
        "evidence_level": "CORRECTNESS_ONLY",
        "status": "RUNNING",
        "contract": CONTRACT,
    }
    gluon_source = Path(persistent.__code__.co_filename).with_name(
        "gluon_persistent.py"
    )
    metadata["source_sha256"][gluon_source.name] = hashlib.sha256(
        gluon_source.read_bytes()
    ).hexdigest()
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    failures = 0
    count = 0
    with args.output.open("x") as handle:
        for name, cell, cpu_inputs in cases():
            expected, _ = reference(
                *(x.float() for x in cpu_inputs), cell, mma_dtype=torch.bfloat16
            )
            target = expected.bfloat16()
            tensors = tuple(x.cuda() for x in cpu_inputs)
            for backend in args.backends:
                if backend != "triton_step" and tensors[0].shape[-1] > 128:
                    record = {
                        "case": name,
                        "cell": cell,
                        "actual_backend": backend,
                        "status": "UNSUPPORTED_D",
                        "limit": 128,
                    }
                elif (
                    backend == "triton_persistent"
                    and tensors[0].shape[-1] == 128
                    and torch.cuda.get_device_capability() == (12, 0)
                ):
                    record = {
                        "case": name,
                        "cell": cell,
                        "actual_backend": backend,
                        "status": "UNSUPPORTED_SHARED_MEMORY",
                        "required_bytes": 137216,
                        "observed_limit_bytes": 101376,
                        "evidence": "r2 actual OutOfResources on SM120",
                    }
                else:
                    run = BACKENDS[backend]
                    details = {}
                    with torch.no_grad():
                        actual, final = run(*tensors, cell, diagnostics=details)
                        debug, _ = run(*tensors, cell, snapshot_dtype=torch.float32)
                    torch.cuda.synchronize()
                    output = actual.cpu()
                    metrics = errors(output, target)
                    valid = (
                        torch.allclose(
                            output, target, atol=CONTRACT["atol"], rtol=CONTRACT["rtol"]
                        )
                        and metrics["max_error"] <= CONTRACT["max_error_limit"]
                    )
                    final_valid = torch.equal(final.cpu(), output[:, :, -1:])
                    valid = valid and final_valid
                    record = {
                        "case": name,
                        "cell": cell,
                        "actual_backend": backend,
                        "shape_B_T_H_D": [tensors[0].shape[i] for i in (0, 1, 3, 4)],
                        "status": "PASS" if valid else "CORRECTNESS_FAILED",
                        "bf16": metrics,
                        "fp32_diagnostic": errors(debug.cpu(), expected),
                        "final_state_matches_history": final_valid,
                        "compiler": compiler_record(details, ptx),
                        "backward": "UNIMPLEMENTED",
                    }
                    if name == "C1":
                        record["lifecycle"] = lifecycle(backend, tensors, cell)
                    if not valid:
                        failures += 1
                        torch.save(
                            {
                                "actual": output,
                                "expected": target,
                                "fp32_actual": debug.cpu(),
                                "fp32_expected": expected,
                                "inputs": cpu_inputs,
                            },
                            args.output.parent
                            / f"{args.output.stem}-{backend}-{cell}-{name}.failure.pt",
                        )
                    count += 1
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(json.dumps(record), flush=True)
    metadata.update(
        finished=time.time(),
        evaluated=count,
        failed=failures,
        status="PASS" if not failures else "CORRECTNESS_FAILED",
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
