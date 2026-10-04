"""Finite qualification of hash-pinned upstream Triton forward kernels."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import triton

from flashrnn.flashrnn2.reference import recurrence as reference
from flashrnn.flashrnn2.upstream_triton import SOURCE_HASHES, recurrence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    source_paths = [
        Path(__file__),
        Path(reference.__code__.co_filename),
        Path(recurrence.__code__.co_filename),
    ]
    meta = {
        "scope": "UPSTREAM_TRITON_KERNEL_TORCH_LAYOUT_FORWARD_ONLY",
        "pid": os.getpid(),
        "started": time.time(),
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "upstream_kernel_sha256": SOURCE_HASHES,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
        "atol": 0.002,
        "rtol": 0.02,
        "max_error_limit": 0.003,
        "slstm_initialization": "elementwise, no gate clamp",
        "native_wrapper_parity": "PENDING_EXISTING_EINOPS_CUDA_RUNTIME",
        "status": "RUNNING",
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    failures = 0
    with args.output.open("x") as handle, torch.no_grad():
        for cell in ("lstm", "slstm"):
            for mode in ("nonzero", "zero", "mixed"):
                if cell == "lstm" and mode == "mixed":
                    continue
                b, t, h, d = 3, 17, 2, 64
                states = 2 if cell == "lstm" else 4
                wx = (torch.randn(b, t, 4, h, d) * 0.1).bfloat16()
                r = (torch.randn(4, h, d, d) * (0.1 / d**0.5)).bfloat16()
                bias = (torch.randn(4, h, d) * 0.05).bfloat16()
                initial = (torch.randn(states, b, 1, h, d) * 0.05).bfloat16()
                if cell == "slstm":
                    initial[2].abs_().add_(1)
                if mode in ("zero", "mixed"):
                    initial.zero_()
                if mode == "mixed":
                    initial[2, 0].fill_(1)
                    initial[3].fill_(2)
                cpu_inputs = (wx, r, bias, initial)
                targets = reference(
                    *(x.float() for x in cpu_inputs),
                    cell,
                    mma_dtype=torch.bfloat16,
                    slstm_init="elementwise",
                )
                gpu_inputs = tuple(x.cuda() for x in cpu_inputs)
                snapshots = tuple(x.clone() for x in gpu_inputs)
                actual = recurrence(*gpu_inputs, cell)
                torch.cuda.synchronize()
                errors = []
                passes = []
                for output, target in zip(actual, targets):
                    output, target = output.cpu(), target.bfloat16()
                    error = (output.float() - target.float()).abs().max().item()
                    errors.append(error)
                    passes.append(
                        torch.isclose(
                            output, target, atol=meta["atol"], rtol=meta["rtol"]
                        )
                        .all()
                        .item()
                        and error <= meta["max_error_limit"]
                    )
                readonly = all(torch.equal(x, y) for x, y in zip(gpu_inputs, snapshots))
                passed = all(passes) and readonly
                failures += int(not passed)
                row = {
                    "cell": cell,
                    "initial": mode,
                    "shape_B_T_H_D": [b, t, h, d],
                    "max_error_history_final": errors,
                    "inputs_readonly": readonly,
                    "status": "PASS" if passed else "CORRECTNESS_FAILED",
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps(row), flush=True)
    meta.update(
        finished=time.time(), failed=failures, status="FAIL" if failures else "PASS"
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
