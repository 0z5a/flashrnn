"""CUDA qualification of torch.nn layer mapping against the FP32 reference."""

import argparse
import json
from pathlib import Path

import torch

from flashrnn.flashrnn2.reference import SIZES, recurrence
from flashrnn.flashrnn2.torch_layer import TorchLayer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with args.output.open("x") as handle:
        for cell in ("lstm", "gru", "elman"):
            gates, bias_gates, states = SIZES[cell]
            for batch, steps, heads, width in ((3, 17, 2, 64), (16, 128, 1, 128)):
                shapes = (
                    (batch, steps, width),
                    (gates, heads, width, width),
                    (gates, heads, width, width),
                    (bias_gates, heads, width),
                    (states, batch, 1, heads, width),
                )
                x, w, r, b, s = tuple(
                    (0.05 * torch.randn(shape)).bfloat16() for shape in shapes
                )
                w /= width**0.5
                r /= width**0.5
                # FP32 projection rounds once to BF16, as the candidate layer does.
                wx = (
                    torch.einsum("bti,ghdi->btghd", x.float(), w.float())
                    .bfloat16()
                    .float()
                )
                history, final = recurrence(
                    wx, r.float(), b.float(), s.float(), cell, mma_dtype=torch.bfloat16
                )
                module = TorchLayer(w.cuda(), r.cuda(), b.cuda(), cell)
                with torch.no_grad():
                    actual = module(x.cuda(), s.cuda())
                errors = []
                valid = True
                for output, target in zip(actual, (history[0], final)):
                    output, target = output.cpu().float(), target.bfloat16().float()
                    error = (output - target).abs().max().item()
                    errors.append(error)
                    valid = (
                        valid
                        and torch.allclose(output, target, atol=0.002, rtol=0.02)
                        and error <= 0.003
                    )
                failed += not valid
                record = {
                    "cell": cell,
                    "shape_B_T_H_D": [batch, steps, heads, width],
                    "hidden_and_final_max_error": errors,
                    "status": "PASS" if valid else "CORRECTNESS_FAILED",
                    "actual_backend": "torch.nn",
                    "cudnn_enabled": torch.backends.cudnn.enabled,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "kernel_trace_verified": False,
                    "evidence_level": "CORRECTNESS_ONLY",
                    "contract": {"atol": 0.002, "rtol": 0.02, "max_error_limit": 0.003},
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(json.dumps(record), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
