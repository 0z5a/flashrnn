"""Diagnose full DeltaNet continuation with a fixed GEMM row shape.

The arithmetic order changes intentionally. This does not qualify the original
reference, native FLA kernels or a performance implementation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from gla_reference_gate import main as reference_main
from torch.nn import functional as F

original_linear = F.linear


def token_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    output = torch.cat(
        [original_linear(row[None], weight, bias) for row in rows], dim=0
    )
    return output.reshape(*x.shape[:-1], weight.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", required=True, choices=("deltanet",))
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    sidecar = args.output.with_suffix(".arithmetic.json")
    assert not sidecar.exists() and not args.output.exists()
    sidecar.write_text(
        json.dumps(
            {
                "scope": "FULL_DELTANET_CHANGED_LINEAR_ROW_SHAPE_DIAGNOSTIC",
                "arithmetic_changed": True,
                "linear_input_rows": 1,
                "convolution_unchanged": True,
                "original_gate_replaced": False,
                "performance_claim": False,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    F.linear = token_linear
    reference_main()


if __name__ == "__main__":
    main()
