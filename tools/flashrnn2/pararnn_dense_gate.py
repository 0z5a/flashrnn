"""Qualify the unchanged pinned ParaRNN dense reduction component on CPU.

Load only its two pure Torch methods; the stock package requires a compiled
CUDA extension at import. This is not qualification of that package entry.
"""

import argparse
import ast
import hashlib
import json
import typing
from pathlib import Path
from types import ModuleType

import torch

from flashrnn.flashrnn2.newton_dense import recurrence as newton
from flashrnn.flashrnn2.reference import SIZES
from flashrnn.flashrnn2.reference import recurrence as sequential

PIN = "513d75dade40843bb27ac4a521969997188b5459"
SOURCE_SHA = "ad223e647135c67029f385c2039bb8d1400df2204bdb495cf860af4a51c02285"


def load_solver(path: Path) -> ModuleType:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA:
        raise ValueError("ParaRNN dense source differs from the pinned baseline")
    tree = ast.parse(payload, filename=str(path))
    definition = next(
        x
        for x in tree.body
        if isinstance(x, ast.ClassDef) and x.name == "ParallelSolve"
    )
    definition.body = [
        x
        for x in definition.body
        if isinstance(x, ast.FunctionDef)
        and x.name in {"parallel_reduce", "_reduction_step_dense"}
    ]
    assert len(definition.body) == 2
    selected = ast.Module(body=[definition], type_ignores=[])
    module = ModuleType("pinned_pararnn_dense")
    module.__dict__.update(torch=torch, typ=typing)
    # These two unchanged definitions come only from the hash-pinned source.
    exec(compile(selected, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261010)
    module = load_solver(args.source)

    def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # ParaRNN represents the system's negative subdiagonal Jacobian.
        return module.ParallelSolve.parallel_reduce(
            -a.clone(), b.clone(), module.ParallelSolve._reduction_step_dense
        )

    metadata = {
        "scope": "PINNED_PARARNN_DENSE_REDUCTION_INSIDE_FULL_NEWTON_CPU_FORWARD",
        "repository": "apple-aiml-research/ml-pararnn",
        "revision": PIN,
        "source_sha256": SOURCE_SHA,
        "torch": torch.__version__,
        "seed": 20261010,
        "newton_sha256": hashlib.sha256(
            Path(newton.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu_executed": False,
        "stock_package_entry": "NOT_RUN",
        "gradient": "NOT_TESTED_FOR_INPLACE_UPSTREAM_REDUCTION",
    }
    rows = []
    with torch.no_grad():
        for cell in SIZES:
            gates, bias_gates, states = SIZES[cell]
            shapes = (
                (2, 64, gates, 2, 8),
                (gates, 2, 8, 8),
                (bias_gates, 2, 8),
                (states, 2, 1, 2, 8),
            )
            inputs = [torch.randn(shape, dtype=torch.float64) * 0.1 for shape in shapes]
            inputs[1].div_(8**0.5)
            if cell == "slstm":
                inputs[3][2].abs_().add_(1.5)
            expected = sequential(*inputs, cell)
            result = newton(*inputs, cell, linear_solver=solve)
            outputs = (result.history, result.final)
            errors = [(a - b).abs().max().item() for a, b in zip(outputs, expected)]
            passed = result.converged and all(
                torch.isclose(a, b, atol=1e-8, rtol=1e-7).all().item()
                for a, b in zip(outputs, expected)
            )
            rows.append(
                {
                    "cell": cell,
                    "B_T_H_D": [2, 64, 2, 8],
                    "iterations": len(result.residuals),
                    "residuals": result.residuals,
                    "max_abs_history_final": errors,
                    "status": "PASS" if passed else "FAILED",
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    args.output.write_text("".join(json.dumps(x) + "\n" for x in rows))
    metadata["status"] = "PASS" if all(x["status"] == "PASS" for x in rows) else "FAIL"
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    raise SystemExit(0 if metadata["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
