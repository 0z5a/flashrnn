"""Independently replay saved ParaRNN full-stack outputs and gradients."""

import argparse
import hashlib
import json
import math
from pathlib import Path

SOURCE_REVISION = "513d75dade40843bb27ac4a521969997188b5459"
BUDGET = 1e-4
MODES = ("parallel_CUDA", "parallel_FUSED")


def tensor_keys(cell: str) -> set[str]:
    parameters = ("A", "B", "b") if cell == "gru" else ("A", "B", "C", "b")
    return {"output_0", "output_1", "input_grad"} | {
        f"layer{layer}.{name}" for layer in range(2) for name in parameters
    }


def validate(tensor: dict) -> None:
    shape = tensor["shape"]
    values = tensor["values"]
    if not shape or min(shape) < 1 or math.prod(shape) != len(values):
        raise ValueError("saved tensor shape and element count disagree")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite saved tensor value")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output already exists")
    raw_bytes = args.input.read_bytes()
    raw = json.loads(raw_bytes)
    if (
        raw["source_revision"] != SOURCE_REVISION
        or raw["budget"] != BUDGET
        or (raw["batch"], raw["steps"], raw["layers"], raw["dtype"])
        != (2, 16, 2, "float32")
        or set(raw["records"]) != {"gru", "lstm"}
    ):
        raise ValueError("ParaRNN qualification contract differs")
    rows = []
    for cell in ("gru", "lstm"):
        arms = raw["records"][cell]
        if set(arms) != {"sequential", *MODES}:
            raise ValueError("missing ParaRNN mode")
        keys = tensor_keys(cell)
        reference = arms["sequential"]
        if set(reference) != keys:
            raise ValueError("incomplete reference tensors")
        for name in sorted(keys):
            validate(reference[name])
        for mode in MODES:
            candidate = arms[mode]
            if set(candidate) != keys:
                raise ValueError("incomplete candidate tensors")
            for name in sorted(keys):
                actual, expected = candidate[name], reference[name]
                validate(actual)
                if actual["shape"] != expected["shape"]:
                    raise ValueError("candidate tensor shape differs")
                if name in {"output_0", "output_1", "input_grad"} and actual[
                    "shape"
                ] != [2, 16, 8]:
                    raise ValueError(
                        "full-stack output or input gradient shape differs"
                    )
                errors = [
                    abs(value - target)
                    for value, target in zip(actual["values"], expected["values"])
                ]
                failed = sum(
                    error > BUDGET * (1 + abs(target))
                    for error, target in zip(errors, expected["values"])
                )
                rows.append(
                    {
                        "cell": cell,
                        "mode": mode,
                        "tensor": name,
                        "elements": len(errors),
                        "failed": failed,
                        "max_abs": max(errors),
                    }
                )
    failures = sum(row["failed"] for row in rows)
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS" if failures == 0 else "FAIL_NUMERICAL",
                "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "source_revision": SOURCE_REVISION,
                "source_manifest_sha256": raw["source_manifest_sha256"],
                "extension_sha256": raw["extension_sha256"],
                "budget": BUDGET,
                "tensor_pairs": len(rows),
                "failed_elements": failures,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
