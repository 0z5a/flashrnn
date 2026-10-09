"""Independently check DEER GRU convergence, outputs and gradients."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REVISION = "409140ff98d2426e49090e3552a63da10d9cd6dc"
BUDGET = 1e-4
SHAPES = ((8, 8), (8, 8), (8,)) * 3


def array(record: dict, shape: tuple[int, ...]) -> np.ndarray:
    if record["shape"] != list(shape) or record["dtype"] != "float32":
        raise ValueError("saved tensor shape differs")
    values = np.asarray(record["values"], dtype=np.float64)
    if values.size != np.prod(shape) or not np.isfinite(values).all():
        raise ValueError("saved tensor values differ")
    return values.reshape(shape)


def reference_gru(inputs: np.ndarray, parameters: list[np.ndarray]) -> np.ndarray:
    wir, whr, bhr, wiz, whz, bhz, win, whn, bhn = parameters
    state = np.zeros((2, 8), dtype=np.float64)
    history = []

    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    for step in inputs:
        reset = sigmoid(step @ wir.T + state @ whr.T + bhr)
        update = sigmoid(step @ wiz.T + state @ whz.T + bhz)
        proposal = np.tanh(step @ win.T + reset * (state @ whn.T + bhn))
        state = (1 - update) * proposal + update * state
        history.append(state)
    return np.stack(history)


def compare(actual: np.ndarray, expected: np.ndarray) -> tuple[int, float]:
    delta = np.abs(actual - expected)
    return int(np.count_nonzero(delta > BUDGET * (1 + np.abs(expected)))), float(
        delta.max()
    )


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
        raw["source_revision"] != REVISION
        or raw["budget"] != BUDGET
        or (raw["batch"], raw["steps"], raw["width"], raw["layers"], raw["dtype"])
        != (2, 16, 8, 2, "float32")
        or set(raw["records"]) != {"sequential", "deer"}
    ):
        raise ValueError("DEER numerical contract differs")
    x = array(raw["inputs"], (16, 2, 8))
    if len(raw["parameters"]) != 2:
        raise ValueError("missing GRU layer parameters")
    weights = [
        [array(value, shape) for value, shape in zip(layer, SHAPES, strict=True)]
        for layer in raw["parameters"]
    ]
    expected = [reference_gru(x, weights[0])]
    expected.append(reference_gru(expected[0], weights[1]))
    shapes = {"output_0": (16, 2, 8), "output_1": (16, 2, 8), "input_grad": (16, 2, 8)}
    shapes.update(
        {
            f"layer{layer}.{index}": shape
            for layer in range(2)
            for index, shape in enumerate(SHAPES)
        }
    )
    checked = {}
    rows = []
    for mode in ("sequential", "deer"):
        record = raw["records"][mode]
        if set(record["tensors"]) != set(shapes):
            raise ValueError("DEER forward/gradient tensor set differs")
        for layer in range(2):
            success = record[f"success_{layer}"]
            if (
                success["shape"] != [16, 2, 8]
                or success["dtype"] != "bool"
                or len(success["values"]) != 256
            ):
                raise ValueError("DEER convergence evidence is incomplete")
            if not all(value is True for value in success["values"]):
                raise ValueError("DEER solver did not converge")
        tensors = {
            name: array(record["tensors"][name], shape)
            for name, shape in shapes.items()
        }
        checked[mode] = tensors
        for layer in range(2):
            name = f"output_{layer}"
            failed, maximum = compare(tensors[name], expected[layer])
            rows.append(
                {
                    "mode": mode,
                    "tensor": f"numpy_{name}",
                    "failed": failed,
                    "max_abs": maximum,
                }
            )
    for name in sorted(shapes):
        failed, maximum = compare(checked["deer"][name], checked["sequential"][name])
        rows.append(
            {
                "mode": "deer_vs_sequential",
                "tensor": name,
                "failed": failed,
                "max_abs": maximum,
            }
        )
    failures = sum(row["failed"] for row in rows)
    report = {
        "status": "PASS" if failures == 0 else "FAIL_NUMERICAL",
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_revision": REVISION,
        "source_manifest_sha256": raw["source_manifest_sha256"],
        "budget": BUDGET,
        "paired_tensor_count": len(shapes),
        "independent_forward_count": 4,
        "failed_elements": failures,
        "max_abs": max(row["max_abs"] for row in rows),
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
