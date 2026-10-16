"""Independently replay Haste LSTM/GRU CUDA outputs from saved inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

TOLERANCES = {
    "float16": {"atol": 0.015, "rtol": 0.03},
    "float32": {"atol": 1e-4, "rtol": 1e-4},
    "float64": {"atol": 1e-10, "rtol": 1e-10},
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def replay(row: dict[str, torch.Tensor], cell: str) -> tuple[np.ndarray, np.ndarray]:
    x, w, r, b, s = (
        row[name].numpy().astype(np.float64) for name in ("x", "w", "r", "b", "s")
    )
    wx = np.einsum("bti,ghdi->btghd", x, w)
    state = s[:, :, 0]
    history = []
    for projected in np.moveaxis(wx, 1, 0):
        recurrent = np.einsum("bnp,gnop->bgno", state[0], r)
        if cell == "lstm":
            i, f, g, o = np.moveaxis(projected + recurrent + b, 1, 0)
            c = sigmoid(f) * state[1] + sigmoid(i) * np.tanh(g)
            state = np.stack((sigmoid(o) * np.tanh(c), c))
        else:
            candidate = recurrent[:, 0] + b[0]
            reset = sigmoid(projected[:, 0] + recurrent[:, 1] + b[1])
            update = sigmoid(projected[:, 1] + recurrent[:, 2] + b[2])
            value = np.tanh(projected[:, 2] + b[3] + reset * candidate)
            state = (update * state[0] + (1 - update) * value)[None]
        history.append(state[0, :, 0])
    return np.stack(history, axis=1), state[:, :, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--producer-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.producer_meta.read_text())
    raw_sha256 = hashlib.sha256(args.raw.read_bytes()).hexdigest()
    if (
        metadata["status"] != "RAW_SAVED_INDEPENDENT_AUDIT_PENDING"
        or metadata["cases"] != 5
        or metadata["raw_sha256"] != raw_sha256
        or metadata["tolerances"] != TOLERANCES
    ):
        raise ValueError("Haste producer metadata differs from the raw gate")
    raw = torch.load(args.raw, map_location="cpu", weights_only=True)
    expected_keys = {
        "lstm_float64",
        "lstm_float32",
        "gru_float64",
        "gru_float32",
        "gru_float16",
    }
    if set(raw) != expected_keys:
        raise ValueError("missing or unexpected Haste gate case")
    results = []
    for key in sorted(raw):
        cell, dtype = key.split("_")
        row = raw[key]
        history, final = replay(row, cell)
        tol = TOLERANCES[dtype]
        for arm in ("reference", "native"):
            for name, expected in (("history", history), ("final", final)):
                actual = row[f"{arm}_{name}"].numpy()
                if actual.shape != expected.shape:
                    raise ValueError(f"shape mismatch: {key}/{arm}/{name}")
                failed = ~np.isfinite(actual) | ~np.isclose(actual, expected, **tol)
                results.append(
                    {
                        "case": key,
                        "arm": arm,
                        "tensor": name,
                        "elements": int(actual.size),
                        "failed": int(failed.sum()),
                        "max_abs": float(np.max(np.abs(actual - expected))),
                    }
                )
    report = {
        "status": "PASS"
        if all(row["failed"] == 0 for row in results)
        else "FAIL_NUMERICAL",
        "raw_sha256": raw_sha256,
        "producer_meta_sha256": hashlib.sha256(
            args.producer_meta.read_bytes()
        ).hexdigest(),
        "extension_sha256": metadata["extension_sha256"],
        "tensor_pairs": len(results),
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "tensor_pairs": len(results)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
