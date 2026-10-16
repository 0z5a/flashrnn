"""Replay saved ParaRNN full-model outputs without importing Torch."""

import argparse
import collections
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

REVISION = "513d75dade40843bb27ac4a521969997188b5459"
BUDGET = 1e-4


def rebuild(storage, offset, shape, stride, _grad, _hooks, _metadata=None):
    data, dtype = storage
    size = np.dtype(dtype).itemsize
    tensor = np.ndarray(
        shape,
        dtype=dtype,
        buffer=data,
        offset=offset * size,
        strides=tuple(step * size for step in stride),
    )
    return (tensor.astype("<u4") << 16).view("<f4") if dtype == "<u2" else tensor


class TensorReader(pickle.Unpickler):
    def __init__(self, archive, prefix):
        self.archive, self.prefix = archive, prefix
        super().__init__(io.BytesIO(archive.read(prefix + "data.pkl")))

    def find_class(self, module, name):
        allowed = {
            ("torch._utils", "_rebuild_tensor_v2"): rebuild,
            ("torch", "FloatStorage"): "<f4",
            ("torch", "HalfStorage"): "<f2",
            ("torch", "BFloat16Storage"): "<u2",
            ("torch", "LongStorage"): "<i8",
            ("collections", "OrderedDict"): collections.OrderedDict,
        }
        return allowed[(module, name)]

    def persistent_load(self, token):
        kind, dtype, key, location, size = token
        if kind != "storage" or location != "cpu":
            raise ValueError("qualification must contain CPU tensor storage")
        data = self.archive.read(self.prefix + "data/" + key)
        if len(data) != size * np.dtype(dtype).itemsize:
            raise ValueError("qualification tensor storage has the wrong size")
        return data, dtype


def read_tensors(path: Path):
    with zipfile.ZipFile(path) as archive:
        prefix = archive.namelist()[0].split("/")[0] + "/"
        if archive.read(prefix + "byteorder") != b"little":
            raise ValueError("qualification tensor byte order differs")
        return TensorReader(archive, prefix).load()


def compare(candidate: np.ndarray, baseline: np.ndarray, shape: tuple[int, ...]):
    if candidate.shape != baseline.shape or candidate.shape != shape:
        raise ValueError("qualification tensor shape differs")
    if candidate.dtype != np.dtype("<f4") or baseline.dtype != np.dtype("<f4"):
        raise ValueError("qualification tensors must be float32")
    actual, expected = candidate.astype(np.float64), baseline.astype(np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("qualification contains nonfinite values")
    errors = np.abs(actual - expected)
    return int(np.count_nonzero(errors > BUDGET * (1 + np.abs(expected)))), float(
        errors.max()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output already exists")
    saved = read_tensors(args.input)
    batch, concurrency = saved["batch"], saved["concurrency"]
    if (
        saved["source_revision"] != REVISION
        or saved["budget"] != BUDGET
        or (saved["steps"], saved["layers"], saved["width"]) != (128, 4, 64)
        or saved["cell"] not in ("gru", "lstm")
        or saved["mode"] not in ("parallel_CUDA", "parallel_FUSED")
        or batch not in (1, 4, 16, 32, 64)
        or concurrency not in (1, 8, 32, 64, 128)
        or concurrency < batch
        or concurrency % batch
        or len(saved["records"]) != concurrency // batch
    ):
        raise ValueError("ParaRNN full-model qualification contract differs")
    rows = []
    for group, item in enumerate(saved["records"]):
        ids = item["ids"]
        if ids.shape != (batch, 128) or ids.dtype != np.dtype("<i8"):
            raise ValueError("request fixture differs")
        if np.any((ids < 0) | (ids >= 512)):
            raise ValueError("request IDs exceed vocabulary")
        baseline, candidate = item["baseline"], item["candidate"]
        if len(baseline["layers"]) != 4 or len(candidate["layers"]) != 4:
            raise ValueError("missing full-model layer output")
        tensors = [("logits", (batch, 512), baseline["logits"], candidate["logits"])]
        tensors.extend(
            (f"layer{layer}", (batch, 128, 64), base, actual)
            for layer, (base, actual) in enumerate(
                zip(baseline["layers"], candidate["layers"], strict=True)
            )
        )
        for name, shape, base, actual in tensors:
            failed, maximum = compare(actual, base, shape)
            rows.append(
                {
                    "group": group,
                    "tensor": name,
                    "elements": int(base.size),
                    "failed": failed,
                    "max_abs": maximum,
                }
            )
    failures = sum(row["failed"] for row in rows)
    result = {
        "status": "PASS" if failures == 0 else "FAIL_NUMERICAL",
        "backend": "NumPy",
        "torch_imported": False,
        "gpu_executed": False,
        "raw_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_manifest_sha256": saved["source_manifest_sha256"],
        "extension_sha256": saved["extension_sha256"],
        "native_gate_audit_sha256": saved["native_gate_audit_sha256"],
        "tensor_pairs": len(rows),
        "elements": sum(row["elements"] for row in rows),
        "failed_elements": failures,
        "max_abs": max(row["max_abs"] for row in rows),
        "results": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
