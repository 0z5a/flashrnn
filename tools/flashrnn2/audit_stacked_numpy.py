"""Independently audit CPU Torch ZIP tensors using the existing NumPy runtime."""

import argparse
import collections
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np


def rebuild(storage, offset, shape, stride, _grad, _hooks, _metadata=None):
    data, dtype = storage
    itemsize = np.dtype(dtype).itemsize
    array = np.ndarray(
        shape,
        dtype=dtype,
        buffer=data,
        offset=offset * itemsize,
        strides=tuple(s * itemsize for s in stride),
    )
    if dtype == "<u2":
        return (array.astype("<u4") << 16).view("<f4")
    return array


class TensorReader(pickle.Unpickler):
    def __init__(self, archive, prefix):
        self.archive, self.prefix = archive, prefix
        super().__init__(io.BytesIO(archive.read(prefix + "data.pkl")))

    def find_class(self, module, name):
        allowed = {
            ("torch._utils", "_rebuild_tensor_v2"): rebuild,
            ("torch", "DoubleStorage"): "<f8",
            ("torch", "FloatStorage"): "<f4",
            ("torch", "HalfStorage"): "<f2",
            ("torch", "BFloat16Storage"): "<u2",
            ("torch", "LongStorage"): "<i8",
            ("collections", "OrderedDict"): collections.OrderedDict,
        }
        return allowed[(module, name)]

    def persistent_load(self, token):
        kind, dtype, key, location, size = token
        assert kind == "storage" and location == "cpu"
        data = self.archive.read(self.prefix + "data/" + key)
        assert len(data) == size * np.dtype(dtype).itemsize
        return data, dtype


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    x, y = actual.astype(np.float64), expected.astype(np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    delta = np.abs(x - y)
    assert np.all(delta <= 0.015 + 0.03 * np.abs(y))
    maximum = float(delta.max())
    assert maximum <= 0.03
    return maximum


parser = argparse.ArgumentParser()
parser.add_argument("jsonl", type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
meta = json.loads(args.jsonl.with_suffix(".meta.json").read_text())
tensorfile = args.jsonl.with_suffix(".qualification.pt")
assert (
    hashlib.sha256(tensorfile.read_bytes()).hexdigest() == meta["qualification_sha256"]
)
with zipfile.ZipFile(tensorfile) as archive:
    prefix = archive.namelist()[0].split("/")[0] + "/"
    assert archive.read(prefix + "byteorder") == b"little"
    saved = TensorReader(archive, prefix).load()
assert len(saved["cases"]) == meta["workload"]["groups"] == len(meta["fixture_sha256"])
if meta["baseline"] == "cudnn":
    flags = meta["cudnn_training_flags"]
    assert len(flags) == meta["model"]["layers"]
    assert all(not flag["module"] and not any(flag["heads"]) for flag in flags)
    assert meta["cudnn_operators"]
maximum = 0.0
pairs = 0
for index, case in enumerate(saved["cases"]):
    assert (
        hashlib.sha256(case["ids"].tobytes()).hexdigest()
        == meta["fixture_sha256"][index]
    )
    baseline, candidate = case["baseline"], case["candidate"]
    assert (
        len(baseline["layers"]) == len(candidate["layers"]) == meta["model"]["layers"]
    )
    for expected_layer, actual_layer in zip(
        baseline["layers"], candidate["layers"], strict=True
    ):
        assert len(expected_layer) == len(actual_layer) == 2
        for expected, actual in zip(expected_layer, actual_layer, strict=True):
            maximum = max(maximum, compare(actual, expected))
            pairs += 1
    maximum = max(maximum, compare(candidate["logits"], baseline["logits"]))
    pairs += 1
assert maximum == saved["max_abs"] == meta["qualification_max_abs"]
result = {
    "status": "PASS",
    "backend": "NumPy",
    "torch_imported": False,
    "gpu_executed": False,
    "tensor_pairs": pairs,
    "maximum_absolute_error": maximum,
    "qualification_sha256": meta["qualification_sha256"],
    "performance_claim": False,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
