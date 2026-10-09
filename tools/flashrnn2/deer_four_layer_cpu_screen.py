"""Screen the pinned DEER four-layer GRU on full sequences and gradients."""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from audit_deer_native_gate import compare
from deer_native_gate import REVISION, gru_step

WIDTH = 32
STEPS = 128
LAYERS = 4
SHAPES = ((WIDTH, WIDTH), (WIDTH, WIDTH), (WIDTH,)) * 3


def numpy_layer(inputs: np.ndarray, parameters: list[np.ndarray]) -> np.ndarray:
    wir, whr, bhr, wiz, whz, bhz, win, whn, bhn = parameters
    state = np.zeros((inputs.shape[1], WIDTH), dtype=np.float64)
    history = []
    for step in inputs:
        reset = 1 / (1 + np.exp(-(step @ wir.T + state @ whr.T + bhr)))
        update = 1 / (1 + np.exp(-(step @ wiz.T + state @ whz.T + bhz)))
        proposal = np.tanh(step @ win.T + reset * (state @ whn.T + bhn))
        state = (1 - update) * proposal + update * state
        history.append(state)
    return np.stack(history)


def apply_layer(inputs, layer, method):
    from deer.fseq1d import seq1d

    initial = jnp.zeros((inputs.shape[1], WIDTH), dtype=inputs.dtype)
    result = jax.vmap(
        lambda state, sequence: seq1d(gru_step, state, sequence, layer, method=method)
    )(initial, jnp.swapaxes(inputs, 0, 1))
    return jnp.swapaxes(result.value, 0, 1), jnp.swapaxes(result.success, 0, 1)


def model(parameters, inputs, method):
    histories = []
    masks = []
    for layer in parameters:
        inputs, success = apply_layer(inputs, layer, method)
        histories.append(inputs)
        masks.append(success)
    return tuple(histories), tuple(masks)


def compile_mode(method):
    def forward(weights, inputs):
        return model(weights, inputs, method)

    def loss(weights, inputs):
        return jnp.square(forward(weights, inputs)[0][-1]).mean()

    return jax.jit(forward), jax.jit(jax.grad(loss, argnums=(0, 1)))


def run_batch(batch: int, parameters, input_key, numpy_weights) -> dict:
    from deer.fseq1d import seq1d

    inputs = (
        jax.random.normal(
            jax.random.fold_in(input_key, batch),
            (STEPS, batch, WIDTH),
            dtype=jnp.float32,
        )
        * 0.2
    )
    records = {}
    for name, method in (
        ("sequential", seq1d.Sequential()),
        ("deer", seq1d.DEER()),
    ):
        forward, gradient = compile_mode(method)
        histories, masks = forward(parameters, inputs)
        weight_grads, input_grad = gradient(parameters, inputs)
        records[name] = {
            "history": [
                np.asarray(jax.device_get(value), dtype=np.float64)
                for value in histories
            ],
            "gradient": [
                np.asarray(jax.device_get(value), dtype=np.float64)
                for layer in weight_grads
                for value in layer
            ]
            + [np.asarray(jax.device_get(input_grad), dtype=np.float64)],
            "converged": all(bool(jnp.all(mask)) for mask in masks),
        }

    expected = []
    values = np.asarray(jax.device_get(inputs), dtype=np.float64)
    for layer in numpy_weights:
        values = numpy_layer(values, layer)
        expected.append(values)
    replay = []
    for name in ("sequential", "deer"):
        for index, (actual, reference) in enumerate(
            zip(records[name]["history"], expected, strict=True)
        ):
            failed, maximum = compare(actual, reference)
            replay.append(
                {"method": name, "layer": index, "failed": failed, "max_abs": maximum}
            )
    paired = []
    for kind in ("history", "gradient"):
        for index, (reference, candidate) in enumerate(
            zip(records["sequential"][kind], records["deer"][kind], strict=True)
        ):
            failed, maximum = compare(candidate, reference)
            paired.append(
                {"kind": kind, "index": index, "failed": failed, "max_abs": maximum}
            )
    failures = sum(row["failed"] for row in paired + replay)
    converged = {name: record["converged"] for name, record in records.items()}
    return {
        "batch": batch,
        "status": "PASS"
        if failures == 0 and all(converged.values())
        else "FAIL_NUMERICAL",
        "paired_tensors": len(paired),
        "independent_forward_replays": len(replay),
        "failed_elements": failures,
        "max_abs": max(row["max_abs"] for row in paired + replay),
        "converged": converged,
        "paired": paired,
        "replay": replay,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or any(
        batch not in (1, 4, 16, 32, 64) for batch in args.batches
    ):
        parser.error("output exists or unsupported batch")
    manifest = json.loads(args.manifest.read_text())
    if manifest["revision"] != REVISION or manifest["file_count"] != 16:
        raise ValueError("pinned DEER source differs")
    for entry in manifest["files"]:
        if (
            hashlib.sha256((args.source_root / entry["path"]).read_bytes()).hexdigest()
            != entry["sha256"]
        ):
            raise ValueError(f"pinned DEER file differs: {entry['path']}")
    if jax.default_backend() != "cpu":
        raise ValueError("this preflight must not be reported as a GPU result")
    sys.path.insert(0, str(args.source_root))
    jax.config.update("jax_default_matmul_precision", "highest")
    keys = jax.random.split(jax.random.PRNGKey(20261009), LAYERS * 9 + 1)
    parameters = tuple(
        tuple(
            jax.random.normal(key, shape, dtype=jnp.float32) * 0.05
            for key, shape in zip(
                keys[index * 9 : (index + 1) * 9], SHAPES, strict=True
            )
        )
        for index in range(LAYERS)
    )
    numpy_weights = [
        [np.asarray(jax.device_get(value), dtype=np.float64) for value in layer]
        for layer in parameters
    ]
    results = [
        run_batch(batch, parameters, keys[-1], numpy_weights) for batch in args.batches
    ]
    report = {
        "scope": "CPU-only four-layer full-history and backward screen; no GPU or E2E timing",
        "source_revision": REVISION,
        "source_manifest_sha256": hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "runtime": {
            name: version(name)
            for name in ("jax", "jaxlib", "numpy", "optimistix", "equinox")
        },
        "steps": STEPS,
        "width": WIDTH,
        "layers": LAYERS,
        "dtype": "float32",
        "budget": "1e-4 * (1 + abs(reference))",
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(result["status"] != "PASS" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
