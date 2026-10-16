"""Qualify the pinned DEER GRU sequence solver against its sequential method."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib

REVISION = "409140ff98d2426e49090e3552a63da10d9cd6dc"
SHAPES = ((8, 8), (8, 8), (8,)) * 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gru_step(previous, x, parameters):
    wir, whr, bhr, wiz, whz, bhz, win, whn, bhn = parameters
    reset = jax.nn.sigmoid(wir @ x + whr @ previous + bhr)
    update = jax.nn.sigmoid(wiz @ x + whz @ previous + bhz)
    proposal = jnp.tanh(win @ x + reset * (whn @ previous + bhn))
    return (1 - update) * proposal + update * previous


def full_sequence(first_params, second_params, x, method):
    from deer.fseq1d import seq1d

    def layer(inputs, parameters):
        initial = jnp.zeros((inputs.shape[1], 8), dtype=inputs.dtype)
        transposed = jnp.swapaxes(inputs, 0, 1)
        result = jax.vmap(
            lambda state, sequence: seq1d(
                gru_step, state, sequence, parameters, method=method
            )
        )(initial, transposed)
        return jnp.swapaxes(result.value, 0, 1), jnp.swapaxes(result.success, 0, 1)

    first, success0 = layer(x, first_params)
    second, success1 = layer(first, second_params)
    return first, second, success0, success1


def saved(tensor):
    value = jax.device_get(tensor)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "values": value.reshape(-1).tolist(),
    }


def compile_mode(method):
    def forward(p0, p1, inputs):
        return full_sequence(p0, p1, inputs, method)

    def loss(p0, p1, inputs):
        return jnp.square(forward(p0, p1, inputs)[1]).mean()

    return jax.jit(forward), jax.jit(jax.grad(loss, argnums=(0, 1, 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    manifest = json.loads(args.manifest.read_text())
    if manifest["revision"] != REVISION or manifest["file_count"] != 16:
        raise ValueError("DEER source revision differs")
    for entry in manifest["files"]:
        if sha256(args.source_root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"DEER source file differs: {entry['path']}")
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    if "," in visible or jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise ValueError("exactly one assigned GPU must be visible")
    actual_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if actual_uuid != args.expected_gpu_uuid:
        raise ValueError("GPU differs from the admitted device")
    sys.path.insert(0, str(args.source_root))
    from deer.fseq1d import seq1d

    keys = jax.random.split(jax.random.PRNGKey(20261009), 19)
    parameters = [
        tuple(
            jax.random.normal(key, shape, dtype=jnp.float32) * 0.05
            for key, shape in zip(keys[offset : offset + 9], SHAPES)
        )
        for offset in (0, 9)
    ]
    x = jax.random.normal(keys[18], (16, 2, 8), dtype=jnp.float32) * 0.2
    records = {}
    for name, method in (
        ("sequential", seq1d.Sequential()),
        ("deer", seq1d.DEER()),
    ):
        forward, gradient = compile_mode(method)
        first, last, success0, success1 = forward(*parameters, x)
        grad0, grad1, grad_x = gradient(*parameters, x)
        tensors = {
            "output_0": saved(first),
            "output_1": saved(last),
            "input_grad": saved(grad_x),
        }
        for layer, gradients in enumerate((grad0, grad1)):
            tensors.update(
                {
                    f"layer{layer}.{index}": saved(value)
                    for index, value in enumerate(gradients)
                }
            )
        records[name] = {
            "tensors": tensors,
            "success_0": saved(success0),
            "success_1": saved(success1),
        }
    args.output.write_text(
        json.dumps(
            {
                "source_revision": REVISION,
                "source_manifest_sha256": sha256(args.manifest),
                "jax_version": jax.__version__,
                "jaxlib_version": jaxlib.__version__,
                "dtype": "float32",
                "batch": 2,
                "steps": 16,
                "width": 8,
                "layers": 2,
                "budget": 1e-4,
                "device_uuid": actual_uuid,
                "inputs": saved(x),
                "parameters": [
                    [saved(value) for value in layer] for layer in parameters
                ],
                "records": records,
            },
            separators=(",", ":"),
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
