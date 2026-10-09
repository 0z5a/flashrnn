"""Qualify and time paired complete DEER GRU training requests on one GPU."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from deer_four_layer_cpu_screen import LAYERS, SHAPES, STEPS, WIDTH, model
from deer_native_gate import REVISION, sha256

BUDGET = 1e-4
LEARNING_RATE = 1e-3


def named_parameters(parameters) -> dict:
    return {
        f"layer{layer}.{index}": value
        for layer, weights in enumerate(parameters)
        for index, value in enumerate(weights)
    }


def compile_mode(method):
    def trace(parameters, inputs):
        return model(parameters, inputs, method)

    def train(parameters, inputs):
        def loss(weights):
            return jnp.square(trace(weights, inputs)[0][-1]).mean()

        value, gradients = jax.value_and_grad(loss)(parameters)
        updated = jax.tree.map(
            lambda weight, gradient: weight - LEARNING_RATE * gradient,
            parameters,
            gradients,
        )
        return value, gradients, updated

    return jax.jit(trace), jax.jit(train)


def append_arrays(arrays: dict, prefix: str, result) -> None:
    histories, masks, loss, gradients, updated = result
    arrays.setdefault(f"{prefix}.loss", []).append(np.asarray(jax.device_get(loss)))
    for index, (history, mask) in enumerate(zip(histories, masks, strict=True)):
        arrays.setdefault(f"{prefix}.history.{index}", []).append(
            np.asarray(jax.device_get(history))
        )
        arrays.setdefault(f"{prefix}.success.{index}", []).append(
            np.asarray(jax.device_get(mask))
        )
    for name, value in named_parameters(gradients).items():
        arrays.setdefault(f"{prefix}.gradient.{name}", []).append(
            np.asarray(jax.device_get(value))
        )
    for name, value in named_parameters(updated).items():
        arrays.setdefault(f"{prefix}.updated.{name}", []).append(
            np.asarray(jax.device_get(value))
        )


def qualify(parameters, inputs, compiled: dict, archive: Path) -> dict:
    arrays = {"inputs": inputs}
    for name, value in named_parameters(parameters).items():
        arrays[f"parameter.{name}"] = np.asarray(jax.device_get(value))
    qualified = {}
    for arm in ("baseline", "candidate"):
        trace, train = compiled[arm]
        qualified[arm] = []
        for host_inputs in inputs:
            device_inputs = jax.device_put(host_inputs)
            histories, masks = trace(parameters, device_inputs)
            loss, gradients, updated = train(parameters, device_inputs)
            jax.block_until_ready((histories, masks, loss, gradients, updated))
            qualified[arm].append((loss, gradients, updated))
            append_arrays(
                arrays,
                arm,
                (histories, masks, loss, gradients, updated),
            )
    np.savez_compressed(
        archive,
        **{
            name: np.stack(value) if isinstance(value, list) else value
            for name, value in arrays.items()
        },
    )
    return qualified


def verified_result(actual, expected) -> bool:
    leaves = zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
    return all(
        bool(jnp.all(jnp.abs(value - reference) <= BUDGET * (1 + jnp.abs(reference))))
        for value, reference in leaves
    )


def burst(parameters, host_inputs, train, qualified) -> dict:
    started = time.perf_counter_ns()
    results = [train(parameters, jax.device_put(inputs)) for inputs in host_inputs]
    jax.block_until_ready(results)
    jax.device_get([result[0] for result in results])
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    if not all(
        verified_result(result, expected)
        for result, expected in zip(results, qualified, strict=True)
    ):
        raise ValueError("timed training request differs from qualification")
    return {
        "elapsed_ms": elapsed_ms,
        "requests_per_second": len(host_inputs) * 1000 / elapsed_ms,
        "sequences_per_second": sum(item.shape[1] for item in host_inputs)
        * 1000
        / elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-gate-raw", type=Path, required=True)
    parser.add_argument("--native-gate-audit", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--batch", type=int, choices=(1, 4, 16, 32, 64), required=True)
    parser.add_argument(
        "--concurrency", type=int, choices=(1, 8, 32, 64, 128), required=True
    )
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.concurrency < args.batch
        or args.concurrency % args.batch
        or args.blocks < 20
    ):
        parser.error("requires C >= B, C divisible by B and at least 20 blocks")
    qualification = args.output.with_suffix(".qualification.npz")
    qualification_meta = args.output.with_suffix(".qualification.meta.json")
    audit_path = args.output.with_suffix(".qualification.audit.json")
    meta_path = args.output.with_suffix(".meta.json")
    if any(
        path.exists()
        for path in (
            args.output,
            qualification,
            qualification_meta,
            audit_path,
            meta_path,
        )
    ):
        parser.error("output evidence already exists")
    manifest = json.loads(args.manifest.read_text())
    if manifest["revision"] != REVISION or manifest["file_count"] != 16:
        raise ValueError("pinned DEER source differs")
    for entry in manifest["files"]:
        if sha256(args.source_root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"pinned DEER file differs: {entry['path']}")
    native_raw = json.loads(args.native_gate_raw.read_text())
    native_audit = json.loads(args.native_gate_audit.read_text())
    if (
        native_audit["status"] != "PASS"
        or native_audit["paired_tensor_count"] != 21
        or native_audit["failed_elements"] != 0
        or native_audit["raw_sha256"] != sha256(args.native_gate_raw)
        or native_audit["source_manifest_sha256"] != sha256(args.manifest)
        or native_raw["source_manifest_sha256"] != sha256(args.manifest)
        or native_raw["jax_version"] != jax.__version__
        or native_raw["jaxlib_version"] != jaxlib.__version__
    ):
        raise ValueError("independent small-shape native gate is incomplete")
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    if "," in visible or jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise ValueError("exactly one assigned GPU must be visible")
    actual_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if (
        actual_uuid != args.expected_gpu_uuid
        or native_raw["device_uuid"] != actual_uuid
    ):
        raise ValueError("GPU differs from the qualified device")
    device_name = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()
    if "H100" not in device_name:
        raise ValueError("the H100 training control requires an H100")
    sys.path.insert(0, str(args.source_root))
    from deer.fseq1d import seq1d

    jax.config.update("jax_default_matmul_precision", "highest")
    keys = jax.random.split(jax.random.PRNGKey(20261009), LAYERS * 9)
    parameters = tuple(
        tuple(
            jax.random.normal(key, shape, dtype=jnp.float32) * 0.05
            for key, shape in zip(
                keys[index * 9 : (index + 1) * 9], SHAPES, strict=True
            )
        )
        for index in range(LAYERS)
    )
    groups = args.concurrency // args.batch
    inputs = (
        np.random.default_rng(20261009 + args.batch)
        .standard_normal((groups, STEPS, args.batch, WIDTH))
        .astype(np.float32)
        * 0.2
    )
    compiled = {
        "baseline": compile_mode(seq1d.Sequential()),
        "candidate": compile_mode(seq1d.DEER()),
    }
    qualified = qualify(parameters, inputs, compiled, qualification)
    qualification_meta.write_text(
        json.dumps(
            {
                "source_revision": REVISION,
                "runner_sha256": sha256(Path(__file__)),
                "source_manifest_sha256": sha256(args.manifest),
                "native_gate_raw_sha256": sha256(args.native_gate_raw),
                "native_gate_audit_sha256": sha256(args.native_gate_audit),
                "raw_sha256": sha256(qualification),
                "batch": args.batch,
                "concurrency": args.concurrency,
                "groups": groups,
                "steps": STEPS,
                "width": WIDTH,
                "layers": LAYERS,
                "dtype": "float32",
                "budget": BUDGET,
                "learning_rate": LEARNING_RATE,
            },
            indent=2,
        )
        + "\n"
    )
    audit = subprocess.run(
        [
            sys.executable,
            str(args.auditor),
            str(qualification),
            "--output",
            str(audit_path),
        ],
        check=False,
    )
    if audit.returncode:
        raise ValueError("independent actual-shape training qualification failed")
    report = json.loads(audit_path.read_text())
    if (
        report["status"] != "PASS"
        or report["failed_elements"] != 0
        or report["raw_sha256"] != sha256(qualification)
        or report["source_manifest_sha256"] != sha256(args.manifest)
    ):
        raise ValueError("independent qualification report differs")
    for _ in range(2):
        for arm in ("baseline", "candidate"):
            burst(parameters, inputs, compiled[arm][1], qualified[arm])
    with args.output.open("w") as stream:
        for block in range(args.blocks):
            order = (
                ("baseline", "candidate")
                if block % 2 == 0
                else ("candidate", "baseline")
            )
            times = {
                arm: burst(parameters, inputs, compiled[arm][1], qualified[arm])
                for arm in order
            }
            stream.write(
                json.dumps({"block": block, "status": "PASS", "order": order, **times})
                + "\n"
            )
            stream.flush()
    meta_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "SYNTHETIC_FOUR_LAYER_GRU_COMPLETE_TRAINING_STEP",
                "model": "four-layer GRU sequence-square loss and SGD update",
                "baseline": "DEER Sequential",
                "candidate": "DEER parallel",
                "source_revision": REVISION,
                "runner_sha256": sha256(Path(__file__)),
                "source_manifest_sha256": sha256(args.manifest),
                "native_gate_raw_sha256": sha256(args.native_gate_raw),
                "native_gate_audit_sha256": sha256(args.native_gate_audit),
                "qualification_sha256": sha256(qualification),
                "qualification_meta_sha256": sha256(qualification_meta),
                "qualification_audit_sha256": sha256(audit_path),
                "raw_sha256": sha256(args.output),
                "device_identity_sha256": hashlib.sha256(
                    actual_uuid.encode()
                ).hexdigest(),
                "gpu_model": "H100",
                "gpu_count": 1,
                "jax_version": jax.__version__,
                "jaxlib_version": jaxlib.__version__,
                "dtype": "float32",
                "batch": args.batch,
                "concurrency": args.concurrency,
                "groups": groups,
                "blocks": args.blocks,
                "steps": STEPS,
                "width": WIDTH,
                "layers": LAYERS,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
