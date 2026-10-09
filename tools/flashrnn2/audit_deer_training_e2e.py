"""Independently audit actual-shape DEER training outputs before timing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_deer_native_gate import compare

REVISION = "409140ff98d2426e49090e3552a63da10d9cd6dc"
WIDTH = 32
STEPS = 128
LAYERS = 4
SHAPES = ((WIDTH, WIDTH), (WIDTH, WIDTH), (WIDTH,)) * 3
LEARNING_RATE = 1e-3


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output already exists")
    meta_path = args.input.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    raw_bytes = args.input.read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    if (
        meta["source_revision"] != REVISION
        or meta["raw_sha256"] != raw_sha
        or meta["budget"] != 1e-4
        or meta["learning_rate"] != LEARNING_RATE
        or (meta["steps"], meta["width"], meta["layers"], meta["dtype"])
        != (STEPS, WIDTH, LAYERS, "float32")
        or meta["groups"] * meta["batch"] != meta["concurrency"]
    ):
        raise ValueError("DEER training qualification contract differs")
    groups = meta["groups"]
    batch = meta["batch"]
    names = [f"layer{layer}.{index}" for layer in range(LAYERS) for index in range(9)]
    expected_keys = {"inputs", *(f"parameter.{name}" for name in names)}
    for arm in ("baseline", "candidate"):
        expected_keys.add(f"{arm}.loss")
        expected_keys.update(f"{arm}.history.{layer}" for layer in range(LAYERS))
        expected_keys.update(f"{arm}.success.{layer}" for layer in range(LAYERS))
        expected_keys.update(f"{arm}.gradient.{name}" for name in names)
        expected_keys.update(f"{arm}.updated.{name}" for name in names)
    failures = 0
    maximum = 0.0
    failed_rows = []
    paired_count = 0
    forward_count = 0
    update_count = 0
    loss_count = 0

    def check(actual: np.ndarray, reference: np.ndarray, label: str) -> None:
        nonlocal failures, maximum
        failed, delta = compare(actual.astype(np.float64), reference.astype(np.float64))
        failures += failed
        maximum = max(maximum, delta)
        if failed:
            failed_rows.append(
                {"tensor": label, "failed_elements": failed, "max_abs": delta}
            )

    with np.load(args.input, allow_pickle=False) as arrays:
        if set(arrays.files) != expected_keys:
            raise ValueError("missing or unexpected qualification tensor")

        def numeric(name: str, shape: tuple[int, ...]) -> np.ndarray:
            value = arrays[name]
            if (
                value.shape != shape
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"invalid qualification tensor: {name}")
            return value.astype(np.float64)

        inputs = numeric("inputs", (groups, STEPS, batch, WIDTH))
        parameters = [
            [
                numeric(f"parameter.layer{layer}.{index}", shape)
                for index, shape in enumerate(SHAPES)
            ]
            for layer in range(LAYERS)
        ]
        for arm in ("baseline", "candidate"):
            numeric(f"{arm}.loss", (groups,))
            for layer in range(LAYERS):
                numeric(f"{arm}.history.{layer}", (groups, STEPS, batch, WIDTH))
                success = arrays[f"{arm}.success.{layer}"]
                if (
                    success.shape != (groups, STEPS, batch, WIDTH)
                    or success.dtype != np.bool_
                    or not success.all()
                ):
                    raise ValueError("DEER convergence evidence differs")
            for layer in range(LAYERS):
                for index, shape in enumerate(SHAPES):
                    name = f"layer{layer}.{index}"
                    numeric(f"{arm}.gradient.{name}", (groups, *shape))
                    numeric(f"{arm}.updated.{name}", (groups, *shape))
        for group in range(groups):
            reference = inputs[group]
            for layer in range(LAYERS):
                reference = numpy_layer(
                    reference, [value for value in parameters[layer]]
                )
                for arm in ("baseline", "candidate"):
                    actual = arrays[f"{arm}.history.{layer}"][group]
                    check(actual, reference, f"{arm}.history.{layer}.group{group}")
                    forward_count += 1
            expected_loss = np.asarray(np.square(reference).mean(), dtype=np.float64)
            for arm in ("baseline", "candidate"):
                check(
                    arrays[f"{arm}.loss"][group],
                    expected_loss,
                    f"{arm}.loss.group{group}",
                )
                loss_count += 1
                for layer in range(LAYERS):
                    for index in range(9):
                        name = f"layer{layer}.{index}"
                        gradient = arrays[f"{arm}.gradient.{name}"][group]
                        updated = arrays[f"{arm}.updated.{name}"][group]
                        reference_update = (
                            parameters[layer][index] - LEARNING_RATE * gradient
                        )
                        check(
                            updated,
                            reference_update,
                            f"{arm}.update.{name}.group{group}",
                        )
                        update_count += 1
            for name in names:
                for kind in ("gradient", "updated"):
                    check(
                        arrays[f"candidate.{kind}.{name}"][group],
                        arrays[f"baseline.{kind}.{name}"][group],
                        f"paired.{kind}.{name}.group{group}",
                    )
                    paired_count += 1
            for layer in range(LAYERS):
                check(
                    arrays[f"candidate.history.{layer}"][group],
                    arrays[f"baseline.history.{layer}"][group],
                    f"paired.history.{layer}.group{group}",
                )
                paired_count += 1
            check(
                arrays["candidate.loss"][group],
                arrays["baseline.loss"][group],
                f"paired.loss.group{group}",
            )
            paired_count += 1
    report = {
        "status": "PASS" if failures == 0 else "FAIL_NUMERICAL",
        "raw_sha256": raw_sha,
        "qualification_meta_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        "source_revision": REVISION,
        "source_manifest_sha256": meta["source_manifest_sha256"],
        "native_gate_raw_sha256": meta["native_gate_raw_sha256"],
        "native_gate_audit_sha256": meta["native_gate_audit_sha256"],
        "batch": batch,
        "concurrency": meta["concurrency"],
        "groups": groups,
        "paired_tensor_count": paired_count,
        "independent_forward_count": forward_count,
        "independent_loss_count": loss_count,
        "optimizer_update_count": update_count,
        "failed_elements": failures,
        "max_abs": maximum,
        "failed_rows": failed_rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
