"""Dense Newton correctness and bounded-failure exploration; no speed claims."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from flashrnn.flashrnn2.newton_dense import affine_scan
from flashrnn.flashrnn2.newton_dense import recurrence as newton
from flashrnn.flashrnn2.reference import SIZES
from flashrnn.flashrnn2.reference import recurrence as sequential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    source = Path(newton.__code__.co_filename)
    metadata = {
        "scope": "CPU_FP64_DENSE_NEWTON_MATHEMATICAL_QUALIFICATION",
        "started": time.time(),
        "torch": torch.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(
            Path(sequential.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gradient": "Autograd through executed Newton iterations and Jacobians; detached stopping decision",
        "forward_contract": {"atol": 1e-8, "rtol": 1e-7},
        "gradient_contract": {"atol": 1e-7, "rtol": 1e-6},
        "newton_residual_atol": 1e-10,
        "iteration_cap": 16,
        "status": "RUNNING",
        "gpu_executed": False,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    # Noncommuting dense matrices expose a reversed composition order.
    a = (torch.randn(2, 17, 8, 8, dtype=torch.float64) * 0.1).requires_grad_()
    b = torch.randn(2, 17, 8, dtype=torch.float64, requires_grad=True)
    exact = [b[:, 0]]
    for t in range(1, 17):
        exact.append(b[:, t] + (a[:, t] @ exact[-1][:, :, None]).squeeze(-1))
    exact = torch.stack(exact, 1)
    scanned = affine_scan(a, b)
    torch.testing.assert_close(scanned, exact, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad(scanned.square().sum(), (a, b))
    exact_grad = torch.autograd.grad(exact.square().sum(), (a, b))
    for actual, target in zip(actual_grad, exact_grad):
        torch.testing.assert_close(actual, target, atol=1e-11, rtol=1e-11)
    metadata["dense_scan_forward_and_gradient"] = "PASS"
    failures = 0
    with args.output.open("x") as handle:
        for cell in SIZES:
            gates, bias_gates, states = SIZES[cell]
            for case, steps, width, cap, scale, losses in (
                ("gradients", 17, 4, 16, 0.1, ("final", "history_and_final")),
                ("longer_sequence", 64, 8, 16, 0.1, ("none",)),
                ("iteration_cap_control", 17, 4, 1, 1.2, ("none",)),
            ):
                shapes = (
                    (2, steps, gates, 2, width),
                    (gates, 2, width, width),
                    (bias_gates, 2, width),
                    (states, 2, 1, 2, width),
                )
                values = [
                    torch.randn(shape, dtype=torch.float64) * 0.1 for shape in shapes
                ]
                values[1] = (
                    torch.randn(shapes[1], dtype=torch.float64) * scale / width**0.5
                )
                if cell == "slstm":
                    values[3][2].abs_().add_(1.5)
                for loss_mode in losses:
                    inputs = [
                        x.clone().requires_grad_(loss_mode != "none") for x in values
                    ]
                    expected = sequential(*inputs, cell)
                    result = newton(*inputs, cell, max_iterations=cap, atol=1e-10)
                    actual = (result.history, result.final)
                    errors = [
                        (x - y).abs().max().item() for x, y in zip(actual, expected)
                    ]
                    forward_pass = all(
                        torch.isclose(x, y, **metadata["forward_contract"]).all().item()
                        for x, y in zip(actual, expected)
                    )
                    gradient_errors = {}
                    gradient_pass = True
                    if loss_mode != "none":
                        probes = [torch.randn_like(x) for x in expected]
                        grads = []
                        for outputs in (expected, actual):
                            loss = (outputs[1] * probes[1]).sum()
                            if loss_mode == "history_and_final":
                                loss = loss + (outputs[0] * probes[0]).sum()
                            grads.append(torch.autograd.grad(loss, inputs))
                        for name, target, output in zip(
                            ("dWx", "dR", "db", "dinitial"), *grads
                        ):
                            gradient_errors[name] = (output - target).abs().max().item()
                            gradient_pass &= (
                                torch.isclose(
                                    output, target, **metadata["gradient_contract"]
                                )
                                .all()
                                .item()
                            )
                    passed = forward_pass and gradient_pass and result.converged
                    if case == "iteration_cap_control":
                        status = (
                            "CAP_EXHAUSTED"
                            if not result.converged
                            else "CONVERGED_WITHIN_CAP"
                        )
                    else:
                        status = "PASS" if passed else "FAILED"
                        failures += int(not passed)
                    row = {
                        "cell": cell,
                        "case": case,
                        "B_T_H_D": [2, steps, 2, width],
                        "loss": loss_mode,
                        "iterations": len(result.residuals),
                        "residuals": result.residuals,
                        "converged": result.converged,
                        "logical_jacobian_bytes": result.jacobian_bytes,
                        "max_abs_history_final": errors,
                        "forward_pass": forward_pass,
                        "gradient_max_abs": gradient_errors,
                        "gradient_pass": gradient_pass if gradient_errors else None,
                        "status": status,
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(json.dumps(row), flush=True)
    metadata.update(
        status="FAIL" if failures else "PASS_WITH_RETAINED_CAP_CONTROLS",
        failed_cases=failures,
        finished=time.time(),
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
