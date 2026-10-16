"""Check hash-pinned Haste Torch equations against the FlashRNN cell contract."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

from flashrnn.flashrnn2.reference import SIZES, recurrence

SOURCE_HASHES = {
    "lstm": "64dc001df365623cf61607753964a8002e2892691b197c43ffbd3bbfadf4ed26",
    "gru": "567e1835648f5b55f461676b4f0a7153ac6e33c5a3abaf48d85d5f9f4651c3e4",
}


def source_function(root: Path, cell: str):
    path = root / "frameworks/pytorch" / f"{cell}.py"
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SOURCE_HASHES[cell]
    name = f"{cell.upper()}Script"
    node = next(
        part
        for part in ast.parse(payload, filename=str(path)).body
        if isinstance(part, ast.FunctionDef) and part.name == name
    )
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
    return scope[name]


def run_case(root: Path, cell: str) -> list[dict[str, object]]:
    torch.manual_seed(20261005)
    batch, steps, width, input_size = 3, 7, 4, 5
    gates, bias_gates, states = SIZES[cell]
    shapes = (
        (batch, steps, input_size),
        (gates, 1, width, input_size),
        (gates, 1, width, width),
        (bias_gates, 1, width),
        (states, batch, 1, 1, width),
    )
    x, w, r, b, s = tuple(
        (0.1 * torch.randn(shape, dtype=torch.float64)).requires_grad_()
        for shape in shapes
    )
    wx = torch.einsum("bti,ghdi->btghd", x, w)
    history, final = recurrence(wx, r, b, s, cell)
    expected = (history[0, :, :, 0], final)
    script = source_function(root, cell)
    mask = x.new_empty(0)
    if cell == "lstm":
        order = [0, 2, 1, 3]
        kernel = w[order, 0].reshape(4 * width, input_size).T.contiguous()
        recurrent = r[order, 0].reshape(4 * width, width).T.contiguous()
        bias = b[order, 0].reshape(-1)
        h, c = script(
            False,
            0.0,
            x.transpose(0, 1),
            s[0, :, 0, 0],
            s[1, :, 0, 0],
            kernel,
            recurrent,
            bias,
            mask,
        )
        actual = (h[1:].transpose(0, 1), torch.stack((h[-1], c[-1]))[:, :, None, None])
    else:
        # FlashRNN input gates are reset/update/candidate; recurrent gates differ.
        kernel = w[[1, 0, 2], 0].reshape(3 * width, input_size).T.contiguous()
        recurrent = r[[2, 1, 0], 0].reshape(3 * width, width).T.contiguous()
        bias = torch.cat((b[2, 0], b[1, 0], b[3, 0]))
        recurrent_bias = torch.cat((b[2, 0] * 0, b[1, 0] * 0, b[0, 0]))
        h = script(
            False,
            0.0,
            x.transpose(0, 1),
            s[0, :, 0, 0],
            kernel,
            recurrent,
            bias,
            recurrent_bias,
            mask,
        )
        actual = (h[1:].transpose(0, 1), h[-1][None, :, None, None])
    output_error = max(
        (left - right).abs().max().item() for left, right in zip(actual, expected)
    )
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    coefficient = torch.linspace(
        -1, 1, expected[0].numel(), dtype=torch.float64
    ).reshape_as(expected[0])
    rows = []
    for final_only in (False, True):

        def loss(
            result: tuple[torch.Tensor, torch.Tensor], final_only: bool = final_only
        ) -> torch.Tensor:
            value = result[1].square().sum()
            return value if final_only else value + (result[0] * coefficient).sum()

        expected_grad = torch.autograd.grad(
            loss(expected), (x, w, r, b, s), retain_graph=not final_only
        )
        actual_grad = torch.autograd.grad(
            loss(actual), (x, w, r, b, s), retain_graph=not final_only
        )
        gradient_error = max(
            (left - right).abs().max().item()
            for left, right in zip(actual_grad, expected_grad)
        )
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)
        rows.append(
            {
                "cell": cell,
                "loss": "final_only" if final_only else "history_and_final",
                "batch_steps_width_input": [batch, steps, width, input_size],
                "source_sha256": SOURCE_HASHES[cell],
                "output_max_abs": output_error,
                "gradient_max_abs": gradient_error,
                "status": "PASS",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [row for cell in SOURCE_HASHES for row in run_case(args.source_root, cell)]
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(
        json.dumps({"status": "PASS", "cases": len(rows), "output": str(args.output)})
    )


if __name__ == "__main__":
    main()
