"""Verify gate/bias/state and gradient mapping before using cuDNN timings."""

import unittest

import torch

from flashrnn.flashrnn2.reference import SIZES, recurrence
from flashrnn.flashrnn2.torch_layer import TorchLayer


class TorchLayerTest(unittest.TestCase):
    def test_hidden_final_and_all_gradients(self) -> None:
        torch.manual_seed(20261005)
        for cell in ("lstm", "gru", "elman"):
            for final_only in (False, True):
                with self.subTest(cell=cell, final_only=final_only):
                    gates, bias_gates, states = SIZES[cell]
                    batch, steps, heads, width, input_size = 3, 7, 2, 4, 5
                    shapes = (
                        (batch, steps, input_size),
                        (gates, heads, width, input_size),
                        (gates, heads, width, width),
                        (bias_gates, heads, width),
                        (states, batch, 1, heads, width),
                    )
                    x, w, r, b, s = tuple(
                        (0.1 * torch.randn(shape, dtype=torch.float64)).requires_grad_()
                        for shape in shapes
                    )
                    module = TorchLayer(w, r, b, cell)
                    wx = torch.einsum("bti,ghdi->btghd", x, w)
                    history, final = recurrence(wx, r, b, s, cell)
                    expected = (history[0], final)
                    actual = module(x, s)
                    for a, e in zip(actual, expected):
                        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)
                    coefficient = torch.linspace(
                        -1, 1, actual[0].numel(), dtype=torch.float64
                    ).reshape_as(actual[0])

                    def loss(
                        result: tuple[torch.Tensor, torch.Tensor],
                        final_only: bool = final_only,
                        coefficient: torch.Tensor = coefficient,
                    ) -> torch.Tensor:
                        value = result[1].square().sum()
                        return (
                            value
                            if final_only
                            else value + (result[0] * coefficient).sum()
                        )

                    expected_grad = torch.autograd.grad(loss(expected), (x, w, r, b, s))
                    parameters = tuple(module.parameters())
                    actual_grad = torch.autograd.grad(loss(actual), (x, s, *parameters))
                    torch.testing.assert_close(
                        actual_grad[0], expected_grad[0], atol=1e-11, rtol=1e-11
                    )
                    torch.testing.assert_close(
                        actual_grad[1], expected_grad[4], atol=1e-11, rtol=1e-11
                    )
                    for head in range(heads):
                        dw, dr, dbi, dbh = actual_grad[2 + head * 4 : 6 + head * 4]
                        torch.testing.assert_close(
                            dw.reshape(gates, width, input_size),
                            expected_grad[1][:, head],
                            atol=1e-11,
                            rtol=1e-11,
                        )
                        order = [1, 2, 0] if cell == "gru" else list(range(gates))
                        torch.testing.assert_close(
                            dr.reshape(gates, width, width),
                            expected_grad[2][order, head],
                            atol=1e-11,
                            rtol=1e-11,
                        )
                        if cell == "gru":
                            db = torch.stack(
                                (
                                    dbh.reshape(gates, width)[2],
                                    *dbi.reshape(gates, width).unbind(),
                                )
                            )
                        else:
                            db = dbi.reshape(gates, width)
                        torch.testing.assert_close(
                            db, expected_grad[3][:, head], atol=1e-11, rtol=1e-11
                        )


if __name__ == "__main__":
    unittest.main()
