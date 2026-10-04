"""CPU checks against the unchanged upstream public vanilla implementation."""

import unittest
from functools import partial

import torch

from flashrnn import flashrnn
from flashrnn.flashrnn2.layout import (
    coordinate_owners,
    pack_recurrent,
    unpack_recurrent,
)
from flashrnn.flashrnn2.reference import SIZES, recurrence


def inputs(
    cell: str,
    *,
    batch: int = 3,
    steps: int = 5,
    heads: int = 2,
    width: int = 4,
    nonzero: bool = True,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20261005)
    gates, bias_gates, states = SIZES[cell]

    def sample(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (
            scale * torch.randn(shape, generator=generator, dtype=torch.float64)
        ).requires_grad_()

    wx = sample((batch, steps, gates, heads, width), 0.1)
    recurrent = sample((gates, heads, width, width), 0.1 / width**0.5)
    bias = sample((bias_gates, heads, width), 0.05)
    initial = sample((states, batch, 1, heads, width), 0.05 if nonzero else 0)
    if cell == "slstm" and nonzero:
        with torch.no_grad():
            initial[2].abs_().add_(1)
    return wx, recurrent, bias, initial


class ReferenceTest(unittest.TestCase):
    def test_public_forward_and_all_gradients(self) -> None:
        for cell in SIZES:
            for nonzero in (False, True):
                for final_only in (False, True):
                    with self.subTest(
                        cell=cell, nonzero=nonzero, final_only=final_only
                    ):
                        tensors = inputs(cell, nonzero=nonzero)
                        actual = recurrence(*tensors, cell=cell)
                        expected = flashrnn(
                            *tensors[:3],
                            states=tensors[3],
                            function=cell,
                            backend="vanilla",
                            dtype="float32",
                        )
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
                        coefficient = torch.linspace(-1, 1, actual[0].numel()).reshape(
                            actual[0].shape
                        )

                        def loss(
                            outputs: tuple[torch.Tensor, torch.Tensor],
                            final_only: bool = final_only,
                            coefficient: torch.Tensor = coefficient,
                        ) -> torch.Tensor:
                            value = outputs[1].square().sum()
                            return (
                                value
                                if final_only
                                else value + (outputs[0] * coefficient).sum()
                            )

                        grad_actual = torch.autograd.grad(loss(actual), tensors)
                        grad_expected = torch.autograd.grad(loss(expected), tensors)
                        for a, b in zip(grad_actual, grad_expected):
                            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)

    def test_chunk_and_inputs_readonly(self) -> None:
        for cell in SIZES:
            with self.subTest(cell=cell):
                wx, r, b, s = inputs(cell)
                copies = tuple(x.clone() for x in (wx, r, b, s))
                full, final = recurrence(wx, r, b, s, cell)
                first, middle = recurrence(wx[:, :2], r, b, s, cell)
                last, chunk_final = recurrence(wx[:, 2:], r, b, middle, cell)
                torch.testing.assert_close(
                    full, torch.cat((first, last), dim=2), atol=0, rtol=0
                )
                torch.testing.assert_close(final, chunk_final, atol=0, rtol=0)
                for original, copy in zip((wx, r, b, s), copies):
                    torch.testing.assert_close(original, copy, atol=0, rtol=0)

    def test_directional_gradients(self) -> None:
        for cell in ("lstm", "slstm", "gru", "elman"):
            with self.subTest(cell=cell):
                tensors = inputs(cell, batch=1, steps=2, heads=1, width=2)
                self.assertTrue(
                    torch.autograd.gradcheck(
                        partial(recurrence, cell=cell),
                        tensors,
                        eps=1e-6,
                        atol=1e-5,
                        rtol=1e-4,
                        fast_mode=True,
                    )
                )

    def test_pack_and_owner_coverage(self) -> None:
        for width in (3, 64, 192, 256, 384, 512, 768):
            weight = torch.arange(4 * 2 * width * width).view(4, 2, width, width)
            torch.testing.assert_close(
                unpack_recurrent(pack_recurrent(weight), 4), weight
            )
            owners = coordinate_owners(width, 64)
            self.assertEqual(
                [i for start, end in owners for i in range(start, end)],
                list(range(width)),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
