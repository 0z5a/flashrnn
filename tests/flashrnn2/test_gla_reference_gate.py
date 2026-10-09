"""Invalid mathematical references must never qualify a model comparison."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/flashrnn2"))
from gla_reference_gate import compare


class GLAComparisonTest(unittest.TestCase):
    def test_nonfinite_reference_rejected(self) -> None:
        for actual, expected in (
            (1.0, float("nan")),
            (1.0, float("inf")),
            (float("nan"), 1.0),
            (float("inf"), float("inf")),
        ):
            with self.subTest(actual=actual, expected=expected):
                result = compare(torch.tensor([actual]), torch.tensor([expected]), 1e-5)
                self.assertFalse(result["pass"])
                self.assertEqual(result["failed_elements"], 1)

    def test_finite_identity(self) -> None:
        values = torch.tensor([-100.0, 0.0, 100.0])
        result = compare(values, values.clone(), 1e-5)
        self.assertTrue(result["pass"])
        self.assertEqual(result["failed_elements"], 0)


if __name__ == "__main__":
    unittest.main()
