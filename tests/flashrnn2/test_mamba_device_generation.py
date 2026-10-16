"""Keep invalid values and cache divergence outside the generation gate."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/flashrnn2"))
from mamba_device_generation import compare


class DeviceGenerationComparisonTest(unittest.TestCase):
    def test_nonfinite_values_fail(self) -> None:
        for actual, expected in (
            (1.0, float("nan")),
            (1.0, float("inf")),
            (float("nan"), 1.0),
            (float("inf"), float("inf")),
        ):
            with self.subTest(actual=actual, expected=expected):
                result = compare(torch.tensor([actual]), torch.tensor([expected]), 1e-5)
                self.assertFalse(result["pass"])
                self.assertFalse(result["finite"])
                self.assertEqual(result["failed_elements"], 1)

    def test_cache_budget_rejects_local_corruption(self) -> None:
        expected = torch.tensor([-100.0, 0.0, 100.0])
        actual = expected.clone()
        self.assertTrue(compare(actual, expected, 1e-5)["pass"])
        actual[1] = 1e-4
        result = compare(actual, expected, 1e-5)
        self.assertFalse(result["pass"])
        self.assertTrue(result["finite"])
        self.assertEqual(result["failed_elements"], 1)


if __name__ == "__main__":
    unittest.main()
