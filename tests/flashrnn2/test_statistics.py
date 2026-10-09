import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "paired_analysis", Path(__file__).parents[2] / "tools/flashrnn2/analyze.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class StatisticsTest(unittest.TestCase):
    def test_identity_has_no_speedup(self) -> None:
        result = analysis.summarize([(float(i), float(i)) for i in range(1, 21)])
        self.assertEqual(result["paired_speedup"], 1)
        self.assertEqual(result["ci95"], [1, 1])

    def test_speed_and_latency_are_distinct(self) -> None:
        result = analysis.summarize([(2.0, 1.0)] * 20)
        self.assertAlmostEqual(result["throughput_gain_percent"], 100)
        self.assertAlmostEqual(result["latency_reduction_percent"], 50)

    def test_invalid_samples_rejected(self) -> None:
        for pairs in ([(1.0, 1.0)], [(0.0, 1.0)] * 20, [(1.0, math.nan)] * 20):
            with self.assertRaises(ValueError):
                analysis.summarize(pairs)


if __name__ == "__main__":
    unittest.main()
