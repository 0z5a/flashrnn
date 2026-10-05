"""Contract tests with synthetic paired timings, never GPU benchmark evidence."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools/flashrnn2/analyze_serving.py"


class ServingAnalysisTest(unittest.TestCase):
    def run_fixture(self, candidate: str, fault: str = "") -> None:
        arms = ["native_torch", candidate]
        metadata = {
            "verification_only": fault == "qualification",
            "status": "PASS",
            "device": "cuda",
            "baseline_name": arms[0],
            "candidate_name": arms[1],
            "arm_order": arms,
            "oracles": {"batch": 1, "generated_tokens": 2},
            "concurrency": [2],
            "paired_blocks": 20,
        }
        rows = []
        for block in range(20):
            order = arms if block % 2 == 0 else list(reversed(arms))
            for arm in order:
                ids = [0, 0] if fault == "duplicate_request" else [0, 1]
                tokens = [1] if fault == "truncated_tokens" else [1, 2]
                requests = [{"request_id": index, "token_ids": tokens} for index in ids]
                if fault == "token_mismatch" and arm == candidate and block == 3:
                    requests[0]["token_ids"] = [1, 3]
                rows.append(
                    {
                        "status": "PASS",
                        "token_match": True,
                        "phase": "measurement",
                        "batch": 1,
                        "concurrency": 2,
                        "block": block,
                        "arm": arm,
                        "order": order,
                        "requests": requests,
                        "model_batches": {"prefill": [1, 1], "decode": [1, 1]},
                        "output_tokens": 4,
                        "elapsed_ms": 10 if arm == arms[0] else 5,
                        "tokens_per_second": 400 if arm == arms[0] else 800,
                        "ttft_ms": {"p50": 1},
                        "request_latency_ms": {"p99": 2},
                    }
                )
        if fault == "missing_pair":
            rows.pop()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "synthetic.jsonl", root / "analysis"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            source.with_suffix(".meta.json").write_text(json.dumps(metadata))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--output", str(output)],
                capture_output=True,
                text=True,
                check=False,
            )
            if fault:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.with_suffix(".md").exists())
                self.assertFalse(output.with_suffix(".json").exists())
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(output.with_suffix(".json").read_text())[0]
                self.assertEqual(summary["baseline"], "native_torch")
                self.assertEqual(summary["candidate"], candidate)
                self.assertAlmostEqual(summary["paired_speedup"], 2)
                table = output.with_suffix(".md").read_text()
                self.assertIn("native_torch tok/s", table)
                self.assertIn(candidate + " tok/s", table)

    def test_native_pairs(self) -> None:
        for candidate in ["torchscript", "decode_graph"]:
            with self.subTest(candidate=candidate):
                self.run_fixture(candidate)

    def test_ineligible_or_unmatched_cohorts(self) -> None:
        for fault in [
            "qualification",
            "duplicate_request",
            "truncated_tokens",
            "token_mismatch",
            "missing_pair",
        ]:
            with self.subTest(fault=fault):
                self.run_fixture("torchscript", fault)


if __name__ == "__main__":
    unittest.main()
