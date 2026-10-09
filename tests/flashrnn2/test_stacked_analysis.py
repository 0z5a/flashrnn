"""Check that synthetic stacked-model tables reproduce their paired evidence."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ANALYZER = Path(__file__).parents[2] / "tools/flashrnn2/analyze_stacked_e2e.py"


class StackedAnalysisTest(unittest.TestCase):
    def test_qualified_table_omits_device_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, meta = self.write_fixture(root)
            result = self.analyze(raw, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "report.json").read_text())
            self.assertAlmostEqual(report["paired_speedup"], 2.0)
            self.assertNotIn("device_uuid", report)
            self.assertNotIn("GPU-private", (root / "report.md").read_text())
            self.assertEqual(meta["paired_blocks"], 20)

    def test_changed_raw_and_inflated_throughput_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, meta = self.write_fixture(root)
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            rows[0]["baseline"]["responses_per_second"] *= 10
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertNotEqual(self.analyze(raw, root).returncode, 0)
            meta["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
            raw.with_suffix(".meta.json").write_text(json.dumps(meta))
            result = self.analyze(raw, root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("throughput disagree", result.stderr)

    @staticmethod
    def analyze(raw: Path, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ANALYZER), str(raw), "--output", str(root / "report")],
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def write_fixture(root: Path) -> tuple[Path, dict]:
        raw = root / "timing.jsonl"
        rows = []
        for block in range(20):
            rows.append(
                {
                    "block": block,
                    "order": ["baseline", "candidate"]
                    if block % 2 == 0
                    else ["candidate", "baseline"],
                    "baseline": {
                        "elapsed_ms": 20.0,
                        "responses_per_second": 1600.0,
                        "last_response_ms": 19.0,
                    },
                    "candidate": {
                        "elapsed_ms": 10.0,
                        "responses_per_second": 3200.0,
                        "last_response_ms": 9.0,
                    },
                    "speedup": 2.0,
                    "status": "PASS",
                }
            )
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        qualification = raw.with_suffix(".qualification.pt")
        qualification.write_bytes(b"synthetic qualification fixture")
        meta = {
            "evidence_version": 2,
            "status": "PASS",
            "paired_blocks": len(rows),
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "qualification_sha256": hashlib.sha256(
                qualification.read_bytes()
            ).hexdigest(),
            "qualification_max_abs": 0.0,
            "baseline": "haste",
            "candidate": "flashrnn2_triton_basic",
            "dtype": "torch.float16",
            "device_uuid": "GPU-private-test-fixture",
            "model": {"cell": "gru", "layers": 4},
            "workload": {"batch": 16, "concurrency": 32, "groups": 2},
        }
        raw.with_suffix(".meta.json").write_text(json.dumps(meta))
        return raw, meta


if __name__ == "__main__":
    unittest.main()
