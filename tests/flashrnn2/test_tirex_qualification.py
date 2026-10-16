"""Negative controls for stale kernels and mismatched forecast cohorts."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "tirex_qualification", ROOT / "tools/flashrnn2/tirex_qualification.py"
)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


class TiRexQualificationTest(unittest.TestCase):
    def setUp(self):
        self.manifest = qualification.execution_manifest(
            ROOT, {"triton": "test-version"}, "pins"
        )

    def test_current_identity_accepted_and_legacy_rejected(self):
        qualification.require_manifest(
            {"execution_manifest": self.manifest}, self.manifest
        )
        with self.assertRaises(ValueError):
            qualification.require_manifest(
                {"adapter_sha256": "unchanged"}, self.manifest
            )

    def test_each_execution_source_invalidates_without_adapter_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in qualification.SOURCE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            for name in qualification.SOURCE_FILES:
                with self.subTest(name=name):
                    path = root / name
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n# changed\n")
                    current = qualification.execution_manifest(
                        root, {"triton": "test-version"}, "pins"
                    )
                    with self.assertRaises(ValueError):
                        qualification.require_manifest(
                            {"execution_manifest": self.manifest}, current
                        )
                    path.write_bytes(original)

    def test_runtime_pins_and_compile_options_invalidate(self):
        for key in ("runtime", "reference_source_pins_sha256", "compile_options"):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.manifest)
                changed[key] = "changed"
                with self.assertRaises(ValueError):
                    qualification.require_manifest(
                        {"execution_manifest": self.manifest}, changed
                    )

    def test_exact_diagnostic_cohort_required(self):
        report = {
            "execution_manifest": self.manifest,
            "status": "PASS",
            "budget": 1e-4,
            "batch": 2,
            "queued_series": 32,
            "fixture_sha256": "fixture",
            "qualification_gate_sha256": "gate",
            "comparisons": [{"pass": True}] * 2048,
        }
        audit = {"status": "PASS", "observed_pairs": 2048, "required_pairs": 2048}
        arguments = {
            "current": self.manifest,
            "gate_sha256": "gate",
            "fixture_sha256": "fixture",
            "batch": 2,
            "concurrency": 32,
        }
        qualification.require_diagnostic(report, audit, **arguments)
        for key, value in (
            ("budget", 0.01),
            ("queued_series", 128),
            ("fixture_sha256", "other"),
            ("qualification_gate_sha256", "old"),
            ("comparisons", [{"pass": True}] * 2047),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                qualification.require_diagnostic(
                    report | {key: value}, audit, **arguments
                )
        with self.assertRaises(ValueError):
            qualification.require_diagnostic(
                report, audit | {"status": "INCOMPLETE"}, **arguments
            )

    def test_diagnostic_raw_and_audit_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            audit_path = Path(directory) / "audit.json"
            path.with_suffix(".pt").write_bytes(b"frozen raw archive")
            report = {
                "execution_manifest": self.manifest,
                "status": "PASS",
                "budget": 1e-4,
                "batch": 2,
                "queued_series": 32,
                "fixture_sha256": "fixture",
                "qualification_gate_sha256": "gate",
                "comparisons": [{"pass": True}] * 2048,
                "raw_sha256": qualification.sha256(path.with_suffix(".pt")),
            }
            path.write_text(json.dumps(report))
            audit = {
                "status": "PASS",
                "observed_pairs": 2048,
                "required_pairs": 2048,
                "diagnostic_sha256": qualification.sha256(path),
                "raw_sha256": report["raw_sha256"],
            }
            audit_path.write_text(json.dumps(audit))
            arguments = {
                "current": self.manifest,
                "gate_sha256": "gate",
                "fixture_sha256": "fixture",
                "batch": 2,
                "concurrency": 32,
            }
            qualification.load_diagnostic(path, audit_path, **arguments)
            path.with_suffix(".pt").write_bytes(b"modified raw archive")
            with self.assertRaises(ValueError):
                qualification.load_diagnostic(path, audit_path, **arguments)
            path.with_suffix(".pt").write_bytes(b"frozen raw archive")
            audit_path.write_text(json.dumps(audit | {"diagnostic_sha256": "old"}))
            with self.assertRaises(ValueError):
                qualification.load_diagnostic(path, audit_path, **arguments)


if __name__ == "__main__":
    unittest.main()
