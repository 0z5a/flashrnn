"""Contract tests with synthetic paired timings, never GPU benchmark evidence."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools/flashrnn2/analyze_serving.py"


class ServingAnalysisTest(unittest.TestCase):
    def run_fixture(
        self,
        candidate: str,
        fault: str = "",
        prefill_calls: int = 1,
        cache_checks: int = 2,
        baseline: str = "native_torch",
        batch: int = 1,
        concurrency: int = 2,
        measurement_oracles: bool = False,
    ) -> None:
        arms = [baseline, candidate]
        groups = concurrency // batch
        metadata = {
            "verification_only": fault == "qualification",
            "status": "PASS",
            "device": "cuda",
            "baseline_name": arms[0],
            "candidate_name": arms[1],
            "arm_order": arms,
            "oracles": {
                "batch": batch,
                "generated_tokens": 2,
                "prefill_model_calls": prefill_calls,
                "cache_checks": cache_checks,
                "cache_check_names": [f"state_{i}" for i in range(cache_checks)],
                "measurement_oracle_checks": measurement_oracles,
            },
            "concurrency": [concurrency],
            "paired_blocks": 20,
        }
        rows = []
        for arm in arms:
            if fault == "warmup_missing" and arm == candidate:
                continue
            logits = [
                {"group": group, "step": step, "errors": [{"pass": True}]}
                for group in range(groups)
                for step in range(2)
            ]
            caches = [
                {
                    "group": group,
                    "errors": [
                        {"tensor": f"state_{i}", "pass": True}
                        for i in range(cache_checks)
                    ],
                }
                for group in range(groups)
            ]
            if fault == "warmup_logits" and arm == candidate:
                logits[0]["errors"][0]["pass"] = False
            if fault == "warmup_duplicate_step" and arm == candidate:
                logits[0]["step"] = 1
            if fault == "warmup_cache" and arm == candidate:
                caches[0]["errors"][0]["pass"] = False
            if fault == "warmup_cache_duplicate" and arm == candidate:
                caches[0]["errors"][1]["tensor"] = "state_0"
            rows.append(
                {
                    "status": "PASS",
                    "token_match": True,
                    "phase": "qualification_and_warmup",
                    "batch": batch,
                    "concurrency": concurrency,
                    "arm": arm,
                    "request_count": concurrency,
                    "peak_active_requests": concurrency,
                    "logit_checks": logits,
                    "cache_checks": caches,
                    "cache_check_step": 1,
                }
            )
        for block in range(20):
            order = arms if block % 2 == 0 else list(reversed(arms))
            for arm in order:
                ids = (
                    [0] * concurrency
                    if fault == "duplicate_request"
                    else list(range(concurrency))
                )
                tokens = [1] if fault == "truncated_tokens" else [1, 2]
                requests = [{"request_id": index, "token_ids": tokens} for index in ids]
                if fault == "token_mismatch" and arm == candidate and block == 3:
                    requests[0]["token_ids"] = [1, 3]
                elapsed = 10 if arm == arms[0] else 5
                token_rate = concurrency * 2 * 1000 / elapsed
                if fault == "inflated_throughput" and arm == candidate:
                    token_rate += 1
                logits = [
                    {"group": group, "step": step, "errors": [{"pass": True}]}
                    for group in range(groups)
                    for step in range(2)
                ]
                caches = [
                    {
                        "group": group,
                        "errors": [
                            {"tensor": f"state_{i}", "pass": True}
                            for i in range(cache_checks)
                        ],
                    }
                    for group in range(groups)
                ]
                if arm == candidate and block == 3:
                    if fault == "measurement_logits":
                        logits[0]["errors"][0]["pass"] = False
                    if fault == "measurement_cache":
                        caches[0]["errors"][0]["pass"] = False
                rows.append(
                    {
                        "status": "PASS",
                        "token_match": True,
                        "phase": "measurement",
                        "batch": batch,
                        "concurrency": concurrency,
                        "block": block,
                        "arm": arm,
                        "order": order,
                        "requests": requests,
                        "request_count": concurrency,
                        "peak_active_requests": concurrency,
                        "admission_batches": [batch] * groups,
                        "model_batches": {
                            "prefill": [batch]
                            * (groups * prefill_calls - (fault == "prefill_count")),
                            "decode": [batch] * groups,
                        },
                        "output_tokens": concurrency * 2,
                        "elapsed_ms": elapsed,
                        "tokens_per_second": token_rate,
                        "requests_per_second": concurrency * 1000 / elapsed,
                        "ttft_ms": {"p50": 1},
                        "request_latency_ms": {"p99": 2},
                        "logit_checks": (
                            []
                            if fault == "measurement_missing"
                            and arm == candidate
                            and block == 3
                            else logits
                        ),
                        "cache_checks": caches,
                        "cache_check_step": (
                            1
                            if fault == "measurement_wrong_step"
                            and arm == candidate
                            and block == 3
                            else 0
                        ),
                    }
                )
        if fault == "missing_pair":
            rows.pop()
        if fault == "unknown_phase":
            rows.append({**rows[-1], "phase": "unexpected"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "synthetic.jsonl", root / "analysis"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            metadata["rows_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            source.with_suffix(".meta.json").write_text(json.dumps(metadata))
            if fault == "tampered_raw":
                source.write_text(source.read_text() + "\n")
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
                self.assertEqual(summary["baseline"], baseline)
                self.assertEqual(summary["candidate"], candidate)
                self.assertAlmostEqual(summary["paired_speedup"], 2)
                table = output.with_suffix(".md").read_text()
                self.assertIn(baseline + " tok/s", table)
                self.assertIn(candidate + " tok/s", table)

    def test_native_pairs(self) -> None:
        for candidate in ["torchscript", "decode_graph", "device_trace"]:
            with self.subTest(candidate=candidate):
                self.run_fixture(candidate)

    def test_ineligible_or_unmatched_cohorts(self) -> None:
        for fault in [
            "qualification",
            "duplicate_request",
            "truncated_tokens",
            "token_mismatch",
            "missing_pair",
            "inflated_throughput",
            "warmup_missing",
            "warmup_logits",
            "warmup_duplicate_step",
            "warmup_cache",
            "warmup_cache_duplicate",
            "tampered_raw",
            "unknown_phase",
        ]:
            with self.subTest(fault=fault):
                self.run_fixture("torchscript", fault)

    def test_recurrent_prefill_and_state_contract(self) -> None:
        self.run_fixture("author_cuda", prefill_calls=5, cache_checks=6)
        self.run_fixture("author_cuda", "prefill_count", 5, 6)
        self.run_fixture(
            "flashrnn2_fp32", prefill_calls=5, cache_checks=6, baseline="author_cuda"
        )
        self.run_fixture("flashrnn2_fp32", "warmup_cache", 5, 6, baseline="author_cuda")
        self.run_fixture(
            "flashrnn2_fp32",
            prefill_calls=5,
            cache_checks=6,
            baseline="author_cuda",
            batch=64,
            concurrency=128,
            measurement_oracles=True,
        )
        for fault in (
            "measurement_logits",
            "measurement_cache",
            "measurement_missing",
            "measurement_wrong_step",
        ):
            with self.subTest(fault=fault):
                self.run_fixture(
                    "flashrnn2_fp32",
                    fault,
                    5,
                    6,
                    "author_cuda",
                    64,
                    128,
                    True,
                )


if __name__ == "__main__":
    unittest.main()
