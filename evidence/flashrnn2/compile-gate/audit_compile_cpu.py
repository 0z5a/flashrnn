"""Recompute saved eager/compiled outputs and gradients without compilation."""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("run")
args = parser.parse_args()
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "independent_audit", root / "audit_portable.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
controller = json.loads((root / "evidence" / f"{args.run}-controller.json").read_text())
assert controller["status"] == "COMPLETE"
torch.set_num_threads(1)
summary = []
pairs = 0
for job in controller["jobs"]:
    folder = root / "evidence" / args.run / job["name"]
    record = json.loads((folder / "result.json").read_text())
    assert job["returncode"] == (0 if record["status"] == "PASSED" else 1)
    assert record["atol"] == record["rtol"] == 1e-5 and record["shape"] == [2, 4, 1, 8]
    for name, sha in record["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
    file = folder / "tensors.pt"
    assert module.digest(file) == record["tensor_sha256"]
    data = torch.load(file, weights_only=True)
    for index, key in enumerate(("history", "final")):
        module.compare(
            data["actual"][index], data["expected"][index], 1e-5, record["checks"][key]
        )
        pairs += 1
        module.compare(
            data["chunk"][index],
            data["expected"][index],
            1e-5,
            record["checks"]["chunk_" + key],
        )
        pairs += 1
    for index, key in enumerate(("wx", "recurrent", "bias", "initial")):
        module.compare(
            data["actual_gradients"][index],
            data["expected_gradients"][index],
            1e-5,
            record["checks"]["gradient_" + key],
        )
        pairs += 1
        module.compare(
            data["chunk_gradients"][index],
            data["expected_gradients"][index],
            1e-5,
            record["checks"]["chunk_gradient_" + key],
        )
        pairs += 1
    passed = all(x["pass"] for x in record["checks"].values())
    assert passed == (record["status"] == "PASSED")
    if passed:
        assert len(record["timings"]) == 40
        for block in range(10):
            entries = [x for x in record["timings"] if x["block"] == block]
            assert [x["method"] for x in entries] == (
                ["eager", "compiled", "compiled", "eager"]
                if block % 2 == 0
                else ["compiled", "eager", "eager", "compiled"]
            )
        medians = {
            name: statistics.median(
                x["total_ms"] for x in record["timings"] if x["method"] == name
            )
            for name in ("eager", "compiled")
        }
        assert (
            medians == record["steady_total_median_ms"]
            and medians["eager"] / medians["compiled"] == record["local_probe_speedup"]
        )
    else:
        assert not record["timings"]
    summary.append(
        {
            key: record[key]
            for key in (
                "cell",
                "numerics",
                "status",
                "first_forward_including_compile_ms",
                "first_backward_including_compile_ms",
                "first_chunk_including_compile_ms",
                "steady_recompile",
                "process_peak_rss_mib",
                "completed_child_peak_rss_mib",
            )
        }
        | {
            "median_total_ms": record.get("steady_total_median_ms"),
            "local_probe_speedup": record.get("local_probe_speedup"),
            "failed_checks": {
                k: v for k, v in record["checks"].items() if not v["pass"]
            },
        }
    )
result = {
    "audit": "PASS",
    "cases": len(summary),
    "passed": sum(x["status"] == "PASSED" for x in summary),
    "pairs_recomputed": pairs,
    "rows": summary,
    "auditor_sha256": module.digest(Path(__file__)),
    "performance_scope": "UNQUALIFIED_LOCAL_SMALL_TENSOR_PROBE_NOT_MODEL_E2E",
}
path = root / "evidence" / f"{args.run}-audit.json"
assert not path.exists()
path.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps({k: result[k] for k in ("audit", "cases", "passed", "pairs_recomputed")})
)
