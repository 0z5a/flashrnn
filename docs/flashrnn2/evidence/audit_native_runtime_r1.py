"""Recompute persisted native tensors; retain runner-only traced comparisons."""

import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "evidence/native-runtime-window-r1-offbox/evidence"
torch.set_num_threads(1)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


early = json.loads((ROOT / "evidence/native-runtime-window-r1-early-metadata.json").read_text())
for name, record in early.items():
    assert (RAW / name).read_text() == record["text"]
    assert digest(RAW / name) == record["sha256"]

golden_path = ROOT / "models/mamba-130m/script-goldens-b1-t5.pt"
contracts = json.loads(golden_path.with_suffix(".meta.json").read_text())
assert digest(golden_path) == contracts["artifact_sha256"]
goldens = torch.load(golden_path, map_location="cpu", weights_only=True)
assert len(goldens) == 3
rows = []
checks_recomputed = 0
snapshots = {}
norm_reductions = []
for device in ("cpu", "cuda"):
    prefix = RAW / f"mamba-native-{device}-remote-r1"
    meta = json.loads(prefix.with_suffix(".meta.json").read_text())
    records = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
    assert len(records) == 3 and [r["case"] for r in records] == [0, 1, 2]
    assert prefix.with_suffix(".log").read_bytes() == prefix.with_suffix(".jsonl").read_bytes()
    assert digest(prefix.with_suffix(".pt")) == meta["tensor_sha256"]
    assert meta["oracles"] == contracts
    assert meta["native_parameters"] == 129135360 and meta["native_layers"] == 24
    assert not meta["checkpoint_tensor_identity_verified"]
    for path, sha in meta["source_sha256"].items():
        local = ROOT / path.split("flashrnn2-20261005/", 1)[1]
        assert digest(local) == sha, path
    values = torch.load(prefix.with_suffix(".pt"), map_location="cpu", weights_only=True)
    snapshots[device] = values
    for case, (native, original, record) in enumerate(zip(values, goldens, records, strict=True)):
        original_failures = []
        for phase in ("prefill", "decode"):
            for index, (value, target, logged) in enumerate(zip(native[phase], original[phase], record["native_vs_original_oracle"]["checks"][phase], strict=True)):
                assert value.shape == target.shape and value.dtype == target.dtype == torch.float32
                assert torch.isfinite(value).all() and torch.isfinite(target).all()
                budget = contracts["logits_contract" if index == 0 else "cache_contract"]
                delta = (value - target).abs()
                allowed = budget["atol"] + budget["rtol"] * target.abs()
                failed = delta > allowed
                normalized = delta / allowed
                coord = tuple(int(i) for i in torch.unravel_index(normalized.argmax(), normalized.shape))
                assert float(delta.max()) == logged["max_abs"]
                assert int(failed.sum()) == logged["failed_elements"]
                assert bool((~failed).all()) == logged["pass"]
                assert float(normalized[coord]) == logged["worst_normalized_error"]
                assert list(coord) == logged["worst_coordinate"]
                assert float(value[coord]) == logged["actual_at_worst"]
                assert float(target[coord]) == logged["expected_at_worst"]
                relative = float(delta.norm() / target.norm().clamp_min(1e-30))
                norm_reductions.append({"device": device, "case": case, "phase": phase,
                                        "tensor": logged["tensor"], "recomputed": relative,
                                        "reported": logged["relative_l2"],
                                        "difference": abs(relative - logged["relative_l2"])})
                if index:
                    assert failed.flatten(1).sum(1).tolist() == logged["failed_by_layer"]
                if failed.any():
                    original_failures.append({"phase": phase, "tensor": logged["tensor"], "elements": int(failed.sum())})
                checks_recomputed += 1
        assert torch.equal(native["next_ids"], original["next_ids"])
        assert torch.equal(native["decode"][0][:, -1].argmax(-1), original["decode"][0][:, -1].argmax(-1))
        assert record["native_vs_original_oracle"]["tokens_equal"]
        assert record["native_vs_original_oracle"]["pass"] == (not original_failures)
        traced = record["traced_vs_same_device_native"]
        flat = [c for phase in traced["checks"].values() for c in phase]
        assert traced["tokens_equal"] and traced["pass"] and all(c["pass"] for c in flat)
        assert all(c["failed_elements"] == 0 and math.isfinite(c["max_abs"]) for c in flat)
        rows.append({"device": device, "case": case, "same_platform_runner_pass": True,
                     "same_platform_runner_max_abs": max(c["max_abs"] for c in flat),
                     "original_oracle_pass": not original_failures,
                     "original_oracle_failures": original_failures})
    assert meta["same_device_native_passed_cases"] == 3
    assert meta["native_vs_original_oracle_passed_cases"] == sum(r["original_oracle_pass"] for r in rows if r["device"] == device)

ptx = json.loads((RAW / "native-ptx-jit-r1.json").read_text())
dump = ROOT / "evidence/numerical-window-r2-offbox/evidence/native-library-r5--dump-ptx.log"
assert digest(dump) == ptx["dump_sha256"]
assert digest(ROOT / "source/tools/flashrnn2/native_ptx_jit_probe.py") == ptx["source_sha256"]
blocks = dump.read_text().split("Fatbin ptx code:\n================\n")[1:]
assert len(blocks) == len(ptx["rows"]) == 6
for index, (block, record) in enumerate(zip(blocks, ptx["rows"], strict=True), 1):
    image = (".version" + block.split(".version", 1)[1]).encode()
    assert hashlib.sha256(image).hexdigest() == record["ptx_sha256"]
    assert record["record"] == index
    assert record["returncode"] == record["unload_returncode"] == 0
    assert record["error_name"] == "CUDA_SUCCESS" and not record["jit_error_log"]
    assert not record["kernel_launched"]

report = {"scope": "INDEPENDENT_PERSISTED_NATIVE_TENSOR_AND_RAW_RECORD_AUDIT",
          "tensor_checks_recomputed": checks_recomputed,
          "same_platform_traced_scope": "Runner reports only; fresh traced tensors were not saved",
          "early_files_match_archive": len(early), "all_native_tensors_finite": True,
          "all_greedy_tokens_equal_original": True, "rows": rows,
          "six_ptx_bytes_and_success_records_verified": True,
          "original_budgets_unchanged": True, "performance_claim": False,
          "relative_l2_diagnostic_reductions": norm_reductions,
          "source_sha256": digest(Path(__file__))}
(ROOT / "evidence/native-runtime-window-r1-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
