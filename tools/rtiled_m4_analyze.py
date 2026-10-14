"""M4 analysis: per-comparison hierarchical bootstrap from raw per-start blocks.

Same estimator as the PR #79 cohort: ratio = baseline_ms / candidate_ms, geometric
mean over all paired blocks; 10000 draws resampling process starts, then blocks
within each drawn start; seed 20261013 + comparison index (table order).
Comparisons without five complete PASS starts are reported, never dropped.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path, help="batch directory with control/ and science/")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--starts", type=int, default=5)
    p.add_argument("--blocks", type=int, default=20)
    args = p.parse_args()
    table_path = args.batch / "control/frozen-table.json"
    table = json.loads(table_path.read_text())
    rows, incomplete = [], []
    for index, comp in enumerate(table["comparisons"]):
        a, b, starts = [], [], []
        for start in range(args.starts):
            d = args.batch / "science/phase2" / comp["name"] / f"start{start}"
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            timing = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {}
            starts.append(dict(start=start, status=meta.get("status", "MISSING"), host=meta.get("host"),
                               gpu=meta.get("gpu_uuid")))
            if meta.get("status") != "PASS" or len(timing.get("blocks", [])) != args.blocks:
                continue
            blocks = timing["blocks"]
            assert [x["block"] for x in blocks] == list(range(args.blocks))
            assert all(x["order"] == ("ABBA" if x["block"] % 2 == 0 else "BAAB") for x in blocks)
            assert all(x["a_ms"] > 0 and x["b_ms"] > 0 and x["outputs_within_budget"] for x in blocks)
            a.append([x["a_ms"] for x in blocks])
            b.append([x["b_ms"] for x in blocks])
        base = {k: comp[k] for k in ("name", "experiment", "cell", "width", "batch", "concurrency", "steps")}
        base.update(baseline=json.dumps(comp["baseline"], sort_keys=True),
                    candidate=json.dumps(comp["candidate"], sort_keys=True), starts=starts)
        if len(a) != args.starts:
            incomplete.append(dict(base, complete_starts=len(a)))
            continue
        a, b = np.array(a), np.array(b)
        logs = np.log(a / b)
        rng = np.random.default_rng(20261013 + index)
        s = rng.integers(0, args.starts, (10000, args.starts))
        k = rng.integers(0, args.blocks, (10000, args.starts, args.blocks))
        ci = np.quantile(np.exp(logs[s[:, :, None], k].mean(axis=(1, 2))), [0.025, 0.975])
        rows.append(dict(base, speedup=float(np.exp(logs.mean())), ci_low=float(ci[0]), ci_high=float(ci[1]),
                         baseline_ms=float(np.median(np.median(a, axis=1))),
                         candidate_ms=float(np.median(np.median(b, axis=1))),
                         start_ratios=[float(np.exp(x.mean())) for x in logs],
                         candidate_faster=bool(ci[0] > 1), candidate_slower=bool(ci[1] < 1)))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = dict(table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(), comparisons=len(rows),
                   incomplete=incomplete, unavailable=table.get("unavailable", []), rows=rows,
                   statistics="10000-draw hierarchical bootstrap (starts, then blocks); geometric mean of "
                              "baseline/candidate; seed 20261013+index")
    (args.out / "m4-summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out / "m4-summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "name", "cell", "D", "B", "C", "T", "baseline", "candidate", "speedup",
                    "ci_low", "ci_high", "baseline_ms", "candidate_ms"])
        for r in rows:
            w.writerow([r["experiment"], r["name"], r["cell"], r["width"], r["batch"], r["concurrency"], r["steps"],
                        r["baseline"], r["candidate"], f"{r['speedup']:.4f}", f"{r['ci_low']:.4f}",
                        f"{r['ci_high']:.4f}", f"{r['baseline_ms']:.4f}", f"{r['candidate_ms']:.4f}"])
    print(json.dumps(dict(complete=len(rows), incomplete=len(incomplete), unavailable=len(summary["unavailable"]))))


if __name__ == "__main__":
    main()
