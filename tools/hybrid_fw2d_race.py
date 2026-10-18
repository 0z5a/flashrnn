"""Race verification of the 2-D split CUDA forward (flashrnn2/hybrid_fw2d.cu), rule of max-width.md section 12.

Usage: hybrid_fw2d_race.py --task JSON --out FILE
task: {"cell", "dtype", "width", "batch", "heads": 1, "steps", "runs": 21, "batch_rows": 16 | 32 | 64 (default: by
       batch), "conc": false | true (the head-concurrent grid)}
Variants: correct (content check of every staged h operand against the stored h_{t-1}, and the per-(unit block, K
block) step stamps read before every partial sum; run 0 without delays, runs 1.. with seeded 0-4 us delays), no barrier
B1 (partials), no barrier B2 (exchanged h); each broken variant keeps the barriers of the last step. Each run's outputs
(hidden history, FP32 snapshot, and the FP32 carry) are compared bitwise with the uninstrumented kernel. Also recorded:
the largest difference of the history to the library's default forward (a different kernel and summation order, so
not bitwise).
Rule: the correct kernel counts zero violations in every run and its outputs are bitwise equal to the uninstrumented
kernel; a broken variant is detected when violations are counted in at least 80% of its runs and no run changes the
outputs without a violation.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, hybrid_fw2d
from flashrnn.flashrnn2.state_contract import begin_sequence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    task = json.loads(a.task)
    rec = dict(task=task, started=time.time(), gpu=torch.cuda.get_device_name(), variants={})
    try:
        case = dict(cell=task["cell"], dtype=task["dtype"], batch=task["batch"], steps=task["steps"],
                    width=task["width"], heads=task.get("heads", 1), init="warm")
        leaves, _, _ = bw_qualify.inputs(case, 9500 + task["width"])
        wx, r, b, init = leaves
        cell = task["cell"]
        kw = dict(batch_rows=task.get("batch_rows"), conc=bool(task.get("conc")))

        def run(verify=None):
            ctx = begin_sequence(init, cell, wx.dtype)
            with torch.no_grad():
                return hybrid_fw2d.recurrence(wx, r, b, init, cell, sequence_context=ctx, output_mode="hidden",
                                              snapshot_dtype=torch.float32, _verify=verify, **kw)
        ref = run()
        with torch.no_grad():
            lib, lib_carry = dispatch.recurrence(wx, r, b, init, cell, output_mode="hidden",
                                                 snapshot_dtype=torch.float32)
        torch.cuda.synchronize()
        scale = lambda x: x.abs().max().clamp_min(1e-30)  # noqa: E731
        rec["vs_library"] = dict(h_max_abs=(ref[0] - lib).abs().max().item(),
                                 h_max_rel=((ref[0] - lib).abs().max() / scale(lib)).item(),
                                 carry_max_rel=((ref[1] - lib_carry).abs().max() / scale(lib_carry)).item())
        runs = task.get("runs", 21)
        for variant in (None, "no_b1", "no_b2"):
            rows = []
            for i in range(runs):
                v = dict(check=True, delay_seed=None if i == 0 else 1000 * i + 17, broken=variant)
                out = run(v)
                torch.cuda.synchronize()
                same = torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
                rows.append(dict(run=i, after_read=v["after_read"], stale_partials=v["stale_partials"],
                                 outputs_equal=same))
            count = lambda x: x["after_read"] + x["stale_partials"]  # noqa: E731
            flagged = sum(count(x) > 0 for x in rows)
            silent = sum((not x["outputs_equal"]) and count(x) == 0 for x in rows)
            entry = dict(runs=rows, flagged=flagged, changed=sum(not x["outputs_equal"] for x in rows), silent=silent)
            if variant is None:
                entry["pass_"] = flagged == 0 and all(x["outputs_equal"] for x in rows)
            else:
                entry["detected"] = flagged >= 0.8 * runs and silent == 0
            rec["variants"][variant or "correct"] = entry
        rec["pass_"] = rec["variants"]["correct"]["pass_"] and all(
            rec["variants"][v]["detected"] for v in ("no_b1", "no_b2"))
    except Exception:  # noqa: BLE001
        rec.update(pass_=False, error=traceback.format_exc())
    rec["finished"] = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(dict(pass_=rec.get("pass_"), vs_library=rec.get("vs_library"),
                          summary={k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in rec["variants"].items()},
                          error=(rec.get("error") or "")[-300:])))
    return 0 if rec.get("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
