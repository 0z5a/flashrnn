"""Race verification of the CUDA large-D backward's exchange (flashrnn2/hybrid_bw.cu), same rule as race_verify.py.

Usage: hybrid_bw_race.py --task JSON --out FILE
task: {"cell", "dtype", "width", "batch", "steps", "runs": 21, "batch_rows": 16 | 32 | 64 (default: by batch),
       "cluster": 0 | 8 (default 0)}
With a cluster, the uninstrumented outputs are also compared bitwise with the plain (cluster 0) kernel's
("bitwise_vs_plain"; same accumulation order, so they must match).
Variants: correct (content check; run 0 without delays, runs 1.. with seeded 0-4 us delays), single buffer, no grid
barrier (the last one, before the tail, stays); with a cluster also a single stage buffer ("cl_single") and no
cluster barrier between rounds ("cl_no_sync", the last round's stays). With a cluster the content check also covers
the A operands as read through distributed shared memory (counter after_stage). Each run's outputs (dG, d_initial) are compared bitwise with the
uninstrumented kernel on the same inputs.
Rule (pre-registered, docs/flashrnn2/max-width.md section 12): the correct kernel passes when every run counts zero
violations and its outputs are bitwise equal to the uninstrumented reference; a broken variant is detected when
violations are counted in at least 80% of its runs, and no run may change the outputs without counting a violation.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, hybrid_bw
from flashrnn.flashrnn2.execution_policy import default_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    task = json.loads(a.task)
    rec = dict(task=task, started=time.time(), gpu=torch.cuda.get_device_name(), variants={})
    try:
        case = dict(cell=task["cell"], dtype=task["dtype"], batch=task["batch"], steps=task["steps"],
                    width=task["width"], heads=1, init="warm")
        leaves, cot_h, cot_f = bw_qualify.inputs(case, 9100 + task["width"])
        wx, r, b, init = leaves
        with torch.no_grad():
            history, _ = dispatch.recurrence(wx, r, b, init, task["cell"], output_mode="full",
                                             snapshot_dtype=torch.float32, config=default_config(task["width"]))
        d_history = torch.zeros_like(history)
        d_history[0] += cot_h.float()
        d_final = torch.zeros(init.shape, device=wx.device, dtype=torch.float32)
        d_final += cot_f.float()
        args = (wx, r, b, init, history, d_history, d_final, task["cell"])
        kw = dict(batch_rows=task.get("batch_rows"), cluster=task.get("cluster", 0))
        ref = hybrid_bw.backward(*args, **kw)
        torch.cuda.synchronize()
        if kw["cluster"]:
            plain = hybrid_bw.backward(*args, **dict(kw, cluster=0))
            torch.cuda.synchronize()
            rec["bitwise_vs_plain"] = torch.equal(plain[0], ref[0]) and torch.equal(plain[3], ref[3])
        runs = task.get("runs", 21)
        variants = (None, "single", "no_sync") + (("cl_single", "cl_no_sync") if kw["cluster"] else ())
        for variant in variants:
            rows = []
            for i in range(runs):
                v = dict(check=True, delay_seed=None if i == 0 else 1000 * i + 7, broken=variant)
                out = hybrid_bw.backward(*args, _verify=v, **kw)
                torch.cuda.synchronize()
                same = torch.equal(out[0], ref[0]) and torch.equal(out[3], ref[3])
                rows.append(dict(run=i, after_read=v["after_read"], after_mma=v["after_mma"],
                                 after_stage=v["after_stage"], outputs_equal=same))
            count = lambda x: x["after_read"] + x["after_mma"] + x["after_stage"]  # noqa: E731
            flagged = sum(count(x) > 0 for x in rows)
            silent = sum((not x["outputs_equal"]) and count(x) == 0 for x in rows)
            name = variant or "correct"
            entry = dict(runs=rows, flagged=flagged, changed=sum(not x["outputs_equal"] for x in rows), silent=silent)
            if variant is None:
                entry["pass_"] = flagged == 0 and all(x["outputs_equal"] for x in rows)
            else:
                entry["detected"] = flagged >= 0.8 * runs and silent == 0
            rec["variants"][name] = entry
        rec["pass_"] = rec["variants"]["correct"]["pass_"] and all(
            rec["variants"][v]["detected"] for v in variants[1:]) and rec.get("bitwise_vs_plain", True)
    except Exception:
        rec.update(pass_=False, error=traceback.format_exc())
    rec["finished"] = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(dict(pass_=rec.get("pass_"), summary={k: {kk: vv for kk, vv in v.items() if kk != "runs"}
                                                             for k, v in rec["variants"].items()},
                          error=(rec.get("error") or "")[-300:])))
    return 0 if rec.get("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
