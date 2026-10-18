"""Race verification of the 2-D split CUDA backward (flashrnn2/hybrid_bw2d.cu), rule of max-width.md section 12.

Usage: hybrid_bw2d_race.py --task JSON --out FILE
task: {"cell", "dtype", "width", "batch", "steps", "runs": 21, "batch_rows": 16 | 32 | 64 (default: by batch),
       "dg_dtype": null | "input" (the FRNN_DGLP build, max-width.md section 36),
       "honly": false | true (the FRNN_HONLY build reading the rounded h gradient, max-width.md section 40)}
Variants: correct (content check of every staged dG operand and the per-K-block step stamps read before every partial
sum; run 0 without delays, runs 1.. with seeded 0-4 us delays), no barrier B1 (partials), no barrier B2 (exchanged
dG); each broken variant keeps the barriers of the last step. Each run's outputs (dG, d_initial) are compared bitwise
with the uninstrumented kernel. Also recorded: the largest difference of dG and d_initial to the 1-D kernel
(hybrid_bw.backward, same batch tile; a different summation order, so not bitwise).
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
from flashrnn.flashrnn2 import dispatch, hybrid_bw, hybrid_bw2d
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
        leaves, cot_h, cot_f = bw_qualify.inputs(case, 9300 + task["width"])
        wx, r, b, init = leaves
        with torch.no_grad():
            history, _ = dispatch.recurrence(wx, r, b, init, task["cell"], output_mode="full",
                                             snapshot_dtype=torch.float32, config=default_config(task["width"]))
        d_history = torch.zeros_like(history)
        d_history[0] += cot_h.float()
        d_final = torch.zeros(init.shape, device=wx.device, dtype=torch.float32) + cot_f.float()
        args = (wx, r, b, init, history, d_history, d_final, task["cell"])
        kw = dict(batch_rows=task.get("batch_rows"))
        if task.get("honly"):
            d_hidden = cot_h.to(wx.dtype).contiguous()
            d_history = torch.zeros_like(history)  # the 1-D reference: the same rounded h gradient, zero-filled
            d_history[0] += d_hidden.float()
            args = (wx, r, b, init, history, None, d_final, task["cell"])
            kw.update(d_hidden=d_hidden)
        if task.get("dg_dtype"):
            kw.update(dg_dtype=task["dg_dtype"], wgrad="bmm_lp")
        ref = hybrid_bw2d.backward(*args, **kw)
        one_d = hybrid_bw.backward(*args[:5], d_history, *args[6:], batch_rows=kw["batch_rows"])
        torch.cuda.synchronize()
        scale = lambda x: x.abs().max().clamp_min(1e-30)  # noqa: E731
        rec["vs_1d"] = dict(dg_max_abs=(ref[0].float() - one_d[0]).abs().max().item(),
                            dg_max_rel=((ref[0].float() - one_d[0]).abs().max() / scale(one_d[0])).item(),
                            d_init_max_rel=((ref[3] - one_d[3]).abs().max() / scale(one_d[3])).item())
        runs = task.get("runs", 21)
        for variant in (None, "no_b1", "no_b2"):
            rows = []
            for i in range(runs):
                v = dict(check=True, delay_seed=None if i == 0 else 1000 * i + 13, broken=variant)
                out = hybrid_bw2d.backward(*args, _verify=v, **kw)
                torch.cuda.synchronize()
                same = torch.equal(out[0], ref[0]) and torch.equal(out[3], ref[3])
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
    print(json.dumps(dict(pass_=rec.get("pass_"), vs_1d=rec.get("vs_1d"),
                          summary={k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in rec["variants"].items()},
                          error=(rec.get("error") or "")[-300:])))
    return 0 if rec.get("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
