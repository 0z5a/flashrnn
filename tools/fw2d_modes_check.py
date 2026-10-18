"""Bitwise consistency of the 2-D forward's output modes and snapshot dtypes (max-width.md section 56).

Usage: fw2d_modes_check.py --out FILE --cases JSON
Each case {"cell", "dtype", "width", "batch", "heads", "steps", "conc": bool}. One kernel computes every mode, so:
hidden (FP32) equals component h of full (FP32); last equals the last step of full's h; the input-dtype snapshot equals
the FP32 snapshot rounded to the input dtype (every component); the FP32 carry is identical in every mode and equals
the last step of full (FP32, every component).
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import hybrid_fw2d
from flashrnn.flashrnn2.state_contract import begin_sequence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cases", required=True)
    a = p.parse_args()
    rec = dict(started=time.time(), rows=[])
    for case in json.loads(a.cases):
        row = dict(case)
        try:
            c = dict(case, init="warm")
            leaves, _, _ = bw_qualify.inputs(c, 8100 + case["width"] + case["batch"])
            wx, r, b, init = leaves

            def run(mode, snap):
                with torch.no_grad():
                    return hybrid_fw2d.recurrence(wx, r, b, init, case["cell"],
                                                  sequence_context=begin_sequence(init, case["cell"], wx.dtype),
                                                  output_mode=mode, snapshot_dtype=snap, conc=bool(case.get("conc")))
            full, cf = run("full", torch.float32)
            hidden, ch = run("hidden", torch.float32)
            last, cl = run("last", torch.float32)
            full_lp, cfl = run("full", wx.dtype)
            row.update(hidden_eq_full_h=bool(torch.equal(hidden, full[0])),
                       last_eq_full_h=bool(torch.equal(last, full[0][:, -1:])),
                       snapshot_lp_eq_rounded=bool(torch.equal(full_lp, full.to(wx.dtype))),
                       carry_same_every_mode=bool(torch.equal(cf, ch) and torch.equal(cf, cl) and torch.equal(cf, cfl)),
                       carry_eq_last_step=bool(torch.equal(cf, full[:, :, -1:])))
            row["pass_"] = all(v for k, v in row.items() if k.endswith(("_eq_full_h", "_eq_rounded", "_mode", "_step")))
        except Exception:  # noqa: BLE001
            row.update(pass_=False, error=traceback.format_exc()[-1500:])
        rec["rows"].append(row)
        print(json.dumps(row), flush=True)
    rec.update(finished=time.time(), status="PASS" if all(x["pass_"] for x in rec["rows"]) else "FAIL")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
