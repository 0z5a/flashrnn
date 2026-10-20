"""Per-CTA barrier arrival / release stamps (docs/flashrnn2/max-width.md section 82), report only.

Usage: barrier_trace.py --out FILE [--calls 5] [--steps 256]
LSTM FP16, B = 16, T = steps. Arms (one group, so every CTA is traced):
  fw2d-d512-p16, fw2d-d512-p32, fw2d-d1024-p32cr24, fw2d-d1024-p64     1-D grids, grid.sync B2 (N = 16 .. 64 CTAs)
  fw2d-d512-p32-pf, fw2d-d1024-p64-pf                                 split B2 with the Wx loads in between
  fwc-d512-c16-bx, fwc-d512-c16-push                                   cluster: mbarrier wait / cluster barrier
Per arm and call the kernel writes, for thread 0 of every CTA and step, globaltimer (ns) before and after the B2 /
exchange wait. The file holds the raw stamps relative to the call's first stamp ([T][CTAs][2]); barrier_trace_analyze.py
reads it (arrival spread, extreme-value ratio, wait).
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import hybrid_fw2d, hybrid_fwc
from flashrnn.flashrnn2.state_contract import begin_sequence

ARMS = {
    "fw2d-d512-p16": (512, "fw2d", dict(grid=(16, 1, 0))),
    "fw2d-d512-p32": (512, "fw2d", dict(grid=(32, 1, 0))),
    "fw2d-d1024-p32cr24": (1024, "fw2d", dict(grid=(32, 1, 24))),
    "fw2d-d1024-p64": (1024, "fw2d", dict(grid=(64, 1, 0))),
    "fw2d-d512-p32-pf": (512, "fw2d", dict(grid=(32, 1, 0), prefetch=True, stages=4)),
    "fw2d-d1024-p64-pf": (1024, "fw2d", dict(grid=(64, 1, 0), prefetch=True, stages=4)),
    "fwc-d512-c16-bx": (512, "fwc", dict(cluster=16, bulkx=True)),
    "fwc-d512-c16-push": (512, "fwc", dict(cluster=16)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--steps", type=int, default=256)
    a = ap.parse_args()
    rec = dict(started=time.time(), gpu=torch.cuda.get_device_name(), steps=a.steps, arms={})
    cases = {}
    for name, (width, kind, kw) in ARMS.items():
        if width not in cases:
            case = dict(cell="lstm", dtype="float16", width=width, batch=16, steps=a.steps, heads=1, init="warm")
            leaves, _, _ = bw_qualify.inputs(case, 9100 + width)
            cases[width] = leaves
        wx, r, b, init = cases[width]
        ctx = begin_sequence(init, "lstm", wx.dtype)
        ent = dict(width=width, kind=kind, options=kw, calls=[])
        try:
            for i in range(a.calls + 1):  # the first call builds and warms up; not recorded
                diag = {}
                with torch.no_grad():
                    if kind == "fwc":
                        hybrid_fwc.recurrence(wx, r, b, init, "lstm", sequence_context=ctx, output_mode="hidden",
                                              trace=True, diagnostics=diag, **kw)
                    else:
                        hybrid_fw2d.recurrence(wx, r, b, init, "lstm", sequence_context=ctx, output_mode="hidden",
                                               trace=True, diagnostics=diag, **kw)
                torch.cuda.synchronize()
                tr = diag["trace"]
                if i == 0:
                    ent["ctas"] = tr.shape[1]
                    continue
                base = tr[tr > 0].min().item()
                ent["calls"].append((tr - base).tolist())
            tr0 = torch.tensor(ent["calls"][0])
            arr = tr0[1:, :, 0].double()
            print(f"{name:22s} ctas={ent['ctas']} mean wait {((tr0[1:, :, 1] - tr0[1:, :, 0]).double().mean()):.0f} ns "
                  f"mean spread(max-mean) {(arr.max(1).values - arr.mean(1)).mean():.0f} ns "
                  f"mean std {arr.std(1).mean():.0f} ns", flush=True)
        except Exception:  # noqa: BLE001
            ent["error"] = traceback.format_exc()[-1500:]
            print(f"{name}: ERROR {ent['error'][-200:]}", flush=True)
        rec["arms"][name] = ent
    rec["finished"] = time.time()
    Path(a.out).write_text(json.dumps(rec, default=str))


if __name__ == "__main__":
    main()
