"""Multi-GPU cluster forward over NVSwitch multicast: correctness and per-step time (hybrid_fwc_tmem.recurrence_mg;
docs/flashrnn2/tmem-beta-handoff.md, "Multi-GPU"). Report only.

Usage: tmem_mg_bench.py --devices 4,5,6,7 [--widths 512,1024] [--cells lstm:float16,slstm:bfloat16] [--calls 20]
       [--steps 256] --out FILE
Per (cell, D), B = 16, hidden output in the input dtype:
  tilelang        the library's TileLang forward on devices[0] (reference for max_abs / rel_fro)
  tmem-c16        the single-GPU tensor-memory cluster forward on devices[0], where D fits (D <= 512)
  mg-p{P}-c{C}    recurrence_mg over the first P devices (P = 2, 4, 8 as available and divisible), default cluster C
Records median wall-clock µs per call (the multi-GPU call synchronizes every device), µs per step, phase timers of a
separate prof build (barrier / remote / gemm / pointwise per rank), max_abs and rel_fro to TileLang, repeat-call
bitwise stability, and the per-rank err words. A missing NVSwitch / multicast support is a recorded error.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, hybrid_fwc_tmem
from flashrnn.flashrnn2.execution_policy import KernelConfig
from flashrnn.flashrnn2.state_contract import begin_sequence

TL = {256: dict(family="tilelang", slice_width=32, sync="cluster"),
      512: dict(family="tilelang", slice_width=32, sync="cluster"),
      768: dict(family="tilelang", slice_width=16, sync="grid"),
      1024: dict(family="tilelang", slice_width=16, sync="grid")}


def median(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", required=True)
    ap.add_argument("--widths", default="512,1024")
    ap.add_argument("--cells", default="lstm:float16,slstm:bfloat16")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    devices = [int(x) for x in a.devices.split(",")]
    dev0 = torch.device(f"cuda:{devices[0]}")
    torch.cuda.set_device(dev0)
    rec = dict(started=time.time(), devices=devices, gpu=torch.cuda.get_device_name(dev0), steps=a.steps, rows=[])
    for cd in a.cells.split(","):
        cell, dtype = cd.split(":")
        for width in map(int, a.widths.split(",")):
            case = dict(cell=cell, dtype=dtype, width=width, batch=16, steps=a.steps, heads=1, init="warm")
            leaves, _, _ = bw_qualify.inputs(case, 7300 + width)
            wx, r, b, init = [x.to(dev0) for x in leaves]
            ctx = begin_sequence(init, cell, wx.dtype)
            row = dict(cell=cell, dtype=dtype, width=width, batch=16, arms={})
            ref = None
            arms = {}
            if width in TL:
                arms["tilelang"] = lambda: dispatch.recurrence(wx, r, b, init, cell, output_mode="hidden",
                                                               config=KernelConfig(**TL[width]))[0]
            if hybrid_fwc_tmem.supported(width, 16):
                arms["tmem-c16"] = lambda: hybrid_fwc_tmem.recurrence(wx, r, b, init, cell, sequence_context=ctx,
                                                                      output_mode="hidden", cluster=16)[0]
            for P in (2, 4, 8):
                if P > len(devices):
                    continue
                c = hybrid_fwc_tmem.default_cluster(width, P)
                if c is None:
                    continue
                arms[f"mg-p{P}-c{c}"] = (lambda P=P, c=c: hybrid_fwc_tmem.recurrence_mg(
                    wx, r, b, init, cell, sequence_context=ctx, devices=devices[:P], output_mode="hidden", cluster=c)[0])
            for name, fn in arms.items():
                ent = {}
                try:
                    with torch.no_grad():
                        o1 = fn().clone()
                        o2 = fn()
                        ent["repeat_bitwise"] = bool(torch.equal(o1, o2))
                        for d in devices:
                            torch.cuda.synchronize(d)
                        ts = []
                        for _ in range(a.calls):
                            t0 = time.perf_counter()
                            fn()
                            torch.cuda.synchronize(dev0)
                            ts.append((time.perf_counter() - t0) * 1e6)
                        ent["us_per_call"] = median(ts)
                        ent["us_per_step"] = ent["us_per_call"] / a.steps
                        if name == "tilelang":
                            ref = o1
                        elif ref is not None:
                            ent["max_abs_vs_tilelang"] = (o1.float() - ref.float()).abs().max().item()
                            ent["rel_fro_vs_tilelang"] = ((o1.float() - ref.float()).norm() / ref.float().norm()).item()
                        if name.startswith("mg-") or name.startswith("tmem-"):
                            diag = {}
                            if name.startswith("mg-"):
                                P = int(name.split("-")[1][1:])
                                c = int(name.split("-")[2][1:])
                                hybrid_fwc_tmem.recurrence_mg(wx, r, b, init, cell, sequence_context=ctx,
                                                              devices=devices[:P], output_mode="hidden", cluster=c,
                                                              prof=True, diagnostics=diag)
                            else:
                                hybrid_fwc_tmem.recurrence(wx, r, b, init, cell, sequence_context=ctx,
                                                           output_mode="hidden", cluster=16, prof=True, diagnostics=diag)
                            ent["phases_us_per_step"] = diag.get("phases_us_per_step")
                            ent["phases_per_rank"] = diag.get("phases_per_rank")
                            ent["shared_bytes"] = diag.get("shared_bytes")
                            ent["multicast_granularity"] = diag.get("multicast_granularity")
                except Exception:  # noqa: BLE001
                    ent["error"] = traceback.format_exc()[-1500:]
                row["arms"][name] = ent
                ph = ent.get("phases_us_per_step") or {}
                print(f"{cell} D{width} {name:12s} {ent.get('us_per_step', 0):7.2f} us/step  bar {ph.get('barrier', 0):.2f} "
                      f"remote {ph.get('remote', 0):.2f} gemm {ph.get('gemm', 0):.2f} pw {ph.get('pointwise', 0):.2f}  "
                      f"max_abs {ent.get('max_abs_vs_tilelang', float('nan')):.2e} repeat={ent.get('repeat_bitwise')} "
                      f"err={(ent.get('error') or '')[-160:]}", flush=True)
                torch.cuda.empty_cache()
            rec["rows"].append(row)
    rec["finished"] = time.time()
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
