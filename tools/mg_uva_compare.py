"""The existing Triton multi-GPU forward (multi_gpu.recurrence, peer stores through UVA) on the same GPUs as
fwc_mg_bench, for the multicast comparison (docs/flashrnn2/max-width.md section 84). Report only.

Usage: mg_uva_compare.py --devices 0,1,2,3 [--widths 512,1024] [--peers 2,4] [--calls 20] [--steps 256] --out FILE
Per (cell, D, P): median wall-clock µs per call and per step of multi_gpu.recurrence(alloc="uva") with its default
schedule, rel_fro to the CPU FP64 oracle, repeat-call bitwise stability; errors as text.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import grad_reference, multi_gpu


def median(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", required=True)
    ap.add_argument("--widths", default="512,1024")
    ap.add_argument("--peers", default="2,4")
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
            oracle = grad_reference.recurrence(*[x.detach().double().cpu() for x in leaves], cell)[0][0]
            for P in map(int, a.peers.split(",")):
                if P > len(devices):
                    continue
                ent = dict(cell=cell, dtype=dtype, width=width, peers=P)
                try:
                    devs = devices[:P]  # multi_gpu takes CUDA ordinals

                    def fn():
                        return multi_gpu.recurrence(wx, r, b, init, cell, devices=devs, output_mode="hidden")[0]
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
                            for d in devices[:P]:
                                torch.cuda.synchronize(d)
                            ts.append((time.perf_counter() - t0) * 1e6)
                        ent["us_per_call"] = median(ts)
                        ent["us_per_step"] = ent["us_per_call"] / a.steps
                        od = o1.detach().double().cpu()
                        if od.shape != oracle.shape:
                            od = od.reshape(oracle.shape)
                        ent["rel_fro_vs_oracle"] = ((od - oracle).norm() / oracle.norm()).item()
                except Exception:  # noqa: BLE001
                    ent["error"] = traceback.format_exc()[-1500:]
                rec["rows"].append(ent)
                print(f"{cell} D{width} uva-p{P} {ent.get('us_per_step', 0):7.2f} us/step rel_fro "
                      f"{ent.get('rel_fro_vs_oracle', float('nan')):.2e} repeat={ent.get('repeat_bitwise')} "
                      f"err={(ent.get('error') or '')[-200:]}", flush=True)
    rec["finished"] = time.time()
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
