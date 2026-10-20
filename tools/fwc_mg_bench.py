"""Multi-GPU cluster forward over NVSwitch multicast on H100 (hybrid_fwc.recurrence_mg; docs/flashrnn2/max-width.md
section 84): correctness and per-step time. Report only.

Usage: fwc_mg_bench.py --devices 0,1,2,3,4,5,6,7 [--widths 512,1024] [--cells lstm:float16,slstm:bfloat16]
       [--calls 20] [--steps 256] --out FILE
Per (cell, D), B = 16, hidden output in the input dtype:
  oracle                 flashrnn2.grad_reference on the CPU in FP64 (the reference for rel_fro)
  tilelang               the library's TileLang forward on devices[0]
  fw2d                   hybrid_fw2d on devices[0] where D is in its CONFIGS (D >= 1024)
  fwc-c16-cr8-k2-bx-fm, fwc-wg-c16-cr32-k1-bx-fm-mc   the single-GPU cluster forward (D <= 768)
  mg-p{P}-c{C}-bx-fm     recurrence_mg over the first P devices (P = 2, 4, 8): mma.sync GEMM, bulk + multicast exchange
  mg-p{P}-c{C}-wg-bx-fm  the same with the wgmma GEMM (needs 4 D / (P C) = 128)
Records median wall-clock µs per call (the multi-GPU call synchronizes every device), µs per step, the phase timers of a
separate prof build (wait / remote / mma / pointwise / exchange, per rank), rel_fro and max_abs to the FP64 oracle (an
FP16 kernel: ~2e-4; BF16 ~2e-3; an order of magnitude more is an exchange / layout error), repeat-call bitwise
stability, and errors as text (no multicast support is a recorded error).
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, grad_reference, hybrid_fw2d, hybrid_fwc
from flashrnn.flashrnn2.execution_policy import KernelConfig
from flashrnn.flashrnn2.state_contract import begin_sequence

TL = {512: dict(family="tilelang", slice_width=32, sync="cluster"),
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
            oracle = grad_reference.recurrence(*[x.detach().double().cpu() for x in leaves], cell)[0][0]  # [B,T,H,D]
            arms = {}
            if width in TL:
                arms["tilelang"] = lambda: dispatch.recurrence(wx, r, b, init, cell, output_mode="hidden",
                                                               config=KernelConfig(**TL[width]))[0]
            if width in hybrid_fw2d.CONFIGS:
                arms["fw2d"] = lambda: hybrid_fw2d.recurrence(wx, r, b, init, cell, sequence_context=ctx,
                                                              output_mode="hidden")[0]
            if hybrid_fwc.supported(width, 16):
                arms["fwc-c16-cr8-k2-bx-fm"] = lambda: hybrid_fwc.recurrence(
                    wx, r, b, init, cell, sequence_context=ctx, output_mode="hidden", cluster=16, cr=8, ksplit=2,
                    bulkx=True, fastmath=True)[0]
            if hybrid_fwc.wgmma_supported(width, 16):
                arms["fwc-wg-c16-cr32-k1-bx-fm-mc"] = lambda: hybrid_fwc.recurrence(
                    wx, r, b, init, cell, sequence_context=ctx, output_mode="hidden", cluster=16, cr=min(32, width // 16),
                    ksplit=1, bulkx=True, fastmath=True, wgmma=True, mcast=True)[0]
            for P in (2, 4, 8):
                if P > len(devices):
                    continue
                c_mma, c_wg = hybrid_fwc.mg_clusters(width, P)
                if c_mma is not None:
                    arms[f"mg-p{P}-c{c_mma}-bx-fm"] = (lambda P=P, c=c_mma: hybrid_fwc.recurrence_mg(
                        wx, r, b, init, cell, sequence_context=ctx, devices=devices[:P], output_mode="hidden", cluster=c,
                        fastmath=True)[0])
                if c_wg is not None:
                    arms[f"mg-p{P}-c{c_wg}-wg-bx-fm"] = (lambda P=P, c=c_wg: hybrid_fwc.recurrence_mg(
                        wx, r, b, init, cell, sequence_context=ctx, devices=devices[:P], output_mode="hidden", cluster=c,
                        ksplit=1, fastmath=True, wgmma=True)[0])
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
                        od = o1.detach().double().cpu()
                        ent["max_abs_vs_oracle"] = (od - oracle).abs().max().item()
                        ent["rel_fro_vs_oracle"] = ((od - oracle).norm() / oracle.norm()).item()
                        if name.startswith("mg-"):
                            diag = {}
                            P = int(name.split("-")[1][1:])
                            c = int(name.split("-")[2][1:])
                            hybrid_fwc.recurrence_mg(wx, r, b, init, cell, sequence_context=ctx, devices=devices[:P],
                                                     output_mode="hidden", cluster=c, fastmath=True,
                                                     wgmma="-wg-" in name, ksplit=1 if "-wg-" in name else 2,
                                                     prof=True, diagnostics=diag)
                            ent["phases_us_per_step"] = diag.get("phases_us_per_step")
                            ent["phases_per_rank"] = diag.get("phases_per_rank")
                            ent["shared_bytes"] = diag.get("shared_bytes")
                            ent["cr"] = diag.get("cr")
                            ent["multicast_granularity"] = diag.get("multicast_granularity")
                        elif name.startswith("fwc-"):
                            diag = {}
                            kw = (dict(cluster=16, cr=min(32, width // 16), ksplit=1, bulkx=True, fastmath=True, wgmma=True,
                                       mcast=True) if "wg" in name else dict(cluster=16, cr=8, ksplit=2, bulkx=True,
                                                                             fastmath=True))
                            hybrid_fwc.recurrence(wx, r, b, init, cell, sequence_context=ctx, output_mode="hidden",
                                                  prof=True, diagnostics=diag, **kw)
                            ent["phases_us_per_step"] = diag.get("phases_us_per_step")
                except Exception:  # noqa: BLE001
                    ent["error"] = traceback.format_exc()[-1500:]
                row["arms"][name] = ent
                ph = ent.get("phases_us_per_step") or {}
                print(f"{cell} D{width} {name:28s} {ent.get('us_per_step', 0):7.2f} us/step  wait {ph.get('barrier', 0):.2f} "
                      f"remote {ph.get('remote', 0):.2f} mma {ph.get('mma', 0):.2f} pw {ph.get('pointwise', 0):.2f} "
                      f"xch {ph.get('exchange', 0):.2f}  rel_fro {ent.get('rel_fro_vs_oracle', float('nan')):.2e} "
                      f"repeat={ent.get('repeat_bitwise')} err={(ent.get('error') or '')[-160:]}", flush=True)
                torch.cuda.empty_cache()
            rec["rows"].append(row)
    rec["finished"] = time.time()
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
