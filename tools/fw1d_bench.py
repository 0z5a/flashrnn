"""Development bench for the B = 16 forward (docs/flashrnn2/max-width.md section 76), report only.

Usage: fw1d_bench.py --out FILE [--widths 512,1024] [--cells lstm:float16,slstm:bfloat16] [--calls 30]
Per (cell, D), B = 16, T = 256, the training forward (output "full", FP32 snapshot) of:
  tilelang            the library's TileLang forward (the measured-runtime selection)
  fw2d                the CUDA 2-D forward with its default grid (D in hybrid_fw2d.CONFIGS only)
  fw1d[...]           the CUDA forward with a 1-D grid (PC = D / 16, Q = 1, CR register k-steps) and the section 76
                      options: base, prefetch (split B2 with this step's Wx loads), stages (cp.async depth)
Records the median ms per call (CUDA events), the per-step phase times of the CUDA variants (phase-timer build), the
largest difference of every arm's output to tilelang's, and whether the fw1d variants are bitwise equal to fw1d base.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, hybrid_fw2d
from flashrnn.flashrnn2.execution_policy import KernelConfig
from flashrnn.flashrnn2.state_contract import begin_sequence

TL = {512: dict(family="tilelang", slice_width=32, sync="cluster"),
      1024: dict(family="tilelang", slice_width=16, sync="grid"),
      768: dict(family="tilelang", slice_width=16, sync="grid")}


def median_ms(fn, calls):
    fn(); fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(calls):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="512,1024")
    ap.add_argument("--cells", default="lstm:float16,slstm:bfloat16")
    ap.add_argument("--calls", type=int, default=30)
    a = ap.parse_args()
    rec = dict(started=time.time(), gpu=torch.cuda.get_device_name(), rows=[])
    for cd in a.cells.split(","):
        cell, dtype = cd.split(":")
        for width in map(int, a.widths.split(",")):
            case = dict(cell=cell, dtype=dtype, width=width, batch=16, steps=256, heads=1, init="warm")
            leaves, _, _ = bw_qualify.inputs(case, 4100 + width)
            wx, r, b, init = leaves
            ctx = begin_sequence(init, cell, wx.dtype)
            arms = {}

            def tl():
                with torch.no_grad():
                    return dispatch.recurrence(wx, r, b, init, cell, output_mode="full", snapshot_dtype=torch.float32,
                                               config=KernelConfig(**TL[width]))[0]
            arms["tilelang"] = (tl, None)
            if hybrid_fw2d.supported(width):
                arms["fw2d"] = (dict(), "cuda")
            pc = width // 16
            for cr in (0, 16):
                arms[f"fw1d-cr{cr}"] = (dict(grid=(pc, 1, cr)), "cuda")
                arms[f"fw1d-cr{cr}-pf"] = (dict(grid=(pc, 1, cr), prefetch=True), "cuda")
                arms[f"fw1d-cr{cr}-st4"] = (dict(grid=(pc, 1, cr), stages=4), "cuda")
                arms[f"fw1d-cr{cr}-pf-st4"] = (dict(grid=(pc, 1, cr), prefetch=True, stages=4), "cuda")
            outs, row = {}, dict(cell=cell, dtype=dtype, width=width, batch=16, arms={})
            for name, (spec, kind) in arms.items():
                ent = {}
                try:
                    if kind is None:
                        fn = spec
                    else:
                        def fn(spec=spec):
                            with torch.no_grad():
                                return hybrid_fw2d.recurrence(wx, r, b, init, cell, sequence_context=ctx,
                                                              output_mode="full", snapshot_dtype=torch.float32,
                                                              **spec)[0]
                    out = fn().clone()
                    ent["ms"] = median_ms(fn, a.calls)
                    outs[name] = out
                    if kind is not None:  # phase timers (separate build; not the timed one)
                        diag = {}
                        with torch.no_grad():
                            hybrid_fw2d.recurrence(wx, r, b, init, cell, sequence_context=ctx, output_mode="full",
                                                   snapshot_dtype=torch.float32, prof=True, diagnostics=diag, **spec)
                        ent["phases_us_per_step"] = diag.get("phases_us_per_step")
                        ent["shared_bytes"] = diag.get("shared_bytes")
                except Exception:  # noqa: BLE001
                    ent["error"] = traceback.format_exc()[-1500:]
                row["arms"][name] = ent
                torch.cuda.empty_cache()
            ref = outs.get("tilelang")
            for name, out in outs.items():
                if ref is not None:
                    row["arms"][name]["max_abs_vs_tilelang"] = (out[0] - ref[0]).abs().max().item()
                for cr in (0, 16):
                    base = outs.get(f"fw1d-cr{cr}")
                    if name.startswith(f"fw1d-cr{cr}") and base is not None:
                        row["arms"][name]["bitwise_vs_fw1d_base"] = bool(torch.equal(out, base))
            rec["rows"].append(row)
            print(json.dumps({k: (v.get("ms"), v.get("phases_us_per_step"), v.get("max_abs_vs_tilelang"),
                                  v.get("bitwise_vs_fw1d_base"), (v.get("error") or "")[-200:])
                              for k, v in row["arms"].items()}, default=str), flush=True)
    rec["finished"] = time.time()
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
