"""Development bench for the B = 16 forward (docs/flashrnn2/max-width.md sections 76 and 78), report only.

Usage: fw1d_bench.py --out FILE [--widths 512,1024] [--cells lstm:float16,slstm:bfloat16] [--calls 30]
Per (cell, D), B = 16, T = 256, the forward with hidden output in the input dtype (what cuDNN returns) of:
  tilelang     the library's TileLang forward (the measured-runtime selection)
  cudnn        torch.nn.LSTM (cuDNN, LSTM only): total forward time, and the recurrent kernels alone from
               torch.profiler (names containing "lstm"), since cuDNN also runs the input projection
  fw2d         the CUDA 2-D forward with its default grid (D in hybrid_fw2d.CONFIGS only)
  fwc-c{C}-cr{CR}-k{KSPLIT}[-pull]
               the cluster forward (hybrid_fwc, section 80): C CTAs in one cluster, CR register k-steps, KSPLIT
               interleaved accumulators, push (default) or pull (symmetric buffers) exchange; D <= 768
  fw1d-...     the CUDA forward with a 1-D grid (PC x 1 x CR) and the section 76 / 78 options:
               pf = split-B2 Wx prefetch, st4 = 4-stage cp.async staging, bulk = one bulk copy per staged row,
               flag = flat per-CTA flag barrier as B2, sp = gate partials through shared memory
Records the median ms per call (CUDA events), the per-step phase times of the CUDA variants (phase-timer build), the
largest difference of every arm's output to tilelang's, and whether the fw1d variants are bitwise equal to the fw1d
base of the same grid.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import bw_qualify
from flashrnn.flashrnn2 import dispatch, hybrid_fw2d, hybrid_fwc
from flashrnn.flashrnn2.execution_policy import KernelConfig
from flashrnn.flashrnn2.state_contract import begin_sequence

TL = {512: dict(family="tilelang", slice_width=32, sync="cluster"),
      768: dict(family="tilelang", slice_width=16, sync="grid"),
      1024: dict(family="tilelang", slice_width=16, sync="grid")}
# (PC, CR) grids per D: PC = D / 16 (64-column CTAs) and D / 32 (128-column CTAs; register k-steps where R does not
# fit shared memory next to the staged block).
GRIDS = {512: [(32, 0), (16, 0)], 768: [(48, 0), (24, 8)], 1024: [(64, 0), (32, 24)]}
VARIANTS = {"pf-st3": dict(prefetch=True, stages=3),
            "pf-st4": dict(prefetch=True, stages=4),
            "pf-st6": dict(prefetch=True, stages=6),
            "pf-st8": dict(prefetch=True, stages=8),
            "bulk": dict(bulk=True),
            "bulk-pf": dict(bulk=True, prefetch=True),
            "bulk-flag": dict(bulk=True, flagbar=True),
            "bulk-flag-sp": dict(bulk=True, flagbar=True, smempart=True),
            "bulk-flag-pf-sp": dict(bulk=True, flagbar=True, prefetch=True, smempart=True),
            "flag-pf-sp-st4": dict(flagbar=True, prefetch=True, smempart=True, stages=4),
            "flag-pf-sp-st8": dict(flagbar=True, prefetch=True, smempart=True, stages=8)}


def median_ms(fn, calls):
    fn(); fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(calls):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return sorted(ts)[len(ts) // 2]


def cudnn_arm(wx, r, b, width, dtype):
    """nn.LSTM with R and the bias from the case; the input is a random x (cuDNN runs its own input projection)."""
    lstm = torch.nn.LSTM(width, width, batch_first=True).to("cuda", dtype)
    with torch.no_grad():
        lstm.weight_hh_l0.copy_(r.reshape(4 * width, width))
        lstm.bias_ih_l0.copy_(b.reshape(4 * width))
        lstm.bias_hh_l0.zero_()
    lstm.flatten_parameters()
    x = torch.randn(wx.shape[0], wx.shape[1], width, device="cuda").to(dtype) * 0.5

    def fn():
        with torch.no_grad():
            return lstm(x)[0]
    return fn


def kernel_us(fn, pattern, calls=5):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    tot = 0.0
    for e in prof.key_averages():
        if pattern in e.key.lower():
            tot += (getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0))
    return tot / calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="512,768,1024")
    ap.add_argument("--cells", default="lstm:float16,slstm:bfloat16")
    ap.add_argument("--calls", type=int, default=30)
    a = ap.parse_args()
    rec = dict(started=time.time(), gpu=torch.cuda.get_device_name(), cudnn=torch.backends.cudnn.version(), rows=[])
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
                    return dispatch.recurrence(wx, r, b, init, cell, output_mode="hidden", config=KernelConfig(**TL[width]))[0]
            arms["tilelang"] = (tl, None)
            if cell == "lstm":
                arms["cudnn"] = (cudnn_arm(wx, r, b, width, wx.dtype), "cudnn")
            if hybrid_fw2d.supported(width):
                arms["fw2d"] = (dict(), "cuda")
            for cl in (16, 8):
                if not hybrid_fwc.supported(width, cl):
                    continue
                cr0 = hybrid_fwc.min_register_ksteps(width, cl)
                for ks in (1, 2, 4):
                    arms[f"fwc-c{cl}-cr{cr0}-k{ks}"] = (dict(cluster=cl, cr=cr0, ksplit=ks), "fwc")
                arms[f"fwc-c{cl}-cr{cr0}-k2-pull"] = (dict(cluster=cl, cr=cr0, ksplit=2, pull=True), "fwc")
                if cr0 + 8 <= width // 16:
                    arms[f"fwc-c{cl}-cr{cr0 + 8}-k2"] = (dict(cluster=cl, cr=cr0 + 8, ksplit=2), "fwc")
            for pc, cr in GRIDS[width]:
                arms[f"fw1d-p{pc}-cr{cr}"] = (dict(grid=(pc, 1, cr)), "cuda")
                for name, opts in VARIANTS.items():
                    arms[f"fw1d-p{pc}-cr{cr}-{name}"] = (dict(grid=(pc, 1, cr), **opts), "cuda")
            outs, row = {}, dict(cell=cell, dtype=dtype, width=width, batch=16, arms={})
            for name, (spec, kind) in arms.items():
                ent = {}
                try:
                    if kind in (None, "cudnn"):
                        fn = spec
                    elif kind == "fwc":
                        def fn(spec=spec):
                            with torch.no_grad():
                                return hybrid_fwc.recurrence(wx, r, b, init, cell, sequence_context=ctx,
                                                             output_mode="hidden", **spec)[0]
                    else:
                        def fn(spec=spec):
                            with torch.no_grad():
                                return hybrid_fw2d.recurrence(wx, r, b, init, cell, sequence_context=ctx,
                                                              output_mode="hidden", **spec)[0]
                    out = fn().clone()
                    ent["ms"] = median_ms(fn, a.calls)
                    ent["us_per_step"] = ent["ms"] * 1000 / 256
                    if kind == "cudnn":
                        ent["recurrent_kernel_us_per_step"] = kernel_us(fn, "lstm") / 256
                    else:
                        outs[name] = out
                    if kind in ("cuda", "fwc"):  # phase timers (separate build; not the timed one)
                        diag = {}
                        mod = hybrid_fwc if kind == "fwc" else hybrid_fw2d
                        with torch.no_grad():
                            mod.recurrence(wx, r, b, init, cell, sequence_context=ctx, output_mode="hidden",
                                           prof=True, diagnostics=diag, **spec)
                        ent["phases_us_per_step"] = diag.get("phases_us_per_step")
                        ent["shared_bytes"] = diag.get("shared_bytes")
                except Exception:  # noqa: BLE001
                    ent["error"] = traceback.format_exc()[-1500:]
                row["arms"][name] = ent
                torch.cuda.empty_cache()
            ref = outs.get("tilelang")
            for name, out in outs.items():
                if ref is not None:
                    row["arms"][name]["max_abs_vs_tilelang"] = (out.float() - ref.float()).abs().max().item()
                for pc, cr in GRIDS[width]:
                    base = outs.get(f"fw1d-p{pc}-cr{cr}")
                    if name.startswith(f"fw1d-p{pc}-cr{cr}") and base is not None:
                        row["arms"][name]["bitwise_vs_fw1d_base"] = bool(torch.equal(out, base))
            rec["rows"].append(row)
            for k, v in row["arms"].items():
                ph = v.get("phases_us_per_step") or {}
                print(f"{cell} D{width} {k:28s} {v.get('us_per_step', 0):6.2f} us/step  B2 {ph.get('barrier_b2', ph.get('barrier', 0)):.2f} "
                      f"gemm {ph.get('gemm', 0):.2f} pw {ph.get('pointwise', 0):.2f}  "
                      f"ker {v.get('recurrent_kernel_us_per_step', 0):.2f}  bw={v.get('bitwise_vs_fw1d_base')} "
                      f"err={(v.get('error') or '')[-160:]}", flush=True)
    rec["finished"] = time.time()
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
