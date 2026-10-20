"""FP64 qualification of the cluster forward's fast-math and wgmma variants (docs/flashrnn2/max-width.md section 82).

Usage: fwc_fp64_gate.py --out FILE [--widths 512,768] [--steps 64]
Per (cell, D), B = 16: the hidden history of the FP64 oracle (flashrnn2.grad_reference on the same rounded inputs)
against the TileLang forward, fwc bulk (exact math), fwc bulk + fast math, fwc st.async + fast math and, where
4 D / 16 = 128, the wgmma variants (register k-steps 16 and 32, with and without fast math): rel_fro and
max_abs_scaled (bw_qualify.compare), repeat-call bitwise stability, bitwise equality to the bulk variant.
Gate (pre-registered in section 82): every variant finite, rel_fro <= C[dtype] (bw_qualify.BUDGET: 1e-2 FP16,
5e-2 BF16), and rel_fro <= max(1.25 x bulk's rel_fro, C[dtype] / 10); repeat calls bitwise. The wgmma variants are
not expected to be bitwise to bulk (another accumulation order); the fast-math ones are not expected to be bitwise
either. Writes JSON with per-variant metrics, checks and an overall "pass".
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, grad_reference, hybrid_fwc
from flashrnn.flashrnn2.execution_policy import KernelConfig
from flashrnn.flashrnn2.state_contract import begin_sequence

TL = {512: dict(family="tilelang", slice_width=32, sync="cluster"),
      768: dict(family="tilelang", slice_width=16, sync="grid"),
      1024: dict(family="tilelang", slice_width=16, sync="grid")}
FACTOR = 1.25


def variants(width):
    v = {"bx": dict(bulkx=True), "bx-fm": dict(bulkx=True, fastmath=True), "sa-fm": dict(stasync=True, fastmath=True),
         "bx-fm-mc": dict(bulkx=True, fastmath=True, mcast=True)}
    if hybrid_fwc.wgmma_supported(width, 16):
        ks = width // 16
        for cr in (16, ks):
            v[f"wg-bx-cr{cr}"] = dict(bulkx=True, wgmma=True, cr=cr)
            v[f"wg-bx-fm-cr{cr}"] = dict(bulkx=True, wgmma=True, fastmath=True, cr=cr)
        v["wg-sa-fm-cr16"] = dict(stasync=True, wgmma=True, fastmath=True, cr=16)
        v[f"wg-bx-fm-mc-cr{ks}"] = dict(bulkx=True, wgmma=True, fastmath=True, mcast=True, cr=ks)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="512,768")
    ap.add_argument("--steps", type=int, default=64)
    a = ap.parse_args()
    rec = dict(started=time.time(), gpu=torch.cuda.get_device_name(), factor=FACTOR, budget=bw_qualify.BUDGET, cases={})
    overall = True
    for cell, dtype in (("lstm", "float16"), ("slstm", "bfloat16")):
        for width in map(int, a.widths.split(",")):
            case = dict(cell=cell, dtype=dtype, width=width, batch=16, steps=a.steps, heads=1, init="warm")
            leaves, _, _ = bw_qualify.inputs(case, 8200 + width)
            wx, r, b, init = leaves
            hist, _ = grad_reference.recurrence(*[x.detach().double().cpu() for x in leaves], cell)
            ref = hist[0]  # [B, T, H, D] FP64 hidden history
            ctx = begin_sequence(init, cell, wx.dtype)
            out = dict(variants={}, pass_=True)
            outs = {}
            with torch.no_grad():
                try:
                    tl = dispatch.recurrence(wx, r, b, init, cell, output_mode="hidden", config=KernelConfig(**TL[width]))[0]
                    out["variants"]["tilelang"] = bw_qualify.compare(tl, ref)
                except Exception:  # noqa: BLE001
                    out["variants"]["tilelang"] = dict(error=traceback.format_exc()[-800:])
                for name, kw in variants(width).items():
                    try:
                        o1 = hybrid_fwc.recurrence(wx, r, b, init, cell, sequence_context=ctx, output_mode="hidden", **kw)[0]
                        o2 = hybrid_fwc.recurrence(wx, r, b, init, cell, sequence_context=ctx, output_mode="hidden", **kw)[0]
                        m = bw_qualify.compare(o1, ref)
                        m["repeat_bitwise"] = bool(torch.equal(o1, o2))
                        outs[name] = o1
                        if "bx" in outs:
                            m["bitwise_vs_bx"] = bool(torch.equal(o1, outs["bx"]))
                        out["variants"][name] = m
                    except Exception:  # noqa: BLE001
                        out["variants"][name] = dict(error=traceback.format_exc()[-800:])
            c = bw_qualify.BUDGET[dtype]
            base = out["variants"].get("bx", {}).get("rel_fro")
            for name, m in out["variants"].items():
                if name == "tilelang":
                    continue
                if "error" in m or base is None:
                    ok = False
                else:
                    ok = (m["finite"] and m["rel_fro"] <= c and m["rel_fro"] <= max(FACTOR * base, c / 10)
                          and m["repeat_bitwise"])
                m["check"] = ok
                out["pass_"] = out["pass_"] and ok
            overall = overall and out["pass_"]
            rec["cases"][f"{cell}-d{width}"] = out
            for name, m in out["variants"].items():
                print(f"{cell} D{width} {name:14s} rel_fro={m.get('rel_fro', float('nan')):.3e} "
                      f"max_abs_scaled={m.get('max_abs_scaled', float('nan')):.3e} repeat={m.get('repeat_bitwise')} "
                      f"vs_bx={m.get('bitwise_vs_bx')} check={m.get('check')} err={(m.get('error') or '')[-120:]}",
                      flush=True)
    rec["pass"] = overall
    rec["finished"] = time.time()
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))
    print("GATE", "PASS" if overall else "FAIL")


if __name__ == "__main__":
    main()
