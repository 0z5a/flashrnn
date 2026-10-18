"""Accuracy of the fused preactivation (flashrnn2/preact.py) against FP64 and against the library's TF32 path
(triton_rtiled_bw.preactivations, tf32=True as the training backward runs it), max-width.md section 35.
Usage: preact_check.py --out FILE --cases JSON; case {"dtype", "width", "batch", "steps"}.
Pass: the fused path's largest error relative to max |pre| is at most that of the TF32 path (plus 1e-7)."""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from flashrnn.flashrnn2 import preact
from flashrnn.flashrnn2.triton_rtiled_bw import preactivations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cases", required=True)
    a = p.parse_args()
    rec = dict(started=time.time(), rows=[])
    for case in json.loads(a.cases):
        row = dict(case)
        try:
            g = torch.Generator().manual_seed(case["width"] + case["batch"])
            dt = getattr(torch, case["dtype"])
            B, T, D = case["batch"], case["steps"], case["width"]
            wx = (torch.randn(B, T, 4, 1, D, generator=g) * 0.5).to(dt).cuda()
            r = (torch.randn(4, 1, D, D, generator=g) * (0.5 / D ** 0.5)).to(dt).cuda()
            b = (torch.randn(4, 1, D, generator=g) * 0.1).to(dt).cuda()
            init = (torch.randn(2, B, 1, 1, D, generator=g) * 0.3).cuda()
            hist = (torch.randn(2, B, T, 1, D, generator=g) * 0.5).cuda()
            h = preact.h_prev_of(wx, init, hist)
            fused = preact.preactivations_fused(wx, r, b, h)
            tf32, h_ref = preactivations(wx, r, b, init, hist, True)
            ref = (h.double().reshape(B * T, D) @ r.double().reshape(4 * D, D).t() + b.double().reshape(1, 4 * D)
                   ).view(B, T, 4, 1, D) + wx.double()
            scale = ref.abs().max()
            row.update(fused_rel=((fused.double() - ref).abs().max() / scale).item(),
                       tf32_rel=((tf32.double() - ref).abs().max() / scale).item(),
                       h_prev_equal=bool(torch.equal(h, h_ref)))
            row["pass_"] = row["h_prev_equal"] and row["fused_rel"] <= row["tf32_rel"] + 1e-7
        except Exception:  # noqa: BLE001
            row.update(pass_=False, error=traceback.format_exc()[-1200:])
        rec["rows"].append(row)
        print(json.dumps(row), flush=True)
    rec.update(finished=time.time(), status="PASS" if all(x["pass_"] for x in rec["rows"]) else "FAIL")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
