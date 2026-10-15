"""TileLang v2 fault localization: run one variant on a small case and compare with the FP64 oracle."""
import argparse
import json
import traceback
from pathlib import Path

import torch

from flashrnn.flashrnn2 import reference, tilelang_rtiled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("m3", "sync", "bulk"), required=True)
    p.add_argument("--cell", default="lstm")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--slice", type=int, default=32)
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--steps", type=int, default=17)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rec = dict(vars(a))
    try:
        g = torch.Generator().manual_seed(5)
        states = 4 if a.cell == "slstm" else 2
        vals = [torch.randn(a.batch, a.steps, 4, a.heads, a.width, generator=g) * 0.1,
                torch.randn(4, a.heads, a.width, a.width, generator=g) * (0.1 / a.width ** 0.5),
                torch.randn(4, a.heads, a.width, generator=g) * 0.03,
                torch.randn(states, a.batch, 1, a.heads, a.width, generator=g) * 0.05]
        if a.cell == "slstm":
            vals[3][2] = vals[3][2].abs() + 1
        vals = [v.half().cuda() for v in vals]
        if a.variant == "m3":
            y, fin = tilelang_rtiled.recurrence(*vals, a.cell, slice_width=a.slice)
        else:
            y, fin = tilelang_rtiled.recurrence_v2(*vals, a.cell, slice_width=a.slice, exchange=a.variant)
            key = next(k for k in tilelang_rtiled._KERNELS_V2)
            Path(a.out).with_suffix(".cu").write_text(tilelang_rtiled._KERNELS_V2[key].get_kernel_source())
        torch.cuda.synchronize()
        ref, final = reference.recurrence(*(v.double() for v in vals), a.cell, torch.float16)
        rec.update(status="RAN", hidden_max_abs=float((y.double() - ref[0]).abs().max()),
                   carry_max_abs=float((fin.double() - final).abs().max()))
    except Exception:  # noqa: BLE001 - the failure is the result
        rec.update(status="ERROR", error=traceback.format_exc()[-2500:])
    Path(a.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec.get(k) for k in ("variant", "status", "hidden_max_abs", "carry_max_abs")}), flush=True)


if __name__ == "__main__":
    main()
