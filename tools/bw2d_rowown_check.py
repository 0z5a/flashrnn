"""Bitwise check of the 2-D backward's row ownership (FRNN_ROWOWN = 1) against the first version's column-sub-block
ownership (0): same elements and arithmetic per CTA, so dG and d_initial must be equal bit for bit (max-width.md
section 22). Usage: bw2d_rowown_check.py --out FILE --cases JSON [--option rowown | asyncload | dg_dtype]
--option names the layout switch compared (default rowown; asyncload compares cp.async staging against synchronous
staging, max-width.md section 24; dg_dtype compares the input-dtype dG store against the FP32 store, both with wgrad
"bmm_lp": the stored dG must equal the FP32 dG rounded to the input dtype, and d_initial and dR must be bitwise
equal, max-width.md section 36).
Each case: {"cell", "dtype", "width", "batch", "steps"}; the batch tile is the library's for the batch."""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, hybrid_bw2d
from flashrnn.flashrnn2.execution_policy import default_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--option", default="rowown", choices=("rowown", "asyncload", "dg_dtype"))
    a = p.parse_args()
    rec = dict(started=time.time(), rows=[])
    for case in json.loads(a.cases):
        row = dict(case)
        try:
            c = dict(case, heads=1, init="warm")
            leaves, cot_h, cot_f = bw_qualify.inputs(c, 7700 + case["width"] + case["batch"])
            wx, r, b, init = leaves
            with torch.no_grad():
                history, _ = dispatch.recurrence(wx, r, b, init, case["cell"], output_mode="full",
                                                 snapshot_dtype=torch.float32, config=default_config(case["width"]))
            d_history = torch.zeros_like(history)
            d_history[0] += cot_h.float()
            d_final = torch.zeros(init.shape, device=wx.device, dtype=torch.float32) + cot_f.float()
            args = (wx, r, b, init, history, d_history, d_final, case["cell"])
            if a.option == "dg_dtype":
                new = hybrid_bw2d.backward(*args, dg_dtype="input", wgrad="bmm_lp")
                old = hybrid_bw2d.backward(*args, wgrad="bmm_lp")
                row.update(dg_bitwise=bool(torch.equal(new[0], old[0].to(wx.dtype))),
                           d_init_bitwise=bool(torch.equal(new[3], old[3])),
                           d_recurrent_bitwise=bool(torch.equal(new[1], old[1])),
                           d_bias_max_rel=((new[2] - old[2]).abs().max() / old[2].abs().max()).item())
            else:
                new = hybrid_bw2d.backward(*args, **{a.option: True})
                old = hybrid_bw2d.backward(*args, **{a.option: False})
                row.update(dg_bitwise=bool(torch.equal(new[0], old[0])),
                           d_init_bitwise=bool(torch.equal(new[3], old[3])))
            row["pass_"] = row["dg_bitwise"] and row["d_init_bitwise"] and row.get("d_recurrent_bitwise", True)
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
