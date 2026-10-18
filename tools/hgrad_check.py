"""Bitwise check of the h-only history gradient (heads.md section 8.7): every training gradient with the default
(d_history = the h component only, on the fast Triton path) against `full_history_grad=True` (the zero-filled full
history gradient). Usage: hgrad_check.py --out FILE --cases JSON; case {"cell", "dtype", "width", "heads", "batch",
"steps"} (library defaults otherwise)."""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import training


def grads(leaves, cot_h, cell, extra):
    xs = [x.detach().requires_grad_(True) for x in leaves]
    y, _ = training.recurrence(*xs, cell, output_mode="hidden", backward_options=dict(extra))
    return torch.autograd.grad((y.float() * cot_h).sum(), xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cases", required=True)
    a = p.parse_args()
    rec = dict(started=time.time(), rows=[])
    for case in json.loads(a.cases):
        row = dict(case)
        try:
            leaves, cot_h, _ = bw_qualify.inputs(dict(case, init="warm"), 5151 + case["width"] + case["heads"])
            new = grads(leaves, cot_h, case["cell"], {})
            old = grads(leaves, cot_h, case["cell"], {"full_history_grad": True})
            row["bitwise"] = [bool(torch.equal(x, y)) for x, y in zip(new, old)]
            row["pass_"] = all(row["bitwise"])
        except Exception:  # noqa: BLE001
            row.update(pass_=False, error=traceback.format_exc()[-1200:])
        rec["rows"].append(row)
        print(json.dumps(row), flush=True)
    rec.update(finished=time.time(), status="PASS" if all(r["pass_"] for r in rec["rows"]) else "FAIL")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
