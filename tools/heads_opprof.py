"""Operator-level profile of one H > 1 training step (heads.md section 8.3, report only).

Usage: heads_opprof.py --out FILE --cases JSON
Each case {"cell", "dtype", "width", "heads", "batch", "steps"} with the library's defaults: two warm-up steps, then
one step under torch.profiler (CUDA activities, input shapes recorded). Writes the 25 most expensive operators by
self CUDA time with their input shapes and call counts, and the total CUDA time of the step, so that every copy and
cast outside the recurrent kernels can be attributed to a line of the training path.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import bw_qualify
from flashrnn.flashrnn2 import training


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cases", required=True)
    a = p.parse_args()
    rec = dict(started=time.time(), rows=[])
    for case in json.loads(a.cases):
        row = dict(case)
        try:
            leaves, cot_h, _ = bw_qualify.inputs(dict(case, init="warm"), 4242 + case["width"] + case["heads"])

            def step():
                xs = [x.detach().requires_grad_(True) for x in leaves]
                y, _ = training.recurrence(*xs, case["cell"], output_mode="hidden")
                return torch.autograd.grad((y.float() * cot_h).sum(), xs)
            for _ in range(2):
                step()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                step()
                torch.cuda.synchronize()
            table = prof.key_averages(group_by_input_shape=True)
            ops = []
            for ev in table:
                cuda_us = getattr(ev, "self_device_time_total", None)
                if cuda_us is None:
                    cuda_us = getattr(ev, "self_cuda_time_total", 0)
                if cuda_us > 0:
                    ops.append(dict(name=ev.key, shapes=str(ev.input_shapes)[:300], calls=ev.count,
                                    self_cuda_ms=round(cuda_us / 1000, 4)))
            ops.sort(key=lambda o: -o["self_cuda_ms"])
            row.update(ops=ops[:25], total_self_cuda_ms=round(sum(o["self_cuda_ms"] for o in ops), 3))
        except Exception:  # noqa: BLE001
            row["error"] = traceback.format_exc()[-1500:]
        rec["rows"].append(row)
        print(json.dumps({k: row.get(k) for k in ("width", "heads", "total_self_cuda_ms", "error")}), flush=True)
    rec.update(finished=time.time(), status="DONE")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
