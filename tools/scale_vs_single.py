"""Multi-GPU forward against the best single-GPU forward (scale-out-plan.md section 23).

Usage: scale_vs_single.py --out FILE --cases JSON [--start N] [--blocks 20] [--calls 3]
Each case {"cell", "width", "batch", "steps", "configs": ["P:D:S:schedule:smem[:stride]", ...], "k_chunk"}.
Arms: "single" (dispatch.recurrence with the library's configuration, hidden output, one GPU) and every multi-GPU
configuration (multi_gpu.recurrence, uva buffers). Per call the kernel time: CUDA events around the single-GPU call;
the slowest GPU's events for the multi-GPU call. Blocks of --calls calls per arm, order rotated per block; outputs are
compared bitwise with each arm's first call (multi-GPU against single-GPU only to FP64 tolerance: different kernels).
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import scale_bench
from flashrnn.flashrnn2 import dispatch, multi_gpu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--devices", default="0,1,2,3")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--calls", type=int, default=3)
    a = p.parse_args()
    devices = [int(x) for x in a.devices.split(",")]
    multi_gpu.enable_peer_access(devices)
    torch.cuda.set_device(devices[0])
    rec = dict(start=a.start, rows=[], started=time.time())
    out_path = Path(a.out) / "vs1.json" if Path(a.out).is_dir() else Path(a.out)
    for case in json.loads(a.cases):
        row = dict(case, blocks=[], outputs_stable={})
        try:
            leaves = scale_bench.inputs(case["cell"], case["batch"], case["steps"], 1, case["width"],
                                        2323 + case["width"], f"cuda:{devices[0]}")  # FP16 (the multi-GPU kernel's)

            def single():
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                with torch.no_grad():
                    out, _ = dispatch.recurrence(*leaves, case["cell"], output_mode="hidden")
                e1.record()
                torch.cuda.synchronize(devices[0])
                return out, e0.elapsed_time(e1)

            def multi(cfg):
                c = scale_bench.parse(cfg)
                out, _, ms = multi_gpu.recurrence(*leaves, case["cell"], devices=devices[:c["peers"]],
                                                  slice_width=c["slice_width"], schedule=c["schedule"],
                                                  stride=c["stride"], rmode=c["rmode"], k_chunk=case["k_chunk"])
                return out, max(ms.values())
            arms = {"single": single, **{cfg: (lambda cfg=cfg: multi(cfg)) for cfg in case["configs"]}}
            names = list(arms)
            first = {}
            for n in names:
                arms[n]()
                first[n] = arms[n]()[0].float()
                row["outputs_stable"][n] = True
            ref = first["single"]
            row["max_rel_vs_single"] = {n: ((first[n] - ref).abs().max() / ref.abs().max()).item() for n in names}
            for blk in range(a.blocks):
                k = (blk + a.start) % len(names)
                order = names[k:] + names[:k]
                entry = dict(order=order)
                for n in order:
                    ts = []
                    for _ in range(a.calls):
                        out, ms = arms[n]()
                        ts.append(ms)
                    if not torch.equal(out.float(), first[n]):
                        row["outputs_stable"][n] = False
                    entry[n] = sum(ts) / len(ts)
                row["blocks"].append(entry)
        except Exception:  # noqa: BLE001
            row["error"] = traceback.format_exc()[-1500:]
        rec["rows"].append(row)
        b = row["blocks"]
        print(json.dumps(dict(width=case["width"], batch=case["batch"], cell=case["cell"],
                              medians={n: round(sorted(x[n] for x in b)[len(b) // 2], 4) for n in (b[0]["order"] if b else [])},
                              error=(row.get("error") or "")[-200:])), flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rec, indent=1))
    rec.update(finished=time.time(), status="DONE")
    out_path.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
