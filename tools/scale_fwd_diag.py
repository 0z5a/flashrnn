"""Diagnosis of the multi-GPU forward failures of scale-out-plan.md section 12 (report only; section 13).

Usage: scale_fwd_diag.py --out DIR --configs C1,C2 --k-chunk N [--cells lstm,slstm] [--repeats 5] [--nccl 0|1]
Per configuration and cell, one process, uva buffers, B = 16, T = 48, the qualification's inputs:
  plain     --repeats multi-GPU calls in a row
  after_ref --repeats times: the single-GPU R-tiled reference on the first GPU, then a multi-GPU call
  after_p1  --repeats times: the P = 1 run of the multi-GPU kernel, then a multi-GPU call
Every call records: error (e.g. exchange timeout) and its wall time, per-GPU kernel time, FP64 error, output digest.
--nccl 1 opens the NCCL symmetric session first (as the qualification process did), else no NCCL communicator exists.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch

import scale_bench
from flashrnn.flashrnn2 import multi_gpu, reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--devices", default="0,1,2,3")
    p.add_argument("--configs", required=True)
    p.add_argument("--k-chunk", type=int, default=512)
    p.add_argument("--cells", default="lstm,slstm")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--nccl", type=int, default=0)
    p.add_argument("--warps", type=int, default=4)
    a = p.parse_args()
    devices = [int(x) for x in a.devices.split(",")]
    multi_gpu.enable_peer_access(devices)
    rec = dict(started=time.time(), nccl=a.nccl, rows=[])
    if a.nccl:
        from flashrnn.flashrnn2 import symm_mem
        symm_mem.session(devices)
    for cell in a.cells.split(","):
        for config in a.configs.split(","):
            c = scale_bench.parse(config)
            leaves = scale_bench.inputs(cell, 16, 48, 1, c["width"], 101 + c["width"], f"cuda:{devices[0]}")
            wx64, r64, b64, i64 = (x.double().cpu() for x in leaves)
            ref64 = reference.recurrence(wx64, r64, b64, i64, cell)[0][0].float()

            def multi(peers=None):
                t0 = time.time()
                row = {}
                try:
                    out, _, ms = multi_gpu.recurrence(
                        *leaves, cell, devices=devices[:peers or c["peers"]], slice_width=c["slice_width"],
                        schedule=c["schedule"] if not peers else "direct", stride=c["stride"] if not peers else 1,
                        rmode=c["rmode"], num_warps=a.warps, k_chunk=a.k_chunk)
                    o = out.cpu()
                    row.update(ms={str(k): round(v, 4) for k, v in ms.items()},
                               fp64=((o - ref64).abs().max() / ref64.abs().max()).item(),
                               digest=hashlib.sha256(o.numpy().tobytes()).hexdigest()[:16])
                except Exception:  # noqa: BLE001
                    row["error"] = traceback.format_exc()[-300:]
                row["wall_s"] = round(time.time() - t0, 3)
                return row
            res = dict(name=config, cell=cell, plain=[], after_ref=[], after_p1=[])
            for _ in range(a.repeats):
                res["plain"].append(multi())
            single_ok = 4 * c["width"] * c["slice_width"] * 2 <= 160 * 1024 and c["width"] // c["slice_width"] <= 132
            for _ in range(a.repeats):
                if single_ok:
                    scale_bench.single_gpu(leaves, cell, c["slice_width"], a.warps, devices[0])
                res["after_ref"].append(dict(multi(), reference_ran=single_ok))
            for _ in range(a.repeats):
                try:
                    p1 = multi(peers=1)
                except ValueError:
                    p1 = dict(error="P = 1 unavailable")
                res["after_p1"].append(dict(multi(), p1=p1))
            rec["rows"].append(res)
            summary = {k: [("E" if x.get("error") else (x["digest"][:6], round(x["fp64"], 5))) for x in res[k]]
                       for k in ("plain", "after_ref", "after_p1")}
            print(json.dumps(dict(name=config, cell=cell, **summary)), flush=True)
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / "diag.json").write_text(json.dumps(rec, indent=1))
    rec.update(finished=time.time(), status="DONE")
    (Path(a.out) / "diag.json").write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
