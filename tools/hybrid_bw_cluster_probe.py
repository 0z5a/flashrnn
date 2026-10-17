"""Feasibility of the distributed-shared-memory (cluster) CUDA backward on this GPU (max-width.md section 18, C0).

Usage: hybrid_bw_cluster_probe.py --task JSON --out FILE
task: {"widths": [...], "batch_rows": [...], "cell": "lstm", "dtype": "float16"}
Per (D, BM): the co-resident clusters of 8 CTAs the GPU accepts (cudaOccupancyMaxActiveClusters), the clusters one
group needs (D / 16 / 8), and a one-call backward (B = BM, T = 4) with cluster = 8 compared bitwise against the plain
kernel. Every entry records its own error; the probe itself only fails on a harness error.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, hybrid_bw
from flashrnn.flashrnn2.execution_policy import default_config


def probe(cell, dtype, width, bm):
    rec = dict(width=width, batch_rows=bm, clusters_per_group=width // hybrid_bw.S // hybrid_bw.WARPS)
    try:
        krw = hybrid_bw.reg_rows(width, bm)
        lib = hybrid_bw._lib(width, krw, cell == "slstm", dtype == "bfloat16", 0, bm, hybrid_bw.WARPS)
        rec.update(reg_rows_per_warp=krw, max_active_clusters=lib.frnn_hybrid_bw_max_clusters())
        case = dict(cell=cell, dtype=dtype, batch=bm, steps=4, width=width, heads=1, init="warm")
        leaves, cot_h, cot_f = bw_qualify.inputs(case, 5100 + width)
        wx, r, b, init = leaves
        with torch.no_grad():
            history, _ = dispatch.recurrence(wx, r, b, init, cell, output_mode="full", snapshot_dtype=torch.float32,
                                             config=default_config(width))
        d_history = torch.zeros_like(history)
        d_history[0] += cot_h.float()
        d_final = torch.zeros(init.shape, device=wx.device, dtype=torch.float32) + cot_f.float()
        args = (wx, r, b, init, history, d_history, d_final, cell)
        plain = hybrid_bw.backward(*args, batch_rows=bm)
        diag = {}
        clus = hybrid_bw.backward(*args, batch_rows=bm, cluster=hybrid_bw.WARPS, diagnostics=diag)
        torch.cuda.synchronize()
        rec.update(launched=True, resident=diag.get("resident"), ctas=diag.get("ctas"),
                   bitwise_vs_plain=torch.equal(plain[0], clus[0]) and torch.equal(plain[3], clus[3]))
    except Exception:
        rec.update(launched=False, error=traceback.format_exc()[-1200:])
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    task = json.loads(a.task)
    rec = dict(task=task, started=time.time(), gpu=torch.cuda.get_device_name(), entries=[])
    try:
        for width in task["widths"]:
            for bm in task["batch_rows"]:
                rec["entries"].append(probe(task.get("cell", "lstm"), task.get("dtype", "float16"), width, bm))
                torch.cuda.empty_cache()
        rec["status"] = "DONE"
    except Exception:
        rec.update(status="ERROR", error=traceback.format_exc())
    rec["finished"] = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(dict(status=rec["status"], entries=[{k: e.get(k) for k in ("width", "batch_rows", "launched",
                                                                               "max_active_clusters", "bitwise_vs_plain")}
                                                         for e in rec["entries"]])))
    return 0 if rec["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
