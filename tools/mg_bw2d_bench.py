"""Multi-GPU 2-D split backward (hybrid_bw2d.cu, FRNN_MG) on one node, one process (scale-out-plan.md section 17).

Modes (one JSON file per call in --out)
  qualify  per case: dG and d_initial of the P-GPU 2-D kernel (uva and nccl_symm buffers) bitwise against the
           single-GPU 2-D kernel with the same batch tile
  race     per case: content-checked runs (run 0 undelayed, runs 1.. with seeded delays) and three broken variants:
           no B1, no B2 (grid barrier and cross-GPU wait), no cross-GPU wait (each keeps the last step's barriers);
           rule of max-width.md section 12 per variant
  paired   per case: --blocks blocks of --calls calls of every arm, order rotated per block; arms "single2d" (one
           GPU, 2-D, the library's batch tile), "mg2d" (P GPUs, 2-D, uva), "mg1d" (P GPUs, 1-D, uva), "mg2d_symm";
           per call the backward kernel time (slowest GPU)
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import scale_bench
from mg_bw_bench import bw_inputs
from flashrnn.flashrnn2 import hybrid_bw, hybrid_bw2d, multi_gpu


def mode_qualify(a, devices):
    rec = dict(mode="qualify", cases=[], started=time.time())
    for case in json.loads(a.cases):
        row = dict(case)
        try:
            args = bw_inputs(case["cell"], case["batch"], case["steps"], case["width"], 1900 + case["width"],
                             f"cuda:{devices[0]}")
            bm = case["batch_rows"]
            ref = hybrid_bw2d.backward(*args, batch_rows=bm, conc=False)  # the grid the multi-GPU kernel splits
            for alloc in ("uva", "nccl_symm"):
                try:
                    diag = {}
                    out = hybrid_bw2d.backward_mg(*args, devices=devices[:case["peers"]], alloc=alloc, batch_rows=bm,
                                                  diagnostics=diag)
                    row[alloc] = dict(dg_bitwise=bool(torch.equal(out[0], ref[0])),
                                      d_init_bitwise=bool(torch.equal(out[3], ref[3])),
                                      max_abs_dg=(out[0] - ref[0]).abs().max().item(), kernel_ms=diag["kernel_ms"],
                                      resident=diag["resident"], slots=diag["slots"])
                except Exception:  # noqa: BLE001
                    row[alloc] = dict(error=traceback.format_exc()[-1500:], dg_bitwise=False, d_init_bitwise=False)
            row["pass_"] = all(row[al]["dg_bitwise"] and row[al]["d_init_bitwise"] for al in ("uva", "nccl_symm"))
        except Exception:  # noqa: BLE001
            row.update(error=traceback.format_exc()[-1500:], pass_=False)
        rec["cases"].append(row)
        print(json.dumps({k: row.get(k) for k in ("width", "peers", "batch", "batch_rows", "cell", "pass_", "error")}),
              flush=True)
        scale_bench.write(Path(a.out) / "qualify.json", rec)
    rec.update(status="PASS" if all(c["pass_"] for c in rec["cases"]) else "FAIL", finished=time.time())
    scale_bench.write(Path(a.out) / "qualify.json", rec)
    return 0


def mode_race(a, devices):
    rec = dict(mode="race", cases=[], started=time.time())
    for case in json.loads(a.cases):
        row = dict(case, variants={})
        try:
            args = bw_inputs(case["cell"], case["batch"], case["steps"], case["width"], 1950 + case["width"],
                             f"cuda:{devices[0]}")
            kw = dict(devices=devices[:case["peers"]], alloc="uva", batch_rows=case["batch_rows"])
            ref = hybrid_bw2d.backward_mg(*args, **kw)
            runs = case.get("runs", 21)
            for variant in (None, "no_b1", "no_b2", "mg_no_wait"):
                rows = []
                for i in range(runs):
                    v = dict(check=True, delay_seed=None if i == 0 else 1000 * i + 17, broken=variant)
                    out = hybrid_bw2d.backward_mg(*args, _verify=v, **kw)
                    same = torch.equal(out[0], ref[0]) and torch.equal(out[3], ref[3])
                    rows.append(dict(run=i, after_read=v["after_read"], stale_partials=v["stale_partials"],
                                     wait_timeouts=v["wait_timeouts"], outputs_equal=same))
                count = lambda x: x["after_read"] + x["stale_partials"]  # noqa: E731
                flagged = sum(count(x) > 0 for x in rows)
                silent = sum((not x["outputs_equal"]) and count(x) == 0 for x in rows)
                entry = dict(runs=rows, flagged=flagged, silent=silent,
                             changed=sum(not x["outputs_equal"] for x in rows),
                             timeouts=sum(x["wait_timeouts"] for x in rows))
                if variant is None:
                    entry["pass_"] = flagged == 0 and all(x["outputs_equal"] for x in rows) and entry["timeouts"] == 0
                else:
                    entry["detected"] = flagged >= 0.8 * runs and silent == 0
                row["variants"][variant or "correct"] = entry
            row["pass_"] = row["variants"]["correct"]["pass_"] and all(
                row["variants"][v]["detected"] for v in ("no_b1", "no_b2", "mg_no_wait"))
        except Exception:  # noqa: BLE001
            row.update(error=traceback.format_exc()[-1500:], pass_=False)
        rec["cases"].append(row)
        print(json.dumps(dict(case=case, pass_=row.get("pass_"), summary={
            k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in row["variants"].items()},
            error=(row.get("error") or "")[-300:])), flush=True)
        scale_bench.write(Path(a.out) / "race.json", rec)
    rec.update(status="DONE", finished=time.time())
    scale_bench.write(Path(a.out) / "race.json", rec)
    return 0


def mode_paired(a, devices):
    rec = dict(mode="paired", start=a.start, blocks_per_case=a.blocks, calls=a.calls, cases=[], started=time.time())
    for case in json.loads(a.cases):
        row = dict(case, blocks=[], outputs_stable={})
        try:
            args = bw_inputs(case["cell"], case["batch"], case["steps"], case["width"], 1990 + case["width"],
                             f"cuda:{devices[0]}")
            mg = dict(devices=devices[:case["peers"]], batch_rows=case["batch_rows"])

            def run(arm):
                diag = {"time": True}
                if arm == "single2d":
                    out = hybrid_bw2d.backward(*args, diagnostics=diag, conc=False)
                    return out, diag["ms"]["kernel"]
                if arm == "mg1d":
                    out = hybrid_bw.backward_mg(*args, alloc="uva", diagnostics=diag, **mg)
                else:
                    out = hybrid_bw2d.backward_mg(*args, alloc="nccl_symm" if arm == "mg2d_symm" else "uva",
                                                  diagnostics=diag, **mg)
                return out, diag["kernel_ms"]
            arms = ["single2d", "mg2d", "mg1d", "mg2d_symm"]
            first = {}
            for arm in arms:
                run(arm)
                first[arm] = run(arm)[0][0]
                row["outputs_stable"][arm] = True
            row["mg2d_bitwise_vs_mg2d_symm"] = bool(torch.equal(first["mg2d"], first["mg2d_symm"]))
            for blk in range(a.blocks):
                k = (blk + a.start) % len(arms)
                order = arms[k:] + arms[:k]
                entry = dict(order=order)
                for arm in order:
                    ts = []
                    for _ in range(a.calls):
                        out, ms = run(arm)
                        ts.append(ms)
                    if not torch.equal(out[0], first[arm]):
                        row["outputs_stable"][arm] = False
                    entry[f"{arm}_ms"] = sum(ts) / len(ts)
                row["blocks"].append(entry)
        except Exception:  # noqa: BLE001
            row["error"] = traceback.format_exc()[-1500:]
        rec["cases"].append(row)
        b = row["blocks"]
        print(json.dumps(dict(case=case, blocks=len(b), error=(row.get("error") or "")[-200:],
                              **{arm: round(sorted(x[f"{arm}_ms"] for x in b)[len(b) // 2], 4) if b else None
                                 for arm in ("single2d", "mg2d", "mg1d", "mg2d_symm")})), flush=True)
        scale_bench.write(Path(a.out) / "paired.json", rec)
    rec.update(status="DONE", finished=time.time())
    scale_bench.write(Path(a.out) / "paired.json", rec)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("qualify", "race", "paired"))
    p.add_argument("--out", required=True)
    p.add_argument("--devices", default="0,1,2,3")
    p.add_argument("--cases", required=True, help="JSON list of {cell, width, peers, batch, steps, batch_rows}")
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--calls", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    devices = [int(x) for x in a.devices.split(",")]
    multi_gpu.enable_peer_access(devices)
    if a.mode != "race":
        from flashrnn.flashrnn2 import symm_mem
        symm_mem.session(devices)
    torch.cuda.set_device(devices[0])
    return {"qualify": mode_qualify, "race": mode_race, "paired": mode_paired}[a.mode](a, devices)


if __name__ == "__main__":
    raise SystemExit(main())
