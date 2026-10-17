"""Multi-GPU backward (hybrid_bw.cu, FRNN_MG) on one node, one process (docs/flashrnn2/scale-out-plan.md section 10).

Modes (one JSON file per call in --out)
  qualify  per case: dG and d_initial of the P-GPU kernel (uva and nccl_symm exchange buffers) bitwise against the
           single-GPU kernel with the same batch tile and register split
  race     per case: content-checked runs (run 0 undelayed, runs 1.. with seeded delays) and three broken variants:
           single buffer, no grid barrier, no cross-GPU wait (each keeps its last barrier); rule of max-width.md
           section 12 per variant
  paired   per case: --blocks blocks of --calls calls of every arm, order rotated per block; arms "single" (one GPU,
           the library's batch tile), "mg_uva", "mg_symm" (P GPUs); per call the kernel time (slowest GPU)
Inputs: seeded on the CPU (scale_bench.inputs), forward history from the library forward on the first GPU.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch

import scale_bench
from flashrnn.flashrnn2 import dispatch, hybrid_bw, multi_gpu
from flashrnn.flashrnn2.execution_policy import default_config


def bw_inputs(cell, batch, steps, width, seed, device):
    wx, r, b, init = scale_bench.inputs(cell, batch, steps, 1, width, seed, device)
    with torch.no_grad():
        history, _ = dispatch.recurrence(wx, r, b, init, cell, output_mode="full", snapshot_dtype=torch.float32,
                                         config=default_config(width))
    g = torch.Generator().manual_seed(seed + 1)
    d_history = (torch.randn(history.shape, generator=g) * 0.1).to(device)
    d_history[1:] = 0  # cotangent on h only, as the training path passes it
    d_final = (torch.randn(init.shape, generator=g) * 0.1).float().to(device)
    return (wx, r, b, init, history, d_history.contiguous(), d_final, cell)


def digest(x):
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()


def single(args, bm, krw, diag=None):
    return hybrid_bw.backward(*args, batch_rows=bm, reg_rows_per_warp=krw, diagnostics=diag)


def mode_qualify(a, devices):
    rec = dict(mode="qualify", cases=[], started=time.time())
    for case in json.loads(a.cases):
        row = dict(case)
        try:
            args = bw_inputs(case["cell"], case["batch"], case["steps"], case["width"], 900 + case["width"],
                             f"cuda:{devices[0]}")
            bm = case["batch_rows"]
            krw = hybrid_bw.reg_rows(case["width"], bm)
            ref = single(args, bm, krw)
            row.update(reg_rows_per_warp=krw, dg_sha256=digest(ref[0]), d_init_sha256=digest(ref[3]))
            for alloc in ("uva", "nccl_symm"):
                try:
                    diag = {}
                    out = hybrid_bw.backward_mg(*args, devices=devices[:case["peers"]], alloc=alloc, batch_rows=bm,
                                                reg_rows_per_warp=krw, diagnostics=diag)
                    row[alloc] = dict(dg_bitwise=bool(torch.equal(out[0], ref[0])),
                                      d_init_bitwise=bool(torch.equal(out[3], ref[3])),
                                      d_recurrent_bitwise=bool(torch.equal(out[1], ref[1])),
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
            args = bw_inputs(case["cell"], case["batch"], case["steps"], case["width"], 950 + case["width"],
                             f"cuda:{devices[0]}")
            kw = dict(devices=devices[:case["peers"]], alloc="uva", batch_rows=case["batch_rows"])
            ref = hybrid_bw.backward_mg(*args, **kw)
            runs = case.get("runs", 21)
            for variant in (None, "single", "no_sync", "mg_no_wait"):
                rows = []
                for i in range(runs):
                    v = dict(check=True, delay_seed=None if i == 0 else 1000 * i + 11, broken=variant)
                    out = hybrid_bw.backward_mg(*args, _verify=v, **kw)
                    same = torch.equal(out[0], ref[0]) and torch.equal(out[3], ref[3])
                    rows.append(dict(run=i, after_read=v["after_read"], after_mma=v["after_mma"],
                                     wait_timeouts=v["wait_timeouts"], outputs_equal=same))
                count = lambda x: x["after_read"] + x["after_mma"]  # noqa: E731
                flagged = sum(count(x) > 0 for x in rows)
                silent = sum((not x["outputs_equal"]) and count(x) == 0 for x in rows)
                entry = dict(runs=rows, flagged=flagged, silent=silent, changed=sum(not x["outputs_equal"] for x in rows),
                             timeouts=sum(x["wait_timeouts"] for x in rows))
                if variant is None:
                    entry["pass_"] = flagged == 0 and all(x["outputs_equal"] for x in rows) and entry["timeouts"] == 0
                else:
                    entry["detected"] = flagged >= 0.8 * runs and silent == 0
                row["variants"][variant or "correct"] = entry
            row["pass_"] = row["variants"]["correct"]["pass_"] and all(
                row["variants"][v]["detected"] for v in ("single", "no_sync", "mg_no_wait"))
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
            args = bw_inputs(case["cell"], case["batch"], case["steps"], case["width"], 990 + case["width"],
                             f"cuda:{devices[0]}")
            bm_single = hybrid_bw.batch_tile(case["batch"])
            mg = dict(devices=devices[:case["peers"]], batch_rows=case["batch_rows"])

            def run(arm):
                diag = {"time": True}
                if arm == "single":
                    out = single(args, bm_single, None, diag)
                    return out, diag["ms"]["kernel"]
                out = hybrid_bw.backward_mg(*args, alloc="uva" if arm == "mg_uva" else "nccl_symm", diagnostics=diag,
                                            **mg)
                return out, diag["kernel_ms"]
            arms = ["single", "mg_uva", "mg_symm"]
            first = {}
            for arm in arms:
                run(arm)
                first[arm] = run(arm)[0][0]
                row["outputs_stable"][arm] = True
            row["mg_bitwise_vs_single"] = bool(torch.equal(first["mg_uva"], first["single"]))
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
                                 for arm in ("single", "mg_uva", "mg_symm")})), flush=True)
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
    return {"qualify": mode_qualify, "race": mode_race, "paired": mode_paired}[a.mode](a, devices)


if __name__ == "__main__":
    raise SystemExit(main())
