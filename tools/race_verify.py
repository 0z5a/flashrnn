"""GPU race verification of the cross-CTA exchanges (one task = one path/configuration/cell).

Paths: triton_fwd (L2 counter barrier), triton_bw (backward, same barrier), tl_cluster (DSMEM bulk copy +
mbarrier), tl_grid (global buffer + cooperative grid barrier). Every kernel runs with the compile-time
instrumentation (flashrnn2 ``_verify``): after each wait the consumer compares the whole received operand with
what the senders stored for that step (counter "after_barrier"), and again after the MMAs ("after_mma"/"after_gemm").

Runs per task
  reference          uninstrumented kernel
  correct            instrumented, no delay, plus SEEDS seeded delay runs (pseudo-random 0-4 us sleeps at the
                     producer, skewed per warp on TileLang, and after each consumer wait)
  broken variants    single buffer / skipped wait or barrier / arrive before store / no block barrier before the
                     send (per path), each without delay and with the same SEEDS
  positive control   tl_cluster with two warpgroups (the variant excluded after it gave 0/12 bitwise twins)

Pre-registered rule (round 2, written before the batch): a correct kernel passes when every run counts zero
violations and its outputs are bitwise equal to the reference; a broken variant is detected when violations are
counted in at least 80% of its completed runs (at least one), and every completed run whose output differs from the
reference must also count violations. A broken variant runs in its own process; a run that corrupts the CUDA
context ends that process, and it and the variant's remaining runs are counted as crashed, not as completed.
"no_sync" is reported only: TileLang inserts the same block barrier itself (generated source), so it is not broken.
Delays are 0-65 us; the cluster consumer delay sits after the wait, inside the window a single buffer exposes.
Writes race.json and DONE.json.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from flashrnn.flashrnn2.state_contract import begin_sequence

SEEDS = 20
BROKEN = {"triton_fwd": ("single", "no_barrier", "arrive_first"), "triton_bw": ("single", "no_barrier", "arrive_first"),
          "tl_cluster": ("single", "no_wait", "no_sync"), "tl_grid": ("single", "no_barrier", "arrive_first")}


def same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def inputs(cell, batch, steps, width, seed=5):
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    wx = torch.randn(batch, steps, 4, 1, width, generator=g) * 0.5
    r = torch.randn(4, 1, width, width, generator=g) * (0.5 / width ** 0.5)
    bias = torch.randn(4, 1, width, generator=g) * 0.1
    init = torch.randn(states, batch, 1, 1, width, generator=g) * 0.2
    if cell == "slstm":
        init[2] = init[2].abs() + 1
    return [x.to(torch.float16).cuda() for x in (wx, r, bias, init)]


def runner(task):
    path, cell, c = task["path"], task["cell"], task["config"]
    values = inputs(cell, c["batch"], c["steps"], c["width"])
    if path == "triton_fwd":
        from flashrnn.flashrnn2 import triton_rtiled

        def run(verify):
            y, carry = triton_rtiled.recurrence(*values, cell, sequence_context=begin_sequence(values[3], cell,
                                                torch.float16), slice_width=c["slice_width"], num_warps=c["warps"],
                                                output_mode="hidden", snapshot_dtype=torch.float32, _verify=verify)
            torch.cuda.synchronize()
            return [y, carry]
        return run
    if path == "triton_bw":
        from flashrnn.flashrnn2 import triton_rtiled, triton_rtiled_bw
        hist, _ = triton_rtiled.recurrence(*values, cell, sequence_context=begin_sequence(values[3], cell,
                                           torch.float16), slice_width=c["slice_width"], num_warps=8,
                                           output_mode="full", snapshot_dtype=torch.float32)
        g = torch.Generator().manual_seed(9)
        d_hist = torch.randn(hist.shape, generator=g).cuda()
        d_final = torch.randn(values[3].shape, generator=g).cuda()

        def run(verify):
            out = triton_rtiled_bw.backward(*values, hist, d_hist, d_final, cell, slice_width=c["slice_width"],
                                            num_warps=c["warps"], _verify=verify, fast=c.get("fast", False))
            torch.cuda.synchronize()
            return list(out)
        return run
    from flashrnn.flashrnn2 import tilelang_cluster
    sync = "cluster" if path == "tl_cluster" else "grid"

    def run(verify):
        y, carry = tilelang_cluster.recurrence(*values, cell, sequence_context=begin_sequence(values[3], cell,
                                               torch.float16), slice_width=c["slice_width"], output_mode="hidden",
                                               snapshot_dtype=torch.float32, sync=sync, batch_tile=c["bm"],
                                               threads=c["threads"], _verify=verify)
        torch.cuda.synchronize()
        return [y, carry]
    return run


def one(run, reference, verify):
    v = dict(verify)
    t0 = time.time()
    try:
        out = run(v)
        equal = all(same(a, b) for a, b in zip(out, reference)) if reference is not None else None
        counts = {k: v[k] for k in ("after_barrier", "after_mma", "after_gemm", "barrier_error") if k in v}
        violations = sum(x for k, x in counts.items() if k != "barrier_error")
        return dict(verify={k: verify[k] for k in verify}, counts=counts, violations=violations, bitwise_equal=equal,
                    seconds=round(time.time() - t0, 3))
    except Exception:  # noqa: BLE001
        return dict(verify=dict(verify), error=traceback.format_exc()[-1500:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--variants", default="", help="comma list of groups to run (correct, broken_<name>, positive)")
    a = p.parse_args()
    task = json.loads(a.task)
    rec = dict(task=task, started=time.time(), runs={})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run = runner(task)
        positive = task.get("positive_control", False)
        reference = None if positive else run(None)
        if positive:
            # Reference for the two-warpgroup control: the qualified one-warpgroup kernel of the same shape.
            from flashrnn.flashrnn2 import tilelang_cluster  # noqa: F401
            base = dict(task, config=dict(task["config"], threads=128), positive_control=False)
            reference = runner(base)(None)
        groups = {}
        if positive:
            groups["positive_threads256"] = [dict(check=True, broken="threads256")] + \
                [dict(check=True, broken="threads256", delay_seed=s) for s in range(1, SEEDS + 1)]
        else:
            groups["correct"] = [dict(check=True)] + [dict(check=True, delay_seed=s) for s in range(1, SEEDS + 1)]
            for b in BROKEN[task["path"]]:
                groups[f"broken_{b}"] = [dict(check=True, broken=b)] + \
                    [dict(check=True, broken=b, delay_seed=s) for s in range(1, SEEDS + 1)]
        if a.variants:
            groups = {k: v for k, v in groups.items() if k in a.variants.split(",")}
        summary = {}
        for name, variants in groups.items():
            runs = [one(run, reference, v) for v in variants]
            rec["runs"][name] = runs
            ok = [r for r in runs if "error" not in r]
            flagged = sum(r["violations"] > 0 for r in ok)
            differs = sum(r["bitwise_equal"] is False for r in ok)
            silent = sum((r["bitwise_equal"] is False) and r["violations"] == 0 for r in ok)
            s = dict(runs=len(runs), errors=len(runs) - len(ok), flagged=flagged, outputs_differ=differs,
                     silent_corruption=silent, completed=len(ok))
            if name == "correct":
                s["pass_"] = not s["errors"] and flagged == 0 and differs == 0
            elif name == "broken_no_sync":
                s["pass_"] = None  # reported only (not a broken kernel)
            else:
                s["pass_"] = len(ok) >= 1 and flagged >= 0.8 * len(ok) and silent == 0
            summary[name] = s
            rec["summary"] = summary
            (out / "race.json").write_text(json.dumps(rec, indent=1, default=str))
        rec["status"] = "PASS" if all(s["pass_"] is not False for s in summary.values()) else "FAIL"
    except Exception:  # noqa: BLE001
        rec.update(status="ERROR", error=traceback.format_exc()[-3000:])
    rec["finished"] = time.time()
    (out / "race.json").write_text(json.dumps(rec, indent=1, default=str))
    (out / "DONE.json").write_text(json.dumps(dict(finished=time.time())))
    print(json.dumps(dict(task=task, status=rec["status"], summary=rec.get("summary"))), flush=True)


if __name__ == "__main__":
    main()
