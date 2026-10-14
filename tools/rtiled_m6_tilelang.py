"""M6: qualify the full-contract TileLang cluster path (flashrnn2.tilelang_cluster) at the M0 standard.

One process per (cell, dtype, D, S). Stages:
  e1        fixture_math FP64 oracle under CELL_BUDGET: history/carry (full, FP32 snapshot), hidden and last
  protocol  bitwise: per-step launches (chunks of 1) = ragged chunks = public chunked calls = one call;
            hidden/last = full components; 20 repeats; non-default stream with a producer; prepared eager
            and CUDA Graph replay (also after changing the bound initial state); difference to Triton R tiling
  stress    oversubscribed grids (many queued clusters), 50 bit-identical repeats
Writes m6.json with every check; status PASS only if every gate passes.
"""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

from flashrnn.flashrnn2 import tilelang_cluster, triton_rtiled
from flashrnn.flashrnn2.execution_policy import KernelConfig, default_config, runtime_identity
from flashrnn.flashrnn2.prepared import PreparedSequence
from flashrnn.flashrnn2.state_contract import begin_sequence

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}
SOURCES = ["flashrnn/flashrnn2/tilelang_cluster.py", "flashrnn/flashrnn2/prepared.py",
           "flashrnn/flashrnn2/execution_policy.py", "flashrnn/flashrnn2/triton_rtiled.py",
           "tools/rtiled_m6_tilelang.py", "fixture_math.py"]


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8),
                                                                     b.contiguous().view(torch.uint8))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def inputs(cell, dtype, batch, steps, heads, width, seed, normalizer="nonzero"):
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    wx = torch.randn(batch, steps, 4, heads, width, generator=g) * 0.5
    r = torch.randn(4, heads, width, width, generator=g) / width ** 0.5
    bias = torch.randn(4, heads, width, generator=g) * 0.1
    initial = torch.randn(states, batch, 1, heads, width, generator=g) * 0.1
    if cell == "slstm":
        initial[2] = 0 if normalizer == "zero" else initial[2].abs() + 1
        initial[3] = 0 if normalizer == "zero" else initial[3]
    return [x.to(dtype).cuda().contiguous() for x in (wx, r, bias, initial)]


def call(values, cell, s, **kw):
    ctx = begin_sequence(values[3], cell, values[0].dtype)
    return tilelang_cluster.recurrence(*values, cell, sequence_context=ctx, slice_width=s, **kw)


def ragged_chunks(steps, sizes=(1, 3, 13, 7, 9), count=3):
    """Uneven chunk lengths: up to `count` leading sizes, then the positive remainder (sum == steps)."""
    chunks, left = [], steps
    for n in sizes:
        if len(chunks) == count or n >= left:
            break
        chunks.append(n)
        left -= n
    return chunks + [left]


def stage_e1(a, rec):
    import fixture_math as fm
    rows = []
    for batch, steps, heads in ((3, 17, 2), (16, 128, 1)):
        for mode in (("warm",) if a.cell == "lstm" else ("zero", "mixed", "warm")):
            case = dict(family="m6", cell=a.cell, dtype=a.dtype, batch=batch, steps=steps, width=a.width,
                        heads=heads, mode=mode)
            fx = fm.make_fixture(case)
            values = [torch.tensor(fx[k], device="cuda", dtype=DT[a.dtype]) for k in ("wx", "r", "bias", "initial")]
            ref, final = fx["reference_fp64"], fx["reference_final_fp64"]
            y, carry = call(values, a.cell, a.slice, snapshot_dtype=torch.float32)
            yh, _ = call(values, a.cell, a.slice, output_mode="hidden", snapshot_dtype=torch.float32)
            yl, _ = call(values, a.cell, a.slice, output_mode="last", snapshot_dtype=torch.float32)
            checks = dict(history=fm.compare(y.cpu().numpy(), ref, fm.CELL_BUDGET),
                          carry=fm.compare(carry.cpu().numpy(), final, fm.CELL_BUDGET),
                          hidden=fm.compare(yh.cpu().numpy(), ref[0], fm.CELL_BUDGET),
                          last=fm.compare(yl.cpu().numpy(), ref[0][:, -1:], fm.CELL_BUDGET))
            rows.append(dict(case=fm.case_name(case), checks=checks,
                             status="PASS" if all(c["pass_"] for c in checks.values()) else "NUMERICAL_FAIL"))
    rec["e1"] = rows
    return all(r["status"] == "PASS" for r in rows)


def stage_protocol(a, rec):
    rows, ok = [], True
    rec["protocol"] = rows  # rows completed before a later exception stay in m6.json
    cfg = KernelConfig(family="tilelang", slice_width=a.slice, sync="cluster")
    for batch, steps, heads in ((1, 33, 1), (3, 17, 2), (33, 40, 2)):
        for normalizer in (("zero", "nonzero") if a.cell == "slstm" else ("nonzero",)):
            values = inputs(a.cell, DT[a.dtype], batch, steps, heads, a.width, 11 + batch, normalizer)
            checks = {}
            full, carry = call(values, a.cell, a.slice, snapshot_dtype=torch.float32)
            # Per-step launches through the FP32 carry: the kernel-boundary oracle for the in-kernel exchange.
            p1 = PreparedSequence(*values, a.cell, snapshot_dtype=torch.float32, config=cfg, chunks=[1] * steps)
            y1, c1 = p1.run()
            checks["per_step_launches_bitwise"] = same(y1, full) and same(c1, carry)
            ragged = ragged_chunks(steps)
            pr = PreparedSequence(*values, a.cell, snapshot_dtype=torch.float32, config=cfg, chunks=ragged)
            yr, cr = pr.run()
            checks["prepared_ragged_chunks_bitwise"] = same(yr, full) and same(cr, carry)
            ctx = begin_sequence(values[3], a.cell, values[0].dtype)
            parts, state, start = [], values[3], 0
            for n in (3, steps - 3):
                y, state = tilelang_cluster.recurrence(values[0][:, start:start + n], values[1], values[2], state,
                                                       a.cell, sequence_context=ctx, slice_width=a.slice,
                                                       snapshot_dtype=torch.float32)
                parts.append(y)
                ctx, start = ctx.advance(n), start + n
            checks["public_chunked_calls_bitwise"] = same(torch.cat(parts, 2), full) and same(state, carry)
            full_in, _ = call(values, a.cell, a.slice)
            yh, ch = call(values, a.cell, a.slice, output_mode="hidden")
            yl, cl = call(values, a.cell, a.slice, output_mode="last")
            checks["hidden_last_equal_full_bitwise"] = (same(yh, full_in[0]) and same(yl, full_in[0][:, -1:])
                                                        and same(ch, carry) and same(cl, carry))
            checks["repeat_20_bitwise"] = all(
                same(y, full) and same(c, carry)
                for y, c in (call(values, a.cell, a.slice, snapshot_dtype=torch.float32) for _ in range(20)))
            # Non-default stream with a producer kernel writing the input right before the call.
            side = torch.cuda.Stream()
            staged = torch.zeros_like(values[0])
            torch.cuda.synchronize()
            with torch.cuda.stream(side):
                staged.copy_(values[0])
                ys, cs = call([staged] + values[1:], a.cell, a.slice, snapshot_dtype=torch.float32)
            side.synchronize()
            checks["side_stream_bitwise"] = same(ys, full) and same(cs, carry)
            # Prepared CUDA Graph replay, then a changed bound initial state must replay like eager.
            pg = PreparedSequence(*[v.clone() for v in values], a.cell, snapshot_dtype=torch.float32, config=cfg,
                                  chunks=ragged)
            try:
                pg.capture()
                checks["graph_replay_bitwise"] = all(same(pg.run()[0], full) and same(pg.final, carry)
                                                     for _ in range(20))
                pg.initial.mul_(0.5)
                yg, cg = pg.run()
                pe = PreparedSequence(*[x.clone() for x in (pg.wx, pg.recurrent, pg.bias, pg.initial)], a.cell,
                                      snapshot_dtype=torch.float32, config=cfg, chunks=ragged)
                ye, ce = pe.run()
                checks["graph_tracks_bound_initial"] = same(yg, ye) and same(cg, ce)
            except Exception as e:  # noqa: BLE001 - capture failure is a recorded result
                checks["graph_replay_bitwise"] = False
                checks["graph_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            finally:
                pg.close()
            # Informational, not a gate: distance to the conservative Triton R-tiled default.
            tcfg = default_config(a.width)
            try:
                tctx = begin_sequence(values[3], a.cell, values[0].dtype)
                ty, tc = triton_rtiled.recurrence(*values, a.cell, sequence_context=tctx,
                                                  slice_width=tcfg.slice_width, num_warps=tcfg.warps,
                                                  snapshot_dtype=torch.float32)
                torch.cuda.synchronize()
                vs_triton = dict(slice_width=tcfg.slice_width, warps=tcfg.warps,
                                 history_max_abs=float((full - ty).abs().max()),
                                 carry_max_abs=float((carry - tc).abs().max()),
                                 bitwise=same(full, ty) and same(carry, tc))
            except Exception as e:  # noqa: BLE001 - recorded, never turns the qualification into ERROR
                vs_triton = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
            row = dict(batch=batch, steps=steps, heads=heads, normalizer=normalizer, chunks=ragged, checks=checks,
                       vs_triton=vs_triton)
            row["status"] = "PASS" if all(v is True for k, v in checks.items() if not k.endswith("_error")) \
                else "FAIL"
            ok &= row["status"] == "PASS"
            rows.append(row)
            write(Path(a.out) / "m6.json", rec)
    return ok


def stage_stress(a, rec):
    rows, ok = [], True
    for batch, steps, heads in ((64, 128, 4), (128, 64, 8)):
        values = inputs(a.cell, DT[a.dtype], batch, steps, heads, a.width, 5)
        first = call(values, a.cell, a.slice, output_mode="last")
        good = all(same(y, first[0]) and same(c, first[1])
                   for y, c in (call(values, a.cell, a.slice, output_mode="last") for _ in range(50)))
        k = a.width // a.slice
        rows.append(dict(batch=batch, steps=steps, heads=heads, ctas=k * ((batch + 15) // 16) * heads,
                         repeats=50, bitwise=good, status="PASS" if good else "FAIL"))
        ok &= good
    rec["stress"] = rows
    return ok


def stage_sanity(a, rec):
    """Small case for compute-sanitizer: one call, per-step launches and a reduced output."""
    values = inputs(a.cell, DT[a.dtype], 3, 9, 2, a.width, 3)
    cfg = KernelConfig(family="tilelang", slice_width=a.slice, sync="cluster")
    full, carry = call(values, a.cell, a.slice, snapshot_dtype=torch.float32)
    y1, c1 = PreparedSequence(*values, a.cell, snapshot_dtype=torch.float32, config=cfg, chunks=[1] * 9).run()
    yl, _ = call(values, a.cell, a.slice, output_mode="last", snapshot_dtype=torch.float32)
    torch.cuda.synchronize()
    ok = same(y1, full) and same(c1, carry) and same(yl, full[0][:, -1:])
    rec["sanity"] = dict(bitwise=ok)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--dtype", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--slice", type=int, required=True)
    p.add_argument("--stages", default="e1,protocol,stress")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rec = dict(cell=a.cell, dtype=a.dtype, width=a.width, slice_width=a.slice, identity=runtime_identity(),
               source={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in SOURCES},
               uuid=os.environ.get("FRNN_GPU_UUID"), started=time.time(), gates={})
    stages = a.stages.split(",")
    for name, fn in (("e1", stage_e1), ("protocol", stage_protocol), ("stress", stage_stress),
                     ("sanity", stage_sanity)):
        if name not in stages:
            continue
        try:
            rec["gates"][name] = "PASS" if fn(a, rec) else "FAIL"
        except Exception:  # noqa: BLE001
            rec["gates"][name] = "ERROR"
            rec[f"{name}_error"] = traceback.format_exc()[-3000:]
        write(Path(a.out) / "m6.json", rec)
    rec["status"] = "PASS" if rec["gates"] and all(v == "PASS" for v in rec["gates"].values()) else "FAIL"
    rec["finished"] = time.time()
    write(Path(a.out) / "m6.json", rec)
    write(Path(a.out) / "DONE.json", dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    print(json.dumps({k: rec[k] for k in ("cell", "dtype", "width", "slice_width", "gates", "status")}), flush=True)


if __name__ == "__main__":
    main()
