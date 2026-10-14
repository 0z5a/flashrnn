"""M0 GPU probe for R-tiled FlashRNN2 (H100): resources, barrier protocol, numerics.

Writes one JSON receipt per stage under --out. Exploratory timings here are not
the paired ABBA/BAAB protocol and must not be reported as speedups.
"""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
import triton

from flashrnn.flashrnn2 import reference, triton_persistent, triton_rtiled
from flashrnn.flashrnn2.state_contract import begin_sequence

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def identity():
    p = torch.cuda.get_device_properties(0)
    return dict(
        gpu=p.name, capability=list(torch.cuda.get_device_capability()), sms=p.multi_processor_count,
        triton=triton.__version__, torch=torch.__version__, cuda=torch.version.cuda,
        uuid=os.environ.get("FRNN_GPU_UUID"), host=os.uname().nodename,
        source={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in SOURCES},
    )


SOURCES = [
    "flashrnn/flashrnn2/triton_rtiled.py",
    "flashrnn/flashrnn2/triton_persistent.py",
    "flashrnn/flashrnn2/state_contract.py",
    "flashrnn/flashrnn2/reference.py",
    "tools/rtiled_m0.py",
]


def fixture(cell, dtype, batch, steps, heads, width, seed, normalizer="nonzero"):
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    scale = 1.0 / width ** 0.5
    wx = torch.randn(batch, steps, 4, heads, width, generator=g) * 0.5
    r = torch.randn(4, heads, width, width, generator=g) * scale
    bias = torch.randn(4, heads, width, generator=g) * 0.1
    initial = torch.randn(states, batch, 1, heads, width, generator=g) * 0.1
    if cell == "slstm":
        initial[2] = 0 if normalizer == "zero" else initial[2].abs() + 1
        initial[3] = 0 if normalizer == "zero" else initial[3]
    return tuple(x.to(dtype).cuda() for x in (wx, r, bias, initial))


def run(values, cell, **kw):
    ctx = begin_sequence(values[3], cell, values[0].dtype)
    return triton_rtiled.recurrence(*values, cell, sequence_context=ctx, **kw)


def run_chunks(values, cell, lengths, **kw):
    wx, r, b, initial = values
    ctx = begin_sequence(initial, cell, wx.dtype)
    outs, carry, start = [], initial, 0
    for n in lengths:
        y, carry = triton_rtiled.recurrence(wx[:, start:start + n], r, b, carry, cell, sequence_context=ctx, **kw)
        outs.append(y)
        ctx, start = ctx.advance(n), start + n
    return torch.cat(outs, 2), carry


def same_bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def oracle(values, cell):
    """FP32 states with the hidden MMA operand rounded to the input dtype (PR #79 oracle)."""
    wx, r, b, initial = values
    ctx = begin_sequence(initial.float(), cell, wx.dtype)
    return reference.recurrence(*(x.float() for x in values), cell, wx.dtype, sequence_context=ctx)


def err(a, b):
    a, b = a.double(), b.double()
    d = (a - b).abs()
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                max_rel=float((d / b.abs().clamp_min(1e-3)).max()))


def ptx_features(ptx):
    keys = ("mma.sync", "wgmma", "ldmatrix", "ld.global.cg", "atom.acquire.gpu", "atom.release.gpu",
            "atom.global.acquire", "atom.global.release", "bar.sync", "barrier.cluster", "mapa", "fence")
    return {k: ptx.count(k) for k in keys}


def stage_resources(out, args):
    rows = []
    for cell in ("lstm", "slstm"):
        for width in args.widths:
            for s in (16, 32, 64, 128):
                if s > triton.next_power_of_2(width):
                    continue
                for warps in (4, 8):
                    for pad in ((False, True) if triton_rtiled.k_parts(width)[1] else (False,)):
                        rec = dict(cell=cell, dtype="float16", width=width, slice_width=s, warps=warps, pad_k=pad)
                        try:
                            values = fixture(cell, torch.float16, 2, 3, 1, width, 7)
                            d = {}
                            run(values, cell, slice_width=s, num_warps=warps, pad_k=pad, diagnostics=d)
                            torch.cuda.synchronize()
                            rec.update(status="COMPILED", registers=d["registers"], spills=d["spills"],
                                       shared_bytes=d["shared_bytes"], splits=d["splits"], k_parts=d["k_parts"],
                                       ptx=ptx_features(d["ptx"]),
                                       error_flag=int(d["error_flag"].item()),
                                       sync_ok=d["sync_counts"].tolist()[0] == d["expected_sync_counts"])
                        except Exception as e:  # noqa: BLE001 - resource failures are data here
                            rec.update(status="FAILED", error=f"{type(e).__name__}: {str(e)[:300]}")
                        rows.append(rec)
                        print(json.dumps(rec), flush=True)
    write(out / "resources.json", dict(identity=identity(), rows=rows))
    return rows


def stage_protocol(out, args, feasible):
    rows = []
    shapes = [(1, 33, 1), (3, 17, 2), (33, 40, 2)]
    for cfg in feasible:
        if past(args):
            break
        for batch, steps, heads in shapes:
            for normalizer in (("zero", "nonzero") if cfg["cell"] == "slstm" else ("nonzero",)):
                for dtype in ("float16", "bfloat16"):
                    rec = dict(cfg, dtype=dtype, batch=batch, steps=steps, heads=heads, normalizer=normalizer)
                    try:
                        values = fixture(cfg["cell"], DT[dtype], batch, steps, heads, cfg["width"], 11, normalizer)
                        kw = dict(slice_width=cfg["slice_width"], num_warps=cfg["warps"], pad_k=cfg["pad_k"])
                        checks = {}
                        d = {}
                        grid = run(values, cfg["cell"], sync="grid", diagnostics=d, **kw)
                        launch = run(values, cfg["cell"], sync="launch", **kw)
                        torch.cuda.synchronize()
                        checks["grid_error_flag"] = int(d["error_flag"].item()) == 0
                        checks["grid_sync_counts"] = d["sync_counts"].tolist()[0] == d["expected_sync_counts"]
                        checks["grid_vs_launch_bitwise"] = all(same_bits(a, b) for a, b in zip(grid, launch))
                        # Force several groups per slot (serial group loop and its barrier).
                        d2 = {}
                        few = run(values, cfg["cell"], sync="grid", max_ctas=cfg_splits(cfg), diagnostics=d2, **kw)
                        torch.cuda.synchronize()
                        checks["serial_groups_bitwise"] = all(same_bits(a, b) for a, b in zip(grid, few))
                        checks["serial_groups_sync"] = (int(d2["error_flag"].item()) == 0
                                                        and d2["sync_counts"].tolist()[0] == d2["expected_sync_counts"])
                        lengths = [n for n in (1, 3, 13, 7, 9, 64) if n][: max(1, steps)]
                        acc, parts = 0, []
                        for n in lengths:
                            if acc + n > steps:
                                break
                            parts.append(n)
                            acc += n
                        if acc < steps:
                            parts.append(steps - acc)
                        chunked = run_chunks(values, cfg["cell"], parts, **kw)
                        checks["chunks_bitwise"] = all(same_bits(a, b) for a, b in zip(grid, chunked))
                        modes = {}
                        for mode in ("hidden", "last"):
                            y, carry = run(values, cfg["cell"], output_mode=mode, **kw)
                            h = grid[0][0] if mode == "hidden" else grid[0][0][:, -1:]
                            modes[mode] = same_bits(y, h) and same_bits(carry, grid[1])
                        checks["reduced_outputs_bitwise"] = all(modes.values())
                        fp32 = run(values, cfg["cell"], snapshot_dtype=torch.float32, **kw)
                        ref, ref_carry = oracle(values, cfg["cell"])
                        rec["oracle_fp32"] = dict(history=err(fp32[0], ref), carry=err(fp32[1], ref_carry))
                        if cfg["width"] <= 128:
                            ctx = begin_sequence(values[3], cfg["cell"], values[0].dtype)
                            single = triton_persistent.recurrence(
                                *values, cfg["cell"], return_compute_carry=True, sequence_context=ctx,
                                snapshot_dtype=torch.float32)
                            rec["single_cta_bitwise"] = all(same_bits(a, b) for a, b in zip(fp32, single))
                            rec["single_cta_diff"] = dict(history=err(fp32[0], single[0]), carry=err(fp32[1], single[1]))
                        rec["checks"] = checks
                        rec["status"] = "PASS" if all(checks.values()) else "FAIL"
                    except Exception as e:  # noqa: BLE001
                        rec.update(status="ERROR", error=traceback.format_exc()[-1500:])
                    rows.append(rec)
                    print(json.dumps({k: v for k, v in rec.items() if k != "error"}), flush=True)
    write(out / "protocol.json", dict(identity=identity(), rows=rows))
    return rows


def past(args):
    return time.time() > args.deadline


def widest(feasible):
    """Fewest splits per (cell, width): the largest compiled slice with four warps."""
    best = {}
    for c in feasible:
        if c["warps"] == 4 and not c["pad_k"]:
            key = (c["cell"], c["width"])
            if key not in best or c["slice_width"] > best[key]["slice_width"]:
                best[key] = c
    return list(best.values())


def cfg_splits(cfg):
    return triton.cdiv(cfg["width"], cfg["slice_width"])


def stage_stress(out, args, feasible):
    rows = []
    for cfg in widest(feasible):
        if past(args):
            break
        kw = dict(slice_width=cfg["slice_width"], num_warps=cfg["warps"], pad_k=cfg["pad_k"])
        for batch, steps, heads in ((16, 256, 1), (64, 128, 4), (128, 64, 8)):
            values = fixture(cfg["cell"], torch.float16, batch, steps, heads, cfg["width"], 5)
            first, flags, counts = None, 0, True
            start = time.time()
            for rep in range(args.stress_reps):
                d = {}
                y = run(values, cfg["cell"], output_mode="last", diagnostics=d, **kw)
                if first is None:
                    first = y
                elif not all(same_bits(a, b) for a, b in zip(first, y)):
                    flags += 1000
                flags += int(d["error_flag"].item())
                counts &= d["sync_counts"].tolist()[0] == d["expected_sync_counts"]
            rec = dict(cfg, batch=batch, steps=steps, heads=heads, reps=args.stress_reps,
                       slots=d["slots"], ctas=d["ctas"], failures=flags, sync_counts_ok=counts,
                       seconds=time.time() - start, status="PASS" if flags == 0 and counts else "FAIL")
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    write(out / "stress.json", dict(identity=identity(), rows=rows))


def timed(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def stage_explore(out, args, feasible):
    rows = []
    for cfg in feasible:
        if cfg["cell"] != "lstm" or past(args):
            continue
        kw = dict(slice_width=cfg["slice_width"], num_warps=cfg["warps"], pad_k=cfg["pad_k"])
        for batch, steps in ((16, 256), (64, 256)):
            values = fixture("lstm", torch.float16, batch, steps, 1, cfg["width"], 3)
            rec = dict(cfg, batch=batch, steps=steps)
            rec["grid_ms"] = timed(lambda: run(values, "lstm", output_mode="hidden", **kw))
            rec["launch_ms"] = timed(lambda: run(values, "lstm", sync="launch", output_mode="hidden", **kw), 3)
            if cfg["width"] <= 128:
                ctx = begin_sequence(values[3], "lstm", torch.float16)
                rec["single_cta_ms"] = timed(lambda: triton_persistent.recurrence(
                    *values, "lstm", return_compute_carry=True, sequence_context=ctx, output_mode="hidden"))
            rec["grid_us_per_step"] = rec["grid_ms"] * 1000 / steps
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    write(out / "explore-timing.json", dict(identity=identity(), note="exploratory, not paired", rows=rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--stages", default="resources,protocol,stress,explore")
    p.add_argument("--widths", default="64,96,128,192,256,384,512,768")
    p.add_argument("--stress-reps", type=int, default=50)
    p.add_argument("--minutes", type=float, default=100.0)
    args = p.parse_args()
    args.deadline = time.time() + 60 * args.minutes
    args.widths = [int(x) for x in args.widths.split(",")]
    out = Path(args.out)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stages = args.stages.split(",")
    write(out / "identity.json", identity())
    rows = stage_resources(out, args)
    ok = [r for r in rows if r["status"] == "COMPILED" and r["error_flag"] == 0 and r["sync_ok"]]
    feasible = [dict(cell=r["cell"], width=r["width"], slice_width=r["slice_width"], warps=r["warps"],
                     pad_k=r["pad_k"]) for r in ok]
    write(out / "feasible.json", feasible)
    # Protocol/numerics: every compiled slice at four warps, and padded K only where it differs.
    checked = [c for c in feasible if c["warps"] == 4 and (not c["pad_k"] or c["width"] in (192, 384))]
    if "protocol" in stages:
        stage_protocol(out, args, checked)
    if "stress" in stages:
        stage_stress(out, args, feasible)
    if "explore" in stages:
        stage_explore(out, args, feasible)
    write(out / "DONE.json", dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


if __name__ == "__main__":
    main()
