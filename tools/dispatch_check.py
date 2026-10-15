"""Contract checks for the table-served one-call recurrence (flashrnn2.dispatch).

Stages
  unit      (CPU is enough) policy_for/select_config: table hit and source, identity mismatch -> defaults,
            explicit policy under a wrong identity raises, continuation chunk without a config raises,
            a configuration without a one-call path raises
  contract  (CUDA) per case (cell, dtype, B, T, H, D):
            served   the configuration select_config returns equals the packaged table entry or default_config
            backend  dispatch == the backend called directly with that configuration (bitwise; full FP32 / full
                     input dtype / hidden / last, and the FP32 carry)
            prepared dispatch == PreparedSequence(config) (bitwise; full FP32, hidden, last)
            chunked  public chunked calls through dispatch with the logical call's config == one call (bitwise)
            e1       fixture_math CELL_BUDGET vs the FP64 oracle: history, carry, hidden, last
                     (LSTM warm; sLSTM zero, mixed and warm normalizer)
            With --table (a frozen-table.json from a table-fill batch), every call is served by that table as an
            explicit policy instead of the packaged tables, and --cases table checks each of its entries.
Writes dispatch.json with every check; status PASS only if every check passes.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import hashlib

import torch

from flashrnn.flashrnn2 import dispatch
from flashrnn.flashrnn2.execution_policy import (ExecutionPolicy, KernelConfig, default_config, runtime_identity,
                                                 shape_key)
from flashrnn.flashrnn2.prepared import PreparedSequence
from flashrnn.flashrnn2.state_contract import begin_sequence

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8),
                                                                     b.contiguous().view(torch.uint8))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def inputs(cell, dtype, batch, steps, heads, width, seed, device):
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    wx = torch.randn(batch, steps, 4, heads, width, generator=g) * 0.5
    r = torch.randn(4, heads, width, width, generator=g) / width ** 0.5
    bias = torch.randn(4, heads, width, generator=g) * 0.1
    initial = torch.randn(states, batch, 1, heads, width, generator=g) * 0.1
    if cell == "slstm":
        initial[2] = initial[2].abs() + 1
    return [x.to(dtype).to(device).contiguous() for x in (wx, r, bias, initial)]


def stage_unit(rec):
    tables = dispatch._packaged()
    ident = dict(tables.identity)
    checks = {}
    key = next(iter(tables.table))
    cell, dtype, b, t, h, d = key.split("/")
    shape = (cell, dtype, int(b[1:]), int(t[1:]), int(h[1:]), int(d[1:]))
    hit = dispatch.policy_for(ident)
    import flashrnn.flashrnn2.dispatch as dmod
    real = dmod.runtime_identity
    try:
        dmod.runtime_identity = lambda: ident  # the table's own identity
        cfg, src = dispatch.select_config(*shape, policy=hit)
        checks["table_hit"] = cfg == tables.table[key] and src == "table"
        cfg, src = dispatch.select_config(cell, dtype, 3, 7, 1, int(d[1:]), policy=hit)
        checks["table_miss_default"] = cfg == default_config(int(d[1:])) and src == "default"
    finally:
        dmod.runtime_identity = real
    other = dict(ident, triton="0.0.0")
    empty = dispatch.policy_for(other)
    cfg, src = dispatch.select_config(*shape, policy=empty)
    checks["identity_mismatch_defaults"] = not empty.table and cfg == default_config(int(d[1:])) and src == "default"
    try:
        dmod.runtime_identity = lambda: other
        try:
            dispatch.select_config(*shape, policy=hit)
            checks["explicit_policy_wrong_identity_raises"] = False
        except ValueError:
            checks["explicit_policy_wrong_identity_raises"] = True
    finally:
        dmod.runtime_identity = real
    values = inputs("lstm", torch.float16, 2, 5, 1, 64, 1, "cpu")
    ctx = begin_sequence(values[3], "lstm", torch.float16).advance(5)
    try:
        dispatch.recurrence(*values, "lstm", sequence_context=ctx)
        checks["continuation_without_config_raises"] = False
    except ValueError:
        checks["continuation_without_config_raises"] = True
    try:
        dispatch.recurrence(*values, "lstm", config=KernelConfig(family="simt", batch_tile=1))
        checks["no_one_call_path_raises"] = False
    except ValueError:
        checks["no_one_call_path_raises"] = True
    rec["unit"] = checks
    return all(checks.values())


def direct(values, cell, cfg, ctx, **kw):
    """The backend that cfg names, called directly."""
    if cfg.family == "tilelang":
        from flashrnn.flashrnn2 import tilelang_cluster
        return tilelang_cluster.recurrence(*values, cell, sequence_context=ctx, slice_width=cfg.slice_width,
                                           sync=cfg.sync, batch_tile=cfg.batch_tile, threads=32 * cfg.warps,
                                           prefetch=cfg.prefetch, **kw)
    if cfg.family == "mma_rtiled":
        from flashrnn.flashrnn2 import triton_rtiled
        return triton_rtiled.recurrence(*values, cell, sequence_context=ctx, slice_width=cfg.slice_width,
                                        num_warps=cfg.warps, pad_k=cfg.pad_k, **kw)
    from flashrnn.flashrnn2 import triton_persistent
    return triton_persistent.recurrence(*values, cell, weight_operand=cfg.weight_operand, return_compute_carry=True,
                                        sequence_context=ctx, num_warps=cfg.warps, **kw)


def stage_contract(case, rec, policy=None):
    import fixture_math as fm
    cell, dtype, batch, steps, heads, width = (case[k] for k in ("cell", "dtype", "batch", "steps", "heads", "width"))
    dt = DT[dtype]
    out = dict(case=case)
    cfg, source = dispatch.select_config(cell, dt, batch, steps, heads, width, policy=policy)
    tables = dispatch._packaged() if policy is None else policy
    key = shape_key(cell, dtype, batch, steps, heads, width)
    expected = tables.table.get(key, default_config(width)) if runtime_identity() == tables.identity else default_config(width)
    out["served"] = dict(config=cfg.as_dict(), source=source, expected=expected.as_dict(), pass_=cfg == expected)
    values = inputs(cell, dt, batch, steps, heads, width, 41 + batch + width, "cuda")
    modes = (("full", torch.float32), ("full", None), ("hidden", None), ("last", None))
    backend = {}
    for mode, snap in modes:
        y, c = dispatch.recurrence(*values, cell, output_mode=mode, snapshot_dtype=snap, policy=policy)
        yd, cd = direct(values, cell, cfg, begin_sequence(values[3], cell, dt), output_mode=mode, snapshot_dtype=snap)
        torch.cuda.synchronize()
        backend[f"{mode}_{'fp32' if snap else 'input'}"] = same(y, yd) and same(c, cd)
    out["backend"] = backend
    prepared = {}
    for mode, snap in (("full", torch.float32), ("hidden", None), ("last", None)):
        y, c = dispatch.recurrence(*values, cell, output_mode=mode, snapshot_dtype=snap, policy=policy)
        pr = PreparedSequence(*values, cell, output_mode=mode, snapshot_dtype=snap, config=cfg)
        yp, cp = pr.run()
        torch.cuda.synchronize()
        prepared[mode] = same(y, yp) and same(c, cp)
    out["prepared"] = prepared
    full, carry = dispatch.recurrence(*values, cell, snapshot_dtype=torch.float32, policy=policy)
    chunks, left = [], steps
    for n in (1, 3, 13):
        if n < left:
            chunks.append(n)
            left -= n
    chunks.append(left)
    ctx, state, start, parts = begin_sequence(values[3], cell, dt), values[3], 0, []
    for n in chunks:
        y, state = dispatch.recurrence(values[0][:, start:start + n], values[1], values[2], state, cell,
                                       sequence_context=ctx, snapshot_dtype=torch.float32, config=cfg)
        parts.append(y)
        ctx, start = ctx.advance(n), start + n
    torch.cuda.synchronize()
    out["chunked"] = dict(chunks=chunks, bitwise=same(torch.cat(parts, 2), full) and same(state, carry))
    e1 = []
    for mode in (("warm",) if cell == "lstm" else ("zero", "mixed", "warm")):
        fc = dict(family="dispatch", cell=cell, dtype=dtype, batch=min(batch, 16), steps=min(steps, 128), width=width,
                  heads=heads, mode=mode)
        fx = fm.make_fixture(fc)
        v = [torch.tensor(fx[k], device="cuda", dtype=dt) for k in ("wx", "r", "bias", "initial")]
        c_cfg, _ = dispatch.select_config(cell, dt, fc["batch"], fc["steps"], heads, width, policy=policy)
        y, c = dispatch.recurrence(*v, cell, snapshot_dtype=torch.float32, config=cfg)
        yh, _ = dispatch.recurrence(*v, cell, output_mode="hidden", snapshot_dtype=torch.float32, config=cfg)
        yl, _ = dispatch.recurrence(*v, cell, output_mode="last", snapshot_dtype=torch.float32, config=cfg)
        ref, final = fx["reference_fp64"], fx["reference_final_fp64"]
        checks = dict(history=fm.compare(y.cpu().numpy(), ref, fm.CELL_BUDGET),
                      carry=fm.compare(c.cpu().numpy(), final, fm.CELL_BUDGET),
                      hidden=fm.compare(yh.cpu().numpy(), ref[0], fm.CELL_BUDGET),
                      last=fm.compare(yl.cpu().numpy(), ref[0][:, -1:], fm.CELL_BUDGET))
        e1.append(dict(mode=mode, fixture=fm.case_name(fc), config_for_fixture_shape=c_cfg.as_dict(),
                       checks=checks, pass_=all(x["pass_"] for x in checks.values())))
    out["e1"] = e1
    out["pass_"] = (out["served"]["pass_"] and all(backend.values()) and all(prepared.values())
                    and out["chunked"]["bitwise"] and all(x["pass_"] for x in e1))
    rec.setdefault("contract", []).append(out)
    return out["pass_"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--stages", default="unit,contract")
    p.add_argument("--cases", default="[]", help="JSON list of {cell, dtype, batch, steps, heads, width}, or 'table'")
    p.add_argument("--table", help="frozen-table.json whose policy_table serves every call (explicit policy)")
    p.add_argument("--table-sha256")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rec = dict(started=time.time(), gates={})
    stages = a.stages.split(",")
    if "unit" in stages:
        try:
            rec["gates"]["unit"] = "PASS" if stage_unit(rec) else "FAIL"
        except Exception:  # noqa: BLE001
            rec["gates"]["unit"] = "ERROR"
            rec["unit_error"] = traceback.format_exc()[-3000:]
        write(Path(a.out) / "dispatch.json", rec)
    policy, cases = None, [] if a.cases == "table" else json.loads(a.cases)
    if a.table:
        assert hashlib.sha256(Path(a.table).read_bytes()).hexdigest() == a.table_sha256, "frozen table changed"
        frozen = json.loads(Path(a.table).read_text())
        policy = ExecutionPolicy(frozen["policy_table"], identity=frozen["policy_identity"])
        rec["table"] = dict(sha256=a.table_sha256, entries=len(policy.table))
        if a.cases == "table":
            for key in sorted(policy.table):
                cell, dtype, b, t, h, d = key.split("/")
                cases.append(dict(cell=cell, dtype=dtype, batch=int(b[1:]), steps=int(t[1:]), heads=int(h[1:]),
                                  width=int(d[1:])))
    if "contract" in stages:
        rec["identity"] = runtime_identity()
        for i, case in enumerate(cases):
            name = f"contract{i}"
            try:
                rec["gates"][name] = "PASS" if stage_contract(case, rec, policy) else "FAIL"
            except Exception:  # noqa: BLE001
                rec["gates"][name] = "ERROR"
                rec[f"{name}_error"] = dict(case=case, trace=traceback.format_exc()[-3000:])
            write(Path(a.out) / "dispatch.json", rec)
    rec["status"] = "PASS" if rec["gates"] and all(v == "PASS" for v in rec["gates"].values()) else "FAIL"
    rec["finished"] = time.time()
    write(Path(a.out) / "dispatch.json", rec)
    (Path(a.out) / "DONE.json").write_text(json.dumps(dict(finished=time.time())))
    print(json.dumps(dict(gates=rec["gates"], status=rec["status"])), flush=True)


if __name__ == "__main__":
    main()
