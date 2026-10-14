"""High-concurrency (HC) serving of the FlashRNN2 classifier: R0 decomposition.

Workload: C requests of T tokens arrive together (closed batch) at the existing
classifier (4 layers, vocab 512, FP16; tools/rtiled_m4.Model). One process per
(cell, D); writes decompose.json.

Strategies (same public kernel path, configuration from the frozen M6+M7 tables):
  s0      C/B0 groups served one after another (the existing harness), config = policy(B0)
  s0p     all C requests in one call per layer, config = policy(C) (the default: no table entry)
  s0p_cfg all C requests in one call per layer, config = policy(B0) (isolates batching from the config)

Measured per (C, strategy): makespan (IDs on host -> logits on host), per-request
completion (its group's completion), throughput; CUDA-event stage spans (H2D,
embedding, input projection, recurrence, slice, readout, D2H) and the profiler's
total kernel time (GPU-busy fraction); the recurrence SM footprint.
Recurrence microbench: µs/step, CTAs and SM·µs per row-step per config and batch.
Checks: logits vs the FP64 model oracle (MODEL_BUDGET) for the first requests;
recurrence rows bitwise independent of batch composition for every config used.
"""
import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import torch

import rtiled_m4 as m4
from flashrnn.flashrnn2 import triton_rtiled
from flashrnn.flashrnn2.execution_policy import TABLES, ExecutionPolicy, KernelConfig, runtime_identity
from flashrnn.flashrnn2.state_contract import begin_sequence

SMS = None


def policy():
    return ExecutionPolicy.from_json(TABLES / "h100-rtiled-m6-20261014.json", TABLES / "h100-rtiled-m7-20261014.json")


def arm(cell, cfg):
    def run(wx, r, b, init):
        ctx = begin_sequence(init, cell, wx.dtype)
        if cfg.family == "tilelang":
            from flashrnn.flashrnn2 import tilelang_cluster
            y, _ = tilelang_cluster.recurrence(wx, r, b, init, cell, sequence_context=ctx, slice_width=cfg.slice_width,
                                               output_mode="hidden", sync=cfg.sync)
        elif cfg.family == "mma_rtiled":
            y, _ = triton_rtiled.recurrence(wx, r, b, init, cell, sequence_context=ctx, slice_width=cfg.slice_width,
                                            num_warps=cfg.warps, pad_k=cfg.pad_k, output_mode="hidden")
        else:
            raise ValueError(f"unsupported family {cfg.family}")
        return y
    return run


def footprint(cfg, batch, width, heads=1):
    """CTAs one recurrence launch keeps resident (one CTA per SM for every R-tiled path here)."""
    k = width // cfg.slice_width
    groups = (batch + 15) // 16 * heads
    if cfg.family == "mma_rtiled":
        splits, groups, resident = triton_rtiled.plan(batch, heads, width, cfg.slice_width, sms=SMS)
        return dict(k=splits, groups=groups, resident_groups=resident, ctas=splits * resident)
    if cfg.sync == "grid":
        from flashrnn.flashrnn2 import tilelang_cluster
        k, resident, slots = tilelang_cluster.grid_plan(batch, heads, width, cfg.slice_width, SMS)
        return dict(k=k, groups=groups, resident_groups=resident, slots=slots, ctas=k * resident)
    # Cluster launch: all clusters are queued; at most SMS // k run at once (fewer if GPCs fragment).
    resident = min(groups, SMS // k)
    return dict(k=k, groups=groups, resident_groups_upper=resident, ctas_upper=k * resident)


def serve(model, ids_groups, fn, events=None):
    """Serve groups one after another; returns logits per group and completion offsets (ms)."""
    t0 = time.perf_counter_ns()
    outs, done = [], []
    for ids in ids_groups:
        mark = (lambda name: events.append((name, torch.cuda.Event(enable_timing=True)))) if events is not None else None
        if mark:
            mark("start")
            events[-1][1].record()
        x = model.embedding[ids.to("cuda")]
        if mark:
            mark("embed"); events[-1][1].record()
        for w, r, b in model.weights:
            wx = torch.einsum("bti,ghdi->btghd", x, w)
            if mark:
                mark("proj"); events[-1][1].record()
            hidden = fn(wx, r, b, model.initial(ids.shape[0], model.dtype))
            if mark:
                mark("recur"); events[-1][1].record()
            x = hidden[:, :, 0]
        logits = x[:, -1] @ model.readout
        if mark:
            mark("readout"); events[-1][1].record()
        outs.append(logits.cpu())
        if mark:
            mark("d2h"); events[-1][1].record()
        done.append((time.perf_counter_ns() - t0) / 1e6)
    return outs, done


def stage_spans(events):
    """GPU-timeline spans between consecutive markers; 'between_groups' is the gap from one group's
    host delivery to the next group's first enqueued work (host-side time with an idle stream)."""
    torch.cuda.synchronize()
    spans = {}
    for (prev, a), (name, b) in zip(events, events[1:]):
        key = "between_groups" if name == "start" else name
        spans[key] = spans.get(key, 0.0) + a.elapsed_time(b)
    return spans


def kernel_time(model, groups, fn):
    """Device activity from the profiler: busy time (union of device intervals) and the top kernels."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter_ns()
        serve(model, groups, fn)
        torch.cuda.synchronize()
        wall = (time.perf_counter_ns() - t0) / 1e6
    device = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    spans = sorted((e.time_range.start, e.time_range.end) for e in device)
    busy, end = 0.0, None
    for a, b in spans:
        if end is None or a > end:
            busy += b - a
            end = b
        elif b > end:
            busy += b - end
            end = b
    by = {}
    for e in device:
        key = "memcpy" if e.name.startswith("Memcpy") else "memset" if e.name.startswith("Memset") else e.name[:120]
        ms, n = by.get(key, (0.0, 0))
        by[key] = (ms + e.time_range.elapsed_us() / 1e3, n + 1)
    top = sorted(by.items(), key=lambda kv: -kv[1][0])[:12]
    return dict(profiled_wall_ms=wall, device_busy_ms=busy / 1e3, device_events=len(device),
                busy_fraction=busy / 1e3 / wall if wall else None,
                top=[dict(name=k, ms=round(v[0], 3), count=v[1]) for k, v in top])


def check_rows(cell, width, batch, steps, cfgs, seed):
    """Recurrence output rows must not depend on which other rows share the launch."""
    g = torch.Generator().manual_seed(seed)
    model = m4.Model(cell, width)
    w, r, b = model.weights[0]
    wx = (torch.randn(batch, steps, 4, 1, width, generator=g) * 0.5).to("cuda", torch.float16)
    out = {}
    for name, cfg in cfgs.items():
        fn = arm(cell, cfg)
        full = fn(wx, r, b, model.initial(batch, torch.float16))
        parts = torch.cat([fn(wx[i:i + 32].contiguous(), r, b, model.initial(min(32, batch - i), torch.float16))
                           for i in range(0, batch, 32)])
        torch.cuda.synchronize()
        out[name] = dict(config=cfg.as_dict(), bitwise=bool(torch.equal(full.view(torch.int16), parts.view(torch.int16))),
                         max_abs=float((full.float() - parts.float()).abs().max()))
    return out


def microbench(cell, width, steps, cfgs, batches, calls=10):
    model = m4.Model(cell, width)
    w, r, b = model.weights[0]
    rows = []
    for batch in batches:
        g = torch.Generator().manual_seed(7 + batch)
        wx = (torch.randn(batch, steps, 4, 1, width, generator=g) * 0.5).to("cuda", torch.float16)
        init = model.initial(batch, torch.float16)
        for name, cfg in cfgs.items():
            fn = arm(cell, cfg)
            try:
                for _ in range(2):
                    fn(wx, r, b, init)
                times = []
                for _ in range(calls):
                    a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    fn(wx, r, b, init)
                    e.record()
                    torch.cuda.synchronize()
                    times.append(a.elapsed_time(e))
                ms = statistics.median(times)
                fp = footprint(cfg, batch, width)
                ctas = fp.get("ctas", fp.get("ctas_upper"))
                rows.append(dict(config=name, batch=batch, ms=ms, us_per_step=1e3 * ms / steps, footprint=fp,
                                 sm_us_per_row_step=ctas * 1e3 * ms / steps / batch))
            except Exception as e:  # noqa: BLE001 - recorded
                rows.append(dict(config=name, batch=batch, error=f"{type(e).__name__}: {str(e)[:300]}"))
            print(json.dumps({k: rows[-1].get(k) for k in ("config", "batch", "us_per_step", "sm_us_per_row_step", "error")}),
                  flush=True)
    return rows


def main():
    global SMS
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--group", type=int, default=32)
    p.add_argument("--concurrency", default="64,256,1024")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--no-micro", action="store_true", help="skip the recurrence microbench and row checks")
    a = p.parse_args()
    SMS = torch.cuda.get_device_properties(0).multi_processor_count
    torch.backends.cuda.matmul.allow_tf32 = False
    ident = runtime_identity()
    pol = policy()
    rec = dict(cell=a.cell, width=a.width, steps=a.steps, group=a.group, sms=SMS, identity=ident, cases=[], started=time.time())
    out = Path(a.out)
    model = m4.Model(a.cell, a.width)
    pick = lambda batch: pol.select(a.cell, "float16", batch, a.steps, 1, a.width, identity=ident)  # noqa: E731
    with torch.inference_mode():
        for conc in map(int, a.concurrency.split(",")):
            case = dict(concurrency=conc)
            try:
                g = torch.Generator().manual_seed(20261015 + conc)
                ids = torch.randint(0, m4.VOCAB, (conc, a.steps), generator=g)
                groups = list(ids.split(a.group))
                strategies = dict(s0=(groups, pick(a.group)), s0p=([ids], pick(conc)), s0p_cfg=([ids], pick(a.group)))
                case["configs"] = {k: dict(config=v[1].as_dict(), footprint=footprint(v[1], len(v[0][0]), a.width))
                                   for k, v in strategies.items()}
                fns = {k: arm(a.cell, v[1]) for k, v in strategies.items()}
                # Correctness: first 32 requests vs the FP64 oracle under the model budget.
                o_logits, _ = model.oracle(ids[:32])
                case["oracle"] = {}
                for k, (grp, _) in strategies.items():
                    outs, _ = serve(model, grp, fns[k])
                    case["oracle"][k] = m4.compare(torch.cat(outs)[:32].cuda(), o_logits)
                # Interleaved repetitions (rotate the order each round).
                names = list(strategies)
                for k in names:
                    serve(model, strategies[k][0], fns[k])
                runs = {k: [] for k in names}
                for rep in range(a.reps):
                    order = names[rep % len(names):] + names[:rep % len(names)]
                    for k in order:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter_ns()
                        _, done = serve(model, strategies[k][0], fns[k])
                        makespan = (time.perf_counter_ns() - t0) / 1e6
                        per_request = [d for d, grp in zip(done, strategies[k][0]) for _ in range(len(grp))]
                        runs[k].append(dict(makespan_ms=makespan, p50_ms=statistics.median(per_request),
                                            p90_ms=sorted(per_request)[int(0.9 * (len(per_request) - 1))]))
                case["runs"] = runs
                case["summary"] = {k: dict(makespan_ms=statistics.median(r["makespan_ms"] for r in v),
                                           p50_ms=statistics.median(r["p50_ms"] for r in v),
                                           p90_ms=statistics.median(r["p90_ms"] for r in v),
                                           tokens_per_s=conc * a.steps / statistics.median(r["makespan_ms"] for r in v) * 1e3)
                                   for k, v in runs.items()}
                # Stage spans (CUDA events) and profiler kernel time, one instrumented run each.
                case["stages"], case["profile"] = {}, {}
                for k, (grp, _) in strategies.items():
                    ev = []
                    torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    serve(model, grp, fns[k], events=ev)
                    torch.cuda.synchronize()
                    wall = (time.perf_counter_ns() - t0) / 1e6
                    case["stages"][k] = dict(wall_ms=wall, spans=stage_spans(ev))
                    try:
                        case["profile"][k] = kernel_time(model, grp, fns[k])
                    except Exception as e:  # noqa: BLE001
                        case["profile"][k] = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
                print(json.dumps(dict(concurrency=conc, summary=case["summary"])), flush=True)
            except Exception:  # noqa: BLE001 - the failure is the result
                case["error"] = traceback.format_exc()[-3000:]
            rec["cases"].append(case)
            m4.write(out / "decompose.json", rec)
        # Recurrence microbench and row-independence checks.
        cfgs = {} if a.no_micro else {"policy_b32": pick(a.group), "default": pick(4096)}
        if a.width <= 512 and not a.no_micro:
            for s in (16, 32, 64):
                if a.width // s <= 16:
                    cfgs[f"tl_cluster_s{s}"] = KernelConfig(family="tilelang", slice_width=s, sync="cluster")
        for s in (16, 32) if not a.no_micro else ():
            cfgs[f"tl_grid_s{s}"] = KernelConfig(family="tilelang", slice_width=s, sync="grid")
        if not a.no_micro:
            cfgs["triton_s32_w4"] = KernelConfig(family="mma_rtiled", warps=4, slice_width=32, sync="grid")
        rec["micro_configs"] = {k: v.as_dict() for k, v in cfgs.items()}
        try:
            rec["microbench"] = microbench(a.cell, a.width, a.steps, cfgs, (32, 128, 512, 1024)) if cfgs else []
        except Exception:  # noqa: BLE001
            rec["microbench_error"] = traceback.format_exc()[-3000:]
        try:
            rec["row_independence"] = check_rows(a.cell, a.width, 256, a.steps, cfgs, 99) if cfgs else {}
        except Exception:  # noqa: BLE001
            rec["row_independence_error"] = traceback.format_exc()[-3000:]
    rec["finished"] = time.time()
    m4.write(out / "decompose.json", rec)
    (out / "DONE.json").write_text(json.dumps(dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))))
    print("HC_DONE", flush=True)


if __name__ == "__main__":
    main()
