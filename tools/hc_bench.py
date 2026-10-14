"""High-concurrency (HC) serving of the FlashRNN2 classifier: decomposition, selection and paired timing.

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

Modes
  decompose  the round-0 measurements above (split-0 fixtures)
  select     one case (cell, D, C, T): qualify every candidate strategy (FP64 oracle on the first 32
             requests, logits vs the reference candidate for all requests, MODEL_BUDGET), then interleaved
             makespan repetitions on split-0 fixtures; writes select.json
  start      one process start of one frozen comparison on split-1 fixtures: qualification, 3 warm-up
             bursts per arm, then ABBA/BAAB blocks of whole-burst makespans (rtiled_m4_analyze format)
  profile    run one configuration's recurrence a few times (for ncu with --launch-skip/--launch-count)
  forward    run the s0p forward of one configuration a few times (for nsys)
A strategy is {"strategy": "s0" | "s0p", "group": rows per call, "config": KernelConfig fields} and may set
"graph": true, which serves each group through a CUDA Graph of the whole 4-layer forward (IDs copied into a
static device buffer, replay, logits to host).
"""
import argparse
import json
import os
import statistics
import sys
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
                                               output_mode="hidden", sync=cfg.sync, batch_tile=cfg.batch_tile,
                                               threads=32 * cfg.warps)
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
    groups = (batch + cfg.batch_tile - 1) // cfg.batch_tile * heads
    if cfg.family == "mma_rtiled":
        splits, groups, resident = triton_rtiled.plan(batch, heads, width, cfg.slice_width, sms=SMS)
        return dict(k=splits, groups=groups, resident_groups=resident, ctas=splits * resident)
    if cfg.sync == "grid":
        from flashrnn.flashrnn2 import tilelang_cluster
        k, resident, slots = tilelang_cluster.grid_plan(batch, heads, width, cfg.slice_width, SMS, cfg.batch_tile)
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


class GraphForward:
    """The whole classifier forward for one (rows, steps) captured once and replayed per group."""

    def __init__(self, model, rows, steps, fn):
        self.model, self.fn = model, fn
        self.ids = torch.zeros((rows, steps), dtype=torch.int64, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.forward()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self.forward()
        torch.cuda.synchronize()

    def forward(self):
        model = self.model
        x = model.embedding[self.ids]
        for w, r, b in model.weights:
            wx = torch.einsum("bti,ghdi->btghd", x, w)
            hidden = self.fn(wx, r, b, model.initial(self.ids.shape[0], model.dtype))
            x = hidden[:, :, 0]
        return x[:, -1] @ model.readout

    def __call__(self, ids):
        self.ids.copy_(ids)
        self.graph.replay()
        return self.logits.cpu()


_GRAPHS = {}


def graph_arm(model, spec, rows, steps, fn):
    key = (id(model), json.dumps(spec, sort_keys=True), rows, steps)
    if key not in _GRAPHS:
        _GRAPHS[key] = GraphForward(model, rows, steps, fn)
    return _GRAPHS[key]


def ids_for(conc, steps, split):
    g = torch.Generator().manual_seed((20261015 if split == 0 else 20261115) + conc)
    return torch.randint(0, m4.VOCAB, (conc, steps), generator=g)


def strategy_arm(cell, spec):
    return arm(cell, KernelConfig(**spec["config"]))


def burst(model, ids, spec, fn):
    groups = list(ids.split(spec["group"]))
    if spec.get("graph"):
        graphs = [graph_arm(model, spec, len(g), g.shape[1], fn) for g in groups]  # built outside the timed span
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        outs, done = [], []
        for g, gf in zip(groups, graphs):
            outs.append(gf(g))
            done.append((time.perf_counter_ns() - t0) / 1e6)
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        outs, done = serve(model, groups, fn)
    makespan = (time.perf_counter_ns() - t0) / 1e6
    per_request = [d for d, grp in zip(done, ids.split(spec["group"])) for _ in range(len(grp))]
    return torch.cat(outs), dict(makespan_ms=makespan, p50_ms=statistics.median(per_request),
                                 p90_ms=sorted(per_request)[int(0.9 * (len(per_request) - 1))])


def qualify_arms(model, ids, specs, cell):
    """Every arm vs the FP64 oracle (first 32 requests) and vs the first arm (all requests)."""
    o_logits, _ = model.oracle(ids[:32])
    outs, rep_ = {}, {}
    for name, spec in specs.items():
        logits, _ = burst(model, ids, spec, strategy_arm(cell, spec))
        outs[name] = logits
        rep_[name] = dict(vs_oracle=m4.compare(logits[:32].cuda(), o_logits))
    first = next(iter(specs))
    for name in specs:
        rep_[name]["vs_first"] = m4.compare(outs[name].cuda(), outs[first].cuda())
        rep_[name]["pass"] = rep_[name]["vs_oracle"]["pass_"] and rep_[name]["vs_first"]["pass_"]
    return rep_, outs


def mode_select(a):
    case = json.loads(a.case)
    cell, width, steps, conc = case["cell"], case["width"], case["steps"], case["concurrency"]
    rec = dict(mode="select", case=case, identity=runtime_identity(), started=time.time())
    model = m4.Model(cell, width)
    ids = ids_for(conc, steps, 0)
    specs = {c["label"]: c for c in case["candidates"]}
    with torch.inference_mode():
        quals, errors = {}, {}
        for name, spec in specs.items():
            try:
                q, _ = qualify_arms(model, ids, {case["reference"]: specs[case["reference"]], name: spec}, cell)
                quals[name] = q[name]
            except Exception as e:  # noqa: BLE001 - an unsupported or failing candidate is a result
                errors[name] = f"{type(e).__name__}: {str(e)[:400]}"
        rec.update(qualification=quals, errors=errors)
        ok = [n for n in specs if n in quals and quals[n]["pass"]]
        # A candidate with a "twin" (same arithmetic, different CTA size) must reproduce it bit for bit.
        twins = {}
        for n in ok:
            t = specs[n].get("twin")
            if t in ok:
                a_, _ = burst(model, ids, specs[n], strategy_arm(cell, specs[n]))
                b_, _ = burst(model, ids, specs[t], strategy_arm(cell, specs[t]))
                twins[n] = dict(twin=t, bitwise=bool(torch.equal(a_.view(torch.int16), b_.view(torch.int16))))
        rec["twin_bitwise"] = twins
        fns = {n: strategy_arm(cell, specs[n]) for n in ok}
        for n in ok:
            burst(model, ids, specs[n], fns[n])
        runs = {n: [] for n in ok}
        for rep_i in range(a.reps):
            order = ok[rep_i % len(ok):] + ok[:rep_i % len(ok)]
            for n in order:
                runs[n].append(burst(model, ids, specs[n], fns[n])[1])
        med = {n: dict(makespan_ms=statistics.median(r["makespan_ms"] for r in v),
                       p90_ms=statistics.median(r["p90_ms"] for r in v)) for n, v in runs.items()}
        rec["screen"] = dict(runs=runs, median=med)
        roles = {}
        for role, labels in case["roles"].items():
            pool = [n for n in labels if n in med]
            best = min(pool, key=lambda n: med[n]["makespan_ms"]) if pool else None
            roles[role] = dict(pool=labels, qualified=pool, selected=specs[best] if best else None, label=best)
        rec["roles"] = roles
    rec["finished"] = time.time()
    m4.write(Path(a.out) / "select.json", rec)
    print(json.dumps(dict(case=case["name"], **{r: v["label"] for r, v in roles.items()},
                          median={n: round(v["makespan_ms"], 3) for n, v in med.items()})), flush=True)
    return 0


def mode_start(a):
    table_path = Path(a.table)
    assert m4.sha(table_path) == a.table_sha256, "frozen table changed"
    table = json.loads(table_path.read_text())
    comp = next(c for c in table["comparisons"] if c["name"] == a.comparison)
    cell, width, steps, conc = comp["cell"], comp["width"], comp["steps"], comp["concurrency"]
    out = Path(a.out)
    meta = dict(comparison=comp, start=a.start, split=1, identity=runtime_identity(), table_sha256=a.table_sha256,
                pid=os.getpid(), host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
                gpu_uuid=os.environ.get("FRNN_GPU_UUID"), cpu_affinity=sorted(os.sched_getaffinity(0)),
                started=time.time(), blocks=a.blocks, order="ABBA even blocks, BAAB odd blocks",
                arm_unit="one burst: C requests (IDs on host) to logits on host", budget=m4.MODEL_BUDGET)
    m4.write(out / "meta.json", meta)
    model = m4.Model(cell, width)
    ids = ids_for(conc, steps, 1)
    specs = {"baseline": comp["baseline"], "candidate": comp["candidate"]}
    with torch.inference_mode():
        report, outs = qualify_arms(model, ids, specs, cell)
        m4.write(out / "qualification.json", report)
        if not all(v["pass"] for v in report.values()):
            meta.update(status="NUMERICAL_FAIL_NO_TIMING", finished=time.time())
            m4.write(out / "meta.json", meta)
            return 3
        golden = outs["baseline"].cuda()
        meta["candidate_bitwise_equal_to_baseline"] = bool(torch.equal(outs["candidate"].view(torch.int16),
                                                                       outs["baseline"].view(torch.int16)))
        fns = {k: strategy_arm(cell, v) for k, v in specs.items()}
        for _ in range(3):
            for k in specs:
                burst(model, ids, specs[k], fns[k])
        blocks = []
        for i in range(a.blocks):
            order = "ABBA" if i % 2 == 0 else "BAAB"
            raw = {"A": [], "B": []}
            checked = True
            for arm_ in order:
                k = "baseline" if arm_ == "A" else "candidate"
                logits, t = burst(model, ids, specs[k], fns[k])
                checked &= m4.compare(logits.cuda(), golden)["pass_"]
                raw[arm_].append(t)
            blocks.append(dict(block=i, order=order, a_ms=sum(t["makespan_ms"] for t in raw["A"]) / 2,
                               b_ms=sum(t["makespan_ms"] for t in raw["B"]) / 2,
                               a_raw=[t["makespan_ms"] for t in raw["A"]], b_raw=[t["makespan_ms"] for t in raw["B"]],
                               a_p90=[t["p90_ms"] for t in raw["A"]], b_p90=[t["p90_ms"] for t in raw["B"]],
                               a_p50=[t["p50_ms"] for t in raw["A"]], b_p50=[t["p50_ms"] for t in raw["B"]],
                               outputs_within_budget=checked))
            if not checked:
                break
    status = "PASS" if len(blocks) == a.blocks and all(b["outputs_within_budget"] for b in blocks) else "FAIL"
    m4.write(out / "timing.json", dict(blocks=blocks, status=status))
    meta.update(status=status, finished=time.time())
    m4.write(out / "meta.json", meta)
    print(json.dumps(dict(comparison=a.comparison, start=a.start, status=status,
                          ratio=sum(b["a_ms"] for b in blocks) / max(1e-9, sum(b["b_ms"] for b in blocks)))), flush=True)
    return 0 if status == "PASS" else 4


def mode_profile(a):
    """Recurrence only: warm-up calls, then the calls a profiler selects by launch index."""
    cfg = KernelConfig(**json.loads(a.config))
    model = m4.Model(a.cell, a.width)
    w, r, b = model.weights[0]
    g = torch.Generator().manual_seed(11)
    wx = (torch.randn(a.batch, a.steps, 4, 1, a.width, generator=g) * 0.5).to("cuda", torch.float16)
    init = model.initial(a.batch, torch.float16)
    fn = arm(a.cell, cfg)
    with torch.inference_mode():
        for _ in range(a.calls):
            fn(wx, r, b, init)
        torch.cuda.synchronize()
    print("HC_PROFILE_DONE", json.dumps(dict(config=cfg.as_dict(), batch=a.batch, steps=a.steps, calls=a.calls)),
          flush=True)
    return 0


def mode_forward(a):
    """The s0p forward (all C requests per layer) of one configuration, repeated for a timeline."""
    cfg = KernelConfig(**json.loads(a.config))
    model = m4.Model(a.cell, a.width)
    ids = ids_for(a.concurrency, a.steps, 0)
    fn = arm(a.cell, cfg)
    with torch.inference_mode():
        for _ in range(a.calls):
            serve(model, [ids], fn)
        torch.cuda.synchronize()
    print("HC_FORWARD_DONE", flush=True)
    return 0


def main():
    global SMS
    SMS = torch.cuda.get_device_properties(0).multi_processor_count
    torch.backends.cuda.matmul.allow_tf32 = False
    if len(sys.argv) > 1 and sys.argv[1] in ("profile", "forward"):
        mode = sys.argv.pop(1)
        p = argparse.ArgumentParser()
        p.add_argument("--out", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--cell", required=True)
        p.add_argument("--width", type=int, required=True)
        p.add_argument("--steps", type=int, default=128)
        p.add_argument("--calls", type=int, default=5)
        p.add_argument("--batch", type=int, default=1024)
        p.add_argument("--concurrency", type=int, default=1024)
        a = p.parse_args()
        return mode_profile(a) if mode == "profile" else mode_forward(a)
    if len(sys.argv) > 1 and sys.argv[1] in ("select", "start"):
        mode = sys.argv.pop(1)
        p = argparse.ArgumentParser()
        p.add_argument("--out", required=True)
        if mode == "select":
            p.add_argument("--case", required=True)
            p.add_argument("--reps", type=int, default=5)
            return mode_select(p.parse_args())
        p.add_argument("--table", required=True)
        p.add_argument("--table-sha256", required=True)
        p.add_argument("--comparison", required=True)
        p.add_argument("--start", type=int, required=True)
        p.add_argument("--blocks", type=int, default=20)
        return mode_start(p.parse_args())
    if len(sys.argv) > 1 and sys.argv[1] == "decompose":
        sys.argv.pop(1)
    return mode_decompose()


def mode_decompose():
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
    sys.exit(main())
