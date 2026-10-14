"""M4 paired timing for R tiling: E2 (D<=128 switch point), E3 (vs pinned FlashRNN1 CUDA fused),
E4 (Triton vs TileLang). One process per invocation.

Workload: the existing external-baseline classifier (4 layers, vocab 512, FP16;
C requests served in B-sized groups of T tokens; IDs transfer, embedding, every
input projection and recurrence, readout and host delivery) with only the
width D changed. Every arm is checked against an FP64 model oracle and against
the other arm under the original model budget before timing.

Modes
  v1check  build/run pinned FlashRNN1 cuda_fused for (cell, D, B); record its
           ConstrINT tiles from the build manifest; cell-level FP64 check
  select   split-0 fixtures: qualify and screen candidate configurations
  start    split-1 fixtures: one process start of one frozen comparison,
           20 ABBA/BAAB blocks
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import triton

from flashrnn.flashrnn2 import reference, triton_persistent, triton_rtiled
from flashrnn.flashrnn2.execution_policy import ExecutionPolicy, KernelConfig, shape_key
from flashrnn.flashrnn2.state_contract import begin_sequence

MODEL_BUDGET = dict(atol=0.015, rtol=0.03, max_abs=0.03)
LAYERS, VOCAB = 4, 512
V1_BASE = Path("/gpfs/scratch/ehpc390/tueb043681/paper-baselines-20261013-v1")
SOURCES = ["tools/rtiled_m4.py", "flashrnn/flashrnn2/triton_rtiled.py", "flashrnn/flashrnn2/triton_persistent.py",
           "flashrnn/flashrnn2/tilelang_rtiled.py", "flashrnn/flashrnn2/reference.py",
           "flashrnn/flashrnn2/execution_policy.py", "flashrnn/flashrnn2/state_contract.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def identity():
    ident = dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                 triton=triton.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                 source={f: sha(f) for f in SOURCES})
    if os.environ.get("FRNN_TILELANG") == "1":
        import tilelang
        ident["tilelang"] = tilelang.__version__
    return ident


# ---------------------------------------------------------------- model

class Model:
    """Same initialization as tools/flashrnn2/stacked_e2e.StackedModel, any width."""

    def __init__(self, cell, width, dtype=torch.float16):
        g = torch.Generator().manual_seed(20261006)
        self.cell, self.width, self.dtype = cell, width, dtype
        self.embedding = (torch.randn(VOCAB, width, generator=g) * 0.1).to("cuda", dtype)
        self.readout = (torch.randn(width, VOCAB, generator=g) * 0.1).to("cuda", dtype)
        self.weights = []
        for _ in range(LAYERS):
            w = (torch.randn(4, 1, width, width, generator=g) * (0.2 / width ** 0.5)).to("cuda", dtype)
            r = (torch.randn(4, 1, width, width, generator=g) * (0.2 / width ** 0.5)).to("cuda", dtype)
            b = (torch.randn(4, 1, width, generator=g) * 0.02).to("cuda", dtype)
            self.weights.append((w, r, b))
        self.states = 4 if cell == "slstm" else 2

    def initial(self, batch, dtype):
        init = torch.zeros((self.states, batch, 1, 1, self.width), device="cuda", dtype=dtype)
        if self.cell == "slstm":
            init[2].fill_(1)
        return init

    def forward(self, ids, arm, trace=False):
        x = self.embedding[ids.to("cuda")]
        layers = []
        for w, r, b in self.weights:
            wx = torch.einsum("bti,ghdi->btghd", x, w)
            hidden = arm(wx, r, b, self.initial(ids.shape[0], self.dtype))
            x = hidden[:, :, 0]
            if trace:
                layers.append(hidden[:, :, 0])
        return x[:, -1] @ self.readout, layers

    def oracle(self, ids):
        """FP64 everywhere except the recurrent hidden operand, rounded to the input dtype."""
        x = self.embedding.double()[ids.to("cuda")]
        layers = []
        for w, r, b in self.weights:
            wx = torch.einsum("bti,ghdi->btghd", x, w.double())
            history, _ = reference.recurrence(wx, r.double(), b.double(), self.initial(ids.shape[0], torch.float64),
                                              self.cell, self.dtype)
            x = history[0][:, :, 0]
            layers.append(x)
        return x[:, -1] @ self.readout.double(), layers


def fixtures(batch, concurrency, steps, split):
    g = torch.Generator().manual_seed(20261006 + batch + 1000 * split)
    return [torch.randint(0, VOCAB, (batch, steps), generator=g) for _ in range(concurrency // batch)]


# ---------------------------------------------------------------- arms

def arm_single(cell, warps):
    def run(wx, r, b, init):
        ctx = begin_sequence(init, cell, wx.dtype)
        y, _ = triton_persistent.recurrence(wx, r, b, init, cell, return_compute_carry=True, sequence_context=ctx,
                                            output_mode="hidden", num_warps=warps)
        return y
    return run


def arm_rtiled(cell, cfg):
    def run(wx, r, b, init):
        ctx = begin_sequence(init, cell, wx.dtype)
        y, _ = triton_rtiled.recurrence(wx, r, b, init, cell, sequence_context=ctx, slice_width=cfg.slice_width,
                                        num_warps=cfg.warps, pad_k=cfg.pad_k, output_mode="hidden")
        return y
    return run


def arm_tilelang(cell, slice_width):
    from flashrnn.flashrnn2 import tilelang_rtiled

    def run(wx, r, b, init):
        y, _ = tilelang_rtiled.recurrence(wx, r, b, init, cell, slice_width=slice_width)
        return y
    return run


_V1 = {}


def v1_module():
    if "flashrnn" in _V1:
        return _V1
    manifest = json.loads((V1_BASE / "pinned-source-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha(V1_BASE / name) == digest, name
    sys.path.insert(0, str(V1_BASE / "source/tools/flashrnn2"))
    from build_without_ninja import TaskBuilder  # noqa: E402

    spec = importlib.util.spec_from_file_location("frnn1_pinned", V1_BASE / "pinned/flashrnn/__init__.py",
                                                  submodule_search_locations=[str(V1_BASE / "pinned/flashrnn")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from frnn1_pinned.flashrnn import cuda_init
    from frnn1_pinned.flashrnn.flashrnn import FlashRNNConfig, flashrnn

    class H100Builder(TaskBuilder):
        def load(self, *positional, **keywords):
            # Same transport as the external-baseline cohort: native sm_90 SASS instead of compute_80 PTX.
            flags = [f.replace("arch=compute_80,code=compute_80", "arch=compute_90,code=sm_90")
                     for f in keywords["extra_cuda_cflags"]]
            keywords["extra_cuda_cflags"] = flags
            return super().load(*positional, **keywords)

    cuda_init._load = H100Builder(Path(os.environ["FRNN_V1_BUILD_ROOT"]), V1_BASE / "pinned/flashrnn/flashrnn").load
    dirs = [Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    os.environ["CUDA_LIB"] = str(next(p for p in dirs if (p / "libcublas.so").is_file()))
    _V1.update(flashrnn=flashrnn, config=FlashRNNConfig, commit=manifest["commit"])
    return _V1


def arm_v1(cell, batch, width):
    v1 = v1_module()
    config = v1["config"](function=cell, backend="cuda_fused", dtype="float16", batch_size=batch,
                          hidden_dim=width, num_heads=1)
    assert config.backend == "cuda_fused"

    def run(wx, r, b, init):
        history, _ = v1["flashrnn"](wx, r, b, init, config=config)
        return history[0]
    return run


def make_arm(spec, cell, batch, width):
    kind = spec["kind"]
    if kind == "v1_cuda_fused":
        return arm_v1(cell, batch, width)
    if kind == "single":
        return arm_single(cell, spec["warps"])
    if kind == "rtiled":
        return arm_rtiled(cell, KernelConfig(family="mma_rtiled", warps=spec["warps"],
                                             slice_width=spec["slice_width"], sync="grid",
                                             pad_k=spec.get("pad_k", False)))
    if kind == "tilelang":
        return arm_tilelang(cell, spec["slice_width"])
    raise ValueError(kind)


# ---------------------------------------------------------------- checks and timing

def compare(actual, expected):
    a, e = actual.double(), expected.double()
    diff = (a - e).abs()
    limit = MODEL_BUDGET["atol"] + MODEL_BUDGET["rtol"] * e.abs()
    finite = bool(torch.isfinite(a).all())
    return dict(pass_=finite and bool((diff <= limit).all()) and float(diff.max()) <= MODEL_BUDGET["max_abs"],
                max_abs=float(diff.max()), worst_normalized=float((diff / limit).max()),
                failed_elements=int(((diff > limit) | (diff > MODEL_BUDGET["max_abs"])).sum()))


def qualify(model, groups, arms):
    """Each arm vs FP64 oracle (logits, every layer's hidden history); arms vs each other."""
    report, goldens = {"groups": []}, None
    ok = True
    outs = {}
    for gi, ids in enumerate(groups):
        o_logits, o_layers = model.oracle(ids)
        row = {}
        per_arm = {}
        for name, fn in arms.items():
            logits, layers = model.forward(ids, fn, trace=True)
            per_arm[name] = (logits, layers)
            checks = dict(logits=compare(logits, o_logits),
                          **{f"layer{i}": compare(h, oh) for i, (h, oh) in enumerate(zip(layers, o_layers))})
            row[f"{name}_vs_oracle"] = checks
            ok &= all(c["pass_"] for c in checks.values())
            outs.setdefault(name, []).append(dict(logits=logits.float().cpu(),
                                                  last_hidden=[h[:, -1].float().cpu() for h in layers]))
        names = list(arms)
        if len(names) == 2:
            (la, ya), (lb, yb) = per_arm[names[0]], per_arm[names[1]]
            cross = dict(logits=compare(lb, la), **{f"layer{i}": compare(b, a) for i, (a, b) in enumerate(zip(ya, yb))})
            row["candidate_vs_baseline"] = cross
            ok &= all(c["pass_"] for c in cross.values())
        outs.setdefault("oracle", []).append(dict(logits=o_logits.float().cpu(),
                                                  last_hidden=[h[:, -1].float().cpu() for h in o_layers]))
        report["groups"].append(row)
    report["status"] = "PASS" if ok else "NUMERICAL_FAIL"
    return report, outs


def burst(model, groups, arm):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    outputs = [model.forward(groups[i], arm)[0].cpu() for i in range(len(groups))]
    return outputs, (time.perf_counter_ns() - start) / 1e6


def golden_ok(outputs, goldens):
    for a, e in zip(outputs, goldens):
        if not compare(a, e)["pass_"]:
            return False
    return True


# ---------------------------------------------------------------- modes

def mode_v1check(args):
    """Build/run pinned v1 cuda_fused for (cell, D, B): record tiles; one-layer FP64 check (T=32)."""
    rec = dict(mode="v1check", cell=args.cell, width=args.width, batch=args.batch, identity=identity())
    t0 = time.time()
    try:
        root = Path(os.environ["FRNN_V1_BUILD_ROOT"])
        before = set(root.glob("*/build-manifest.json"))
        arm = arm_v1(args.cell, args.batch, args.width)
        g = torch.Generator().manual_seed(20261014 + args.width + args.batch)
        model = Model(args.cell, args.width)
        wx = (torch.randn(args.batch, 32, 4, 1, args.width, generator=g) * 0.1).to("cuda", torch.float16)
        _, r, b = model.weights[0]
        init = model.initial(args.batch, torch.float16)
        with torch.inference_mode():
            y = arm(wx, r, b, init)
            torch.cuda.synchronize()
            rec["seconds"] = time.time() - t0
            ref, _ = reference.recurrence(wx.double(), r.double(), b.double(), init.double(), args.cell,
                                          torch.float16)
        after = set(root.glob("*/build-manifest.json"))
        mine = []
        for path in after:
            m = json.loads(path.read_text())
            flags = " ".join(m.get("cuda_cflags", []))
            if f"-DFLASHRNN_HIDDEN_SIZE={args.width} " in flags + " " and \
                    f"-DFLASHRNN_BATCH_SIZE={args.batch} " in flags + " ":
                mine.append(m)
        rec["new_builds"] = len(after - before)
        rec["tiles"] = sorted({f for m in mine for f in m.get("cuda_cflags", []) if f.startswith("-DFLASHRNN_")})
        rec["artifacts"] = [dict(name=m.get("name"), artifact=m.get("artifact"), sha256=m.get("artifact_sha256"))
                            for m in mine]
        rec["one_layer_check"] = compare(y[:, :, 0], ref[0][:, :, 0])
        rec["status"] = "AVAILABLE" if rec["one_layer_check"]["pass_"] else "NUMERICAL_FAIL"
    except Exception:  # noqa: BLE001 - an unavailable baseline is a recorded result
        rec.update(status="BASELINE_UNAVAILABLE", error=traceback.format_exc()[-3000:], seconds=time.time() - t0)
    write(Path(args.out) / "v1check.json", rec)
    print(json.dumps({k: rec.get(k) for k in ("cell", "width", "batch", "status", "seconds")}), flush=True)
    return 0


def screen(model, groups, arms, rounds=8):
    for fn in arms.values():
        for _ in range(2):
            burst(model, groups, fn)
    names = list(arms)
    times = {n: [] for n in names}
    for r in range(rounds):
        for n in names[r % len(names):] + names[:r % len(names)]:
            times[n].append(burst(model, groups, arms[n])[1])
    return {n: sorted(v)[len(v) // 2] for n, v in times.items()}, times


def mode_select(args):
    case = json.loads(args.case)
    cell, width, batch, conc, steps = case["cell"], case["width"], case["batch"], case["concurrency"], case["steps"]
    rec = dict(mode="select", case=case, identity=identity(), split=0, candidates={})
    model = Model(cell, width)
    groups = fixtures(batch, conc, steps, split=0)
    with torch.inference_mode():
        for role, specs in case["roles"].items():
            arms, qual = {}, {}
            for spec in specs:
                name = json.dumps(spec, sort_keys=True)
                try:
                    fn = make_arm(spec, cell, batch, width)
                    report, _ = qualify(model, groups, {"arm": fn})
                    qual[name] = report["status"]
                    if report["status"] == "PASS":
                        arms[name] = fn
                except Exception as e:  # noqa: BLE001
                    qual[name] = f"ERROR: {type(e).__name__}: {str(e)[:300]}"
            medians, raw = screen(model, groups, arms) if arms else ({}, {})
            best = min(medians, key=medians.get) if medians else None
            rec["candidates"][role] = dict(qualification=qual, median_ms=medians, raw_ms=raw,
                                           selected=json.loads(best) if best else None)
            print(json.dumps(dict(case=case["name"], role=role, selected=best,
                                  median=medians.get(best) if best else None)), flush=True)
    rec["status"] = "SELECTED" if all(v["selected"] for v in rec["candidates"].values()) else "INCOMPLETE"
    write(Path(args.out) / "select.json", rec)
    return 0


def mode_start(args):
    table_path = Path(args.table)
    table = json.loads(table_path.read_text())
    assert sha(table_path) == args.table_sha256, "frozen table changed"
    comp = next(c for c in table["comparisons"] if c["name"] == args.comparison)
    cell, width, batch, conc, steps = (comp[k] for k in ("cell", "width", "batch", "concurrency", "steps"))
    ident = identity()
    assert all(ident[k] == table["identity"][k] for k in ("gpu", "triton", "torch")), "identity mismatch"
    if comp["candidate"]["kind"] == "rtiled" and comp["experiment"] == "E3":
        # Exercise the frozen exact-shape table path of the public ExecutionPolicy.
        policy = ExecutionPolicy(table["policy_table"], identity=table["policy_identity"])
        cfg = policy.select(cell, "float16", batch, steps, 1, width, identity=table["policy_identity"])
        assert (cfg.slice_width, cfg.warps, cfg.pad_k) == (comp["candidate"]["slice_width"], comp["candidate"]["warps"],
                                                           comp["candidate"].get("pad_k", False))
    out = Path(args.out)
    meta = dict(comparison=comp, start=args.start, split=1, identity=ident, table_sha256=args.table_sha256,
                pid=os.getpid(), host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
                gpu_uuid=os.environ.get("FRNN_GPU_UUID"), cpu_affinity=sorted(os.sched_getaffinity(0)),
                started=time.time(), blocks=args.blocks, order="ABBA even blocks, BAAB odd blocks",
                arm_unit="one burst: all C/B groups through the model, logits delivered to host",
                budget=MODEL_BUDGET)
    write(out / "meta.json", meta)
    model = Model(cell, width)
    groups = fixtures(batch, conc, steps, split=1)
    arms = {"baseline": make_arm(comp["baseline"], cell, batch, width),
            "candidate": make_arm(comp["candidate"], cell, batch, width)}
    with torch.inference_mode():
        report, outs = qualify(model, groups, arms)
        torch.save(outs, out / "qualification.pt")
        write(out / "qualification.json", report)
        meta["qualification"] = report["status"]
        if report["status"] != "PASS":
            meta.update(status="NUMERICAL_FAIL_NO_TIMING", finished=time.time())
            write(out / "meta.json", meta)
            return 3
        goldens = [o for o in outs["baseline"]]
        goldens = [g["logits"] for g in goldens]
        for _ in range(3):
            for fn in arms.values():
                burst(model, groups, fn)
        blocks = []
        for i in range(args.blocks):
            order = "ABBA" if i % 2 == 0 else "BAAB"
            times = {"A": [], "B": []}
            checked = True
            for arm in order:
                outputs, ms = burst(model, groups, arms["baseline" if arm == "A" else "candidate"])
                checked &= golden_ok(outputs, goldens)
                times[arm].append(ms)
            blocks.append(dict(block=i, order=order, a_ms=sum(times["A"]) / 2, b_ms=sum(times["B"]) / 2,
                               a_raw=times["A"], b_raw=times["B"], outputs_within_budget=checked))
            if not checked:
                break
    status = "PASS" if len(blocks) == args.blocks and all(b["outputs_within_budget"] for b in blocks) else "FAIL"
    write(out / "timing.json", dict(blocks=blocks, status=status))
    meta.update(status=status, finished=time.time())
    write(out / "meta.json", meta)
    print(json.dumps(dict(comparison=args.comparison, start=args.start, status=status,
                          ratio=sum(b["a_ms"] for b in blocks) / max(1e-9, sum(b["b_ms"] for b in blocks)))),
          flush=True)
    return 0 if status == "PASS" else 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("v1check", "select", "start"))
    p.add_argument("--out", required=True)
    p.add_argument("--cell")
    p.add_argument("--width", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--case")
    p.add_argument("--table")
    p.add_argument("--table-sha256")
    p.add_argument("--comparison")
    p.add_argument("--start", type=int)
    p.add_argument("--blocks", type=int, default=20)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    Path(args.out).mkdir(parents=True, exist_ok=True)
    code = {"v1check": mode_v1check, "select": mode_select, "start": mode_start}[args.mode](args)
    write(Path(args.out) / "DONE.json", dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), exit=code))
    return code


if __name__ == "__main__":
    sys.exit(main())
