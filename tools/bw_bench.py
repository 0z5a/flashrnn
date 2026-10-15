"""Paired timing of one training step (forward + backward) per call: FlashRNN2 training path vs pinned FlashRNN1.

Arms (case JSON "roles" lists of specs)
  {"kind": "frnn2", "forward": <KernelConfig dict or null = packaged table/default>, "slice_width": S, "warps": W}
  {"kind": "v1"}  pinned FlashRNN1 cuda_fused forward + backward (FRNN_V1_BASE / FRNN_V1_BUILD_ROOT)
One call = forward with requires_grad, loss <dH, h history>, backward (torch.autograd.grad, no accumulation).

Modes
  tselect  split 0: every spec of every role is qualified (gradients at a reduced shape B <= 16, T <= 64 against the
           FP64 oracle under tools/bw_qualify.py's gate) and the fastest qualified spec per role is selected
  tstart   split 1, one process start of one frozen comparison: both arms re-qualified, then 20 ABBA/BAAB blocks of
           CUDA-event time per call; gradients must stay bit-identical to the arm's first call in every block
Analysis: tools/rtiled_m4_analyze.py (same block format).
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import training
from flashrnn.flashrnn2.execution_policy import KernelConfig

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def identity():
    import triton
    return dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                capability=list(torch.cuda.get_device_capability()))


def values_for(case, split):
    shape = dict(case, init="warm")
    leaves, cot_h, _ = bw_qualify.inputs(shape, 20261200 + 1000 * split + case["width"] + case["batch"])
    return leaves, cot_h


def step_fn(spec, case):
    cell = case["cell"]
    if spec["kind"] == "frnn2":
        forward = KernelConfig(**spec["forward"]) if spec.get("forward") else None
        opts = dict(slice_width=spec["slice_width"], num_warps=spec["warps"])

        def run(leaves, cot_h):
            xs = [x.detach().requires_grad_(True) for x in leaves]
            y, _ = training.recurrence(*xs, cell, output_mode="hidden", config=forward, backward_options=opts)
            return torch.autograd.grad((y.float() * cot_h).sum(), xs)
        return run
    if spec["kind"] == "v1":
        import rtiled_m4
        v1 = rtiled_m4.v1_module()
        config = v1["config"](function=cell, backend="cuda_fused", dtype=case["dtype"], batch_size=case["batch"],
                              hidden_dim=case["width"], num_heads=1)

        def run(leaves, cot_h):
            xs = [x.detach().requires_grad_(True) for x in leaves]
            states, _ = v1["flashrnn"](*xs, config=config)
            return torch.autograd.grad((states[0].float() * cot_h).sum(), xs)
        return run
    raise ValueError(spec["kind"])


def qualify(spec, case):
    """Gradient gate at a reduced shape against the FP64 oracle (bw_qualify budget, no v1 term)."""
    small = dict(case, batch=min(case["batch"], 16), steps=min(case["steps"], 64), init="warm")
    leaves, cot_h, cot_f = bw_qualify.inputs(small, 31 + case["width"])
    grads = step_fn(spec, small)(leaves, cot_h)
    torch.cuda.synchronize()
    ref = bw_qualify.run_oracle(leaves, cot_h, cot_f, small, "hist")
    m = bw_qualify.metrics(list(grads), ref, case["cell"])
    checks = bw_qualify.gate(m, case["dtype"])
    return all(checks.values()), m


def event_ms(fn, leaves, cot_h, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        g = fn(leaves, cot_h)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / calls, g


def same(a, b, exact=True):
    """FlashRNN2 gradients must be bit-identical between calls; FlashRNN1's backward may accumulate in a varying
    order, so its gradients only have to stay within 1e-3 of their scale."""
    if exact:
        return all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b))
    return all(((x.float() - y.float()).abs().max() <= 1e-3 * y.float().abs().max().clamp_min(1e-30)).item()
               for x, y in zip(a, b))


def mode_tselect(args):
    case = json.loads(args.case)
    rec = dict(mode="tselect", case=case, identity=identity(), split=0, candidates={})
    leaves, cot_h = values_for(case, 0)
    for role, specs in case["roles"].items():
        fns, qual = {}, {}
        for spec in specs:
            name = json.dumps(spec, sort_keys=True)
            try:
                ok, m = qualify(spec, case)
                qual[name] = dict(status="PASS" if ok else "GRADIENT_FAIL", metrics=m)
                if ok:
                    fns[name] = step_fn(spec, case)
            except Exception as e:  # noqa: BLE001
                qual[name] = dict(status=f"ERROR: {type(e).__name__}: {str(e)[:300]}")
        med = {}
        if fns:
            for fn in fns.values():
                event_ms(fn, leaves, cot_h, 2)
            names, times = list(fns), {n: [] for n in fns}
            for r in range(6):
                for n in names[r % len(names):] + names[:r % len(names)]:
                    times[n].append(event_ms(fns[n], leaves, cot_h, 3)[0])
            med = {n: sorted(v)[len(v) // 2] for n, v in times.items()}
        best = min(med, key=med.get) if med else None
        rec["candidates"][role] = dict(qualification=qual, median_ms=med, selected=json.loads(best) if best else None)
        print(json.dumps(dict(case=case["name"], role=role, selected=best)), flush=True)
    rec["status"] = "SELECTED" if all(v["selected"] for v in rec["candidates"].values()) else "INCOMPLETE"
    write(Path(args.out) / "select.json", rec)
    return 0


def mode_tstart(args):
    table_path = Path(args.table)
    table = json.loads(table_path.read_text())
    assert sha(table_path) == args.table_sha256, "frozen table changed"
    comp = next(c for c in table["comparisons"] if c["name"] == args.comparison)
    out = Path(args.out)
    meta = dict(comparison=comp, start=args.start, split=1, identity=identity(), table_sha256=args.table_sha256,
                pid=os.getpid(), host=os.uname().nodename, gpu_uuid=os.environ.get("FRNN_GPU_UUID"),
                started=time.time(), blocks=args.blocks, order="ABBA even blocks, BAAB odd blocks",
                arm_unit="CUDA-event time per training step (forward + backward) over back-to-back calls")
    write(out / "meta.json", meta)
    leaves, cot_h = values_for(comp, 1)
    quals = {}
    fns = {}
    for name in ("baseline", "candidate"):
        ok, m = qualify(comp[name], comp)
        quals[name] = dict(pass_=ok, metrics=m)
        fns[name] = step_fn(comp[name], comp)
    write(out / "qualification.json", quals)
    if not all(q["pass_"] for q in quals.values()):
        meta.update(status="GRADIENT_FAIL_NO_TIMING", finished=time.time())
        write(out / "meta.json", meta)
        return 3
    firsts = {n: fn(leaves, cot_h) for n, fn in fns.items()}
    torch.cuda.synchronize()
    for fn in fns.values():
        event_ms(fn, leaves, cot_h, 2)
    trial = max(event_ms(fns["baseline"], leaves, cot_h, 1)[0], event_ms(fns["candidate"], leaves, cot_h, 1)[0])
    calls = max(2, min(20, int(40.0 / max(trial, 0.05))))
    blocks = []
    for i in range(args.blocks):
        order = "ABBA" if i % 2 == 0 else "BAAB"
        times = {"A": [], "B": []}
        stable = True
        for arm in order:
            name = "baseline" if arm == "A" else "candidate"
            ms, g = event_ms(fns[name], leaves, cot_h, calls)
            stable &= same(g, firsts[name], comp[name]["kind"] == "frnn2")
            times[arm].append(ms)
        blocks.append(dict(block=i, order=order, calls=calls, a_ms=sum(times["A"]) / 2, b_ms=sum(times["B"]) / 2,
                           a_raw=times["A"], b_raw=times["B"], outputs_within_budget=stable))
        if not stable:
            break
    status = "PASS" if len(blocks) == args.blocks and all(b["outputs_within_budget"] for b in blocks) else "FAIL"
    write(out / "timing.json", dict(blocks=blocks, status=status,
                                    note=("outputs_within_budget = FlashRNN2 gradients bit-identical to the arm's "
                                          "first call; FlashRNN1 gradients within 1e-3 of scale")))
    meta.update(status=status, finished=time.time(), calls_per_measurement=calls)
    write(out / "meta.json", meta)
    print(json.dumps(dict(comparison=args.comparison, start=args.start, status=status,
                          ratio=sum(b["a_ms"] for b in blocks) / max(1e-9, sum(b["b_ms"] for b in blocks)))), flush=True)
    return 0 if status == "PASS" else 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("tselect", "tstart"))
    p.add_argument("--out", required=True)
    p.add_argument("--case")
    p.add_argument("--table")
    p.add_argument("--table-sha256")
    p.add_argument("--comparison")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--blocks", type=int, default=20)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    {"tselect": mode_tselect, "tstart": mode_tstart}[a.mode](a)


if __name__ == "__main__":
    main()
