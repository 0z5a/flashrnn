"""Timing of several heads in one launch (H > 1) against per-head loops and FlashRNN1 (docs/flashrnn2/heads.md).

Arms (case JSON "arms")
  one         FlashRNN2, all H heads in one launch; the configuration the library selects for the shape
              (forward: dispatch.select_config; training: training.training_forward), recorded per case
  loop        H calls of H = 1, one per head, same configuration as "one" (plain Python loop)
  loop_graph  the same H calls captured into one CUDA graph (forward only): separates the effect of one launch from
              host launch overhead
  v1          pinned FlashRNN1 cuda_fused with num_heads = H, hidden_dim = H * D (FRNN_V1_BASE / FRNN_V1_BUILD_ROOT);
              a failure to build or run is recorded as the arm's result
  v1cuda      the same pinned FlashRNN1 with backend "cuda" (alternating: one recurrent GEMM and one pointwise kernel
              per step, any D)
A case may pin the FlashRNN2 configuration ("config": KernelConfig dict) instead of the library selection.
Kinds: "forward" (no grad, hidden output) and "train" (forward + backward of <cot_h, h history>).

Modes
  sweep   --case JSON --out FILE                 median CUDA-event ms per call of every arm (2 warm-up calls)
  paired  --case JSON --out FILE --start N        ABBA/BAAB blocks of "one" against every other arm; each block entry
          [--blocks 20] [--calls 3]              is the mean of --calls calls; the outputs of every timed call of an
                                                 arm must equal that arm's first call (bitwise)
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, training
from flashrnn.flashrnn2.execution_policy import KernelConfig

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def memory_estimate(case):
    """Bytes the largest arm needs (inputs, outputs, FP32 history and its gradient for training), rounded up."""
    b, t, h, d = case["batch"], case["steps"], case["heads"], case["width"]
    s = 4 if case["cell"] == "slstm" else 2
    e = 2
    inputs = b * t * 4 * h * d * e + 4 * h * d * d * e
    out = s * b * t * h * d * 4 + b * t * h * d * e
    if case["kind"] == "train":
        out += 2 * s * b * t * h * d * 4 + b * t * 4 * h * d * 4 + inputs
    return int(1.5 * (inputs + out))


def head_slice(x, dim, h):
    return x.narrow(dim, h, 1).contiguous()


def config_for(case):
    if case.get("config"):
        return KernelConfig(**case["config"]), "pinned"
    cell, dtype = case["cell"], DT[case["dtype"]]
    args = (cell, dtype, case["batch"], case["steps"], case["heads"], case["width"])
    if case["kind"] == "train":
        cfg, source = training.training_forward(*args)
    else:
        cfg, source = dispatch.select_config(*args)
    return cfg, source


def make_arm(name, case, leaves, cot_h, cfg):
    cell, kind, heads = case["cell"], case["kind"], case["heads"]

    def fwd(xs):
        with torch.no_grad():
            out, _ = dispatch.recurrence(*xs, cell, output_mode="hidden", config=cfg)
        return out

    def trn(xs, cot):
        xs = [x.detach().requires_grad_(True) for x in xs]
        y, _ = training.recurrence(*xs, cell, output_mode="hidden", config=cfg,
                                   backward_options=dict(case.get("backward") or {}))
        return torch.autograd.grad((y.float() * cot).sum(), xs)[0]

    if name == "one":
        return (lambda: fwd(leaves)) if kind == "forward" else (lambda: trn(leaves, cot_h))
    if name in ("loop", "loop_graph"):
        parts = [[head_slice(leaves[0], 3, h), head_slice(leaves[1], 1, h), head_slice(leaves[2], 1, h),
                  head_slice(leaves[3], 3, h)] for h in range(heads)]
        cots = [head_slice(cot_h, 2, h) for h in range(heads)]
        if kind == "forward":
            def run():
                return torch.cat([fwd(p) for p in parts], dim=2)
        else:
            def run():
                return torch.cat([trn(p, c) for p, c in zip(parts, cots)], dim=3)
        if name == "loop":
            return run
        if kind != "forward":
            raise ValueError("loop_graph is a forward-only arm")
        run(); torch.cuda.synchronize()  # compile outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static = run()

        def replay():
            graph.replay()
            return static
        return replay
    if name in ("v1", "v1cuda"):
        import rtiled_m4
        v1 = rtiled_m4.v1_module()
        config = v1["config"](function=cell, backend="cuda_fused" if name == "v1" else "cuda", dtype=case["dtype"],
                              batch_size=case["batch"], hidden_dim=heads * case["width"], num_heads=heads)
        if kind == "forward":
            def run():
                with torch.no_grad():
                    states, _ = v1["flashrnn"](*leaves, config=config)
                return states[0]
        else:
            def run():
                xs = [x.detach().requires_grad_(True) for x in leaves]
                states, _ = v1["flashrnn"](*xs, config=config)
                return torch.autograd.grad((states[0].float() * cot_h).sum(), xs)[0]
        return run
    raise ValueError(name)


def event_ms(fn, calls):
    ts, outs = [], []
    for _ in range(calls):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        out = fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
        outs.append(out)
    return ts, outs


def setup(case):
    torch.cuda.set_device(0)
    need = memory_estimate(case)
    free, _ = torch.cuda.mem_get_info()
    if need > 0.9 * free:
        return None, dict(status="SKIPPED_MEMORY", need_bytes=need, free_bytes=free)
    leaves, cot_h, _ = bw_qualify.inputs(dict(case, init="warm"), 777 + case["width"] + case["heads"])
    cfg, source = config_for(case)
    return (leaves, cot_h, cfg), dict(config=cfg.as_dict(), config_source=source, need_bytes=need, free_bytes=free)


def mode_sweep(args):
    case = json.loads(Path(args.case).read_text())
    rec = dict(case=case, started=time.time(), gpu=torch.cuda.get_device_name(), arms={})
    try:
        state, info = setup(case)
        rec.update(info)
        if state is not None:
            leaves, cot_h, cfg = state
            ref = None
            for name in case["arms"]:
                try:
                    fn = make_arm(name, case, leaves, cot_h, cfg)
                    fn(); fn(); torch.cuda.synchronize()
                    ts, outs = event_ms(fn, case.get("calls", 5))
                    entry = dict(ms=sorted(ts)[len(ts) // 2], all_ms=ts,
                                 repeat_bitwise=all(torch.equal(o, outs[0]) for o in outs))
                    rec["arms"][name] = entry
                    try:  # comparison of outputs is informative only; v1 returns its own layout
                        if name == "one":
                            ref = outs[0]
                        elif ref is not None and name in ("loop", "loop_graph"):
                            entry["bitwise_vs_one"] = torch.equal(outs[0], ref)
                        elif ref is not None:
                            entry["max_abs_vs_one"] = (outs[0].float().reshape(ref.shape) - ref.float()).abs().max().item()
                    except Exception as error:
                        entry["compare_error"] = repr(error)[-300:]
                    del fn, outs
                except Exception:
                    rec["arms"][name] = dict(error=traceback.format_exc()[-3000:])
                torch.cuda.empty_cache()
            rec["status"] = "DONE"
    except Exception:
        rec.update(status="ERROR", error=traceback.format_exc())
    rec["finished"] = time.time()
    write(args.out, rec)
    print(json.dumps(dict(status=rec.get("status"), arms={k: v.get("ms", "err") for k, v in rec["arms"].items()})))
    return 0


def mode_paired(args):
    case = json.loads(Path(args.case).read_text())
    rec = dict(case=case, start=args.start, started=time.time(), gpu=torch.cuda.get_device_name(), blocks=[])
    try:
        state, info = setup(case)
        rec.update(info)
        if state is None:
            rec["status"] = info["status"]
        else:
            leaves, cot_h, cfg = state
            fns, firsts, errors = {}, {}, {}
            for name in case["arms"]:
                try:
                    fns[name] = make_arm(name, case, leaves, cot_h, cfg)
                    firsts[name] = fns[name]()
                    torch.cuda.synchronize()
                except Exception:
                    errors[name] = traceback.format_exc()[-3000:]
            rec["arm_errors"] = errors
            if "one" not in fns:
                raise RuntimeError("the one-launch arm failed")
            stable = {n: True for n in fns}
            for block in range(args.blocks):
                row = {}
                for other in [n for n in fns if n != "one"]:
                    order = ["one", other, other, "one"] if block % 2 == 0 else [other, "one", "one", other]
                    t = {"one": [], other: []}
                    for n in order:
                        ts, outs = event_ms(fns[n], args.calls)
                        t[n].append(sum(ts) / len(ts))
                        stable[n] = stable[n] and all(torch.equal(o, firsts[n]) for o in outs)
                    row[other] = dict(one_ms=sum(t["one"]) / 2, other_ms=sum(t[other]) / 2, order=order)
                rec["blocks"].append(row)
            rec.update(outputs_stable=stable, status="DONE")
    except Exception:
        rec.update(status="ERROR", error=traceback.format_exc())
    rec["finished"] = time.time()
    write(args.out, rec)
    print(json.dumps(dict(status=rec.get("status"), blocks=len(rec["blocks"]), errors=list(rec.get("arm_errors", {})))))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("sweep", "paired"))
    p.add_argument("--case", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--calls", type=int, default=3)
    a = p.parse_args()
    return mode_sweep(a) if a.mode == "sweep" else mode_paired(a)


if __name__ == "__main__":
    raise SystemExit(main())
