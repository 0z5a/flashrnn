"""Time breakdown of one training step (forward + backward), FlashRNN2 against pinned FlashRNN1 (timing only).

Usage: bw_profile.py --out DIR --case '{"cell", "dtype", "width", "batch", "steps"}' [--tilelang]
Medians of 15 CUDA-event timed calls after warm-up:
  fwd_full / fwd_hidden   FlashRNN2 forward (full FP32 history, as the training path runs it / hidden output) per
                          forward backend (Triton default, Triton R-tiled S32/4w for D > 128, and with --tilelang the
                          qualified TileLang cluster/grid configuration)
  bw_phases               backward wrapper split into preactivation GEMM, sequential kernel, dR/dbias, for the
                          kernel variants base / fast (bitwise-identical) and tf32 GEMMs; several slice widths
  step                    complete training steps (training.recurrence + autograd.grad) per variant
  v1_fwd / v1_step        FlashRNN1 cuda_fused forward only / forward + backward (the pinned extension is built in the
                          non-TileLang environment; with --tilelang the run records whether it imports there)
Every FlashRNN2 step variant's gradients are compared with the base variant (bitwise for "fast", max relative
difference otherwise). Writes profile.json.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, training, triton_rtiled_bw
from flashrnn.flashrnn2.execution_policy import KernelConfig, default_config

REPS = 15


def median_ms(fn, reps=REPS):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def tilelang_forward(width):
    if width <= 512:
        return KernelConfig(family="tilelang", slice_width=16 if width <= 256 else 32, sync="cluster")
    return KernelConfig(family="tilelang", slice_width=16, sync="grid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--tilelang", action="store_true")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    case = dict(json.loads(a.case), init="warm")
    cell, width = case["cell"], case["width"]
    leaves, cot_h, _ = bw_qualify.inputs(case, 77 + width + case["batch"])
    rec = dict(case=case, tilelang_env=a.tilelang, started=time.time(), fwd={}, bw_phases={}, step={}, errors={})
    forwards = {"triton_default": default_config(width)}
    if width > 128:
        forwards["triton_s32w4"] = KernelConfig(family="mma_rtiled", warps=4, slice_width=32, sync="grid")
    if a.tilelang:
        forwards["tilelang"] = tilelang_forward(width)
    for name, cfg in forwards.items():
        try:
            with torch.no_grad():
                rec["fwd"][name] = dict(
                    full=median_ms(lambda: dispatch.recurrence(*leaves, cell, output_mode="full",
                                                               snapshot_dtype=torch.float32, config=cfg)),
                    hidden=median_ms(lambda: dispatch.recurrence(*leaves, cell, output_mode="hidden", config=cfg)))
        except Exception:  # noqa: BLE001
            rec["errors"][f"fwd_{name}"] = traceback.format_exc()[-1500:]
    # Backward phases on a fixed history (Triton default forward).
    with torch.no_grad():
        history, _ = dispatch.recurrence(*leaves, cell, output_mode="full", snapshot_dtype=torch.float32,
                                         config=forwards["triton_default"])
    d_hist = torch.zeros_like(history)
    d_hist[0] = cot_h
    d_final = torch.zeros(leaves[3].shape, device="cuda", dtype=torch.float32)
    slices = [s for s in (16, 32, 64) if s <= width and width % s == 0 and width // s <= 132
              and not (width > 512 and s == 64)]
    variants = {}
    for s in slices:
        for warps in (4, 8):
            variants[f"s{s}w{warps}_base"] = dict(slice_width=s, num_warps=warps)
            variants[f"s{s}w{warps}_fast"] = dict(slice_width=s, num_warps=warps, fast=True,
                                                  history_grad_components=(True, False, False))
            variants[f"s{s}w{warps}_fast_tf32"] = dict(variants[f"s{s}w{warps}_fast"], tf32=True)
    reference = {}
    for name, opts in variants.items():
        try:
            def run(opts=opts):
                return triton_rtiled_bw.backward(*leaves, history, d_hist, d_final, cell, **opts)
            out = run()
            torch.cuda.synchronize()
            base_name = name.split("_")[0] + "_base"
            if name.endswith("_base"):
                reference[name] = out
            ref = reference.get(base_name)
            if name.endswith("_fast") and ref is not None:
                equal = all(torch.equal(x, y) for x, y in zip(out, ref))
            elif ref is not None:
                equal = max(((x - y).abs().max() / y.abs().max().clamp_min(1e-30)).item() for x, y in zip(out, ref))
            else:
                equal = None
            phases = []
            for _ in range(5):
                diag = dict(time=True)
                triton_rtiled_bw.backward(*leaves, history, d_hist, d_final, cell, diagnostics=diag, **opts)
                phases.append(diag["ms"])
            med = {k: sorted(x[k] for x in phases)[2] for k in phases[0]}
            rec["bw_phases"][name] = dict(ms=med, total=median_ms(run, 7), vs_base=equal)
        except Exception:  # noqa: BLE001
            rec["errors"][f"bw_{name}"] = traceback.format_exc()[-1500:]
    # Complete training steps.
    best_s = min((n for n in rec["bw_phases"] if n.endswith("_fast")), key=lambda n: rec["bw_phases"][n]["total"],
                 default=None)
    step_variants = {}
    for fname, cfg in forwards.items():
        for bname, opts in variants.items():
            if bname.endswith("_fast") and (bname == best_s or bname.startswith("s32w4") or bname.startswith("s16w4")):
                step_variants[f"{fname}+{bname}"] = (cfg, opts)
        step_variants[f"{fname}+s32w8_base" if width > 32 else f"{fname}+base"] = (cfg, variants.get("s32w8_base", {}))
    for name, (cfg, opts) in step_variants.items():
        try:
            o = {k: v for k, v in opts.items() if k != "history_grad_components"}

            def step(cfg=cfg, o=o):
                xs = [x.detach().requires_grad_(True) for x in leaves]
                y, _ = training.recurrence(*xs, cell, output_mode="hidden", config=cfg, backward_options=o)
                return torch.autograd.grad((y.float() * cot_h).sum(), xs)
            rec["step"][name] = median_ms(step)
        except Exception:  # noqa: BLE001
            rec["errors"][f"step_{name}"] = traceback.format_exc()[-1500:]
    # FlashRNN1.
    try:
        import rtiled_m4
        v1 = rtiled_m4.v1_module()
        config = v1["config"](function=cell, backend="cuda_fused", dtype=case["dtype"], batch_size=case["batch"],
                              hidden_dim=width, num_heads=1)
        with torch.no_grad():
            rec["v1_fwd"] = median_ms(lambda: v1["flashrnn"](*leaves, config=config))

        def v1_step():
            xs = [x.detach().requires_grad_(True) for x in leaves]
            states, _ = v1["flashrnn"](*xs, config=config)
            return torch.autograd.grad((states[0].float() * cot_h).sum(), xs)
        rec["v1_step"] = median_ms(v1_step)
        rec["v1_status"] = "AVAILABLE"
    except Exception:  # noqa: BLE001
        rec["v1_status"] = "UNAVAILABLE"
        rec["errors"]["v1"] = traceback.format_exc()[-1500:]
    rec["finished"] = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(rec, indent=1, default=str))
    (out / "DONE.json").write_text(json.dumps(dict(finished=time.time())))
    print(json.dumps(dict(case=case, fwd=rec["fwd"], step=rec["step"], v1=(rec.get("v1_fwd"), rec.get("v1_step")),
                          errors=list(rec["errors"]))), flush=True)


if __name__ == "__main__":
    main()
