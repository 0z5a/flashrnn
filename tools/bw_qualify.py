"""GPU qualification of the FlashRNN2 training path (forward + R-tiled Triton backward), one case per call.

Case JSON: {cell, dtype, width, slice_width, warps, batch, steps, init (zero|warm), forward (optional KernelConfig),
fast, tf32, fuse (0|1|2: preactivation recompute fused into the backward kernel; needs fast)}.
Stages
  grads     gradients of two losses against the FP64 oracle (flashrnn2.grad_reference on the same rounded inputs):
            "hist" = <dH, h history>; "full" = <dH, h history> + <dF, final carry (h, c, n)>.
            Metrics per tensor (dWx, dR, dbias, d initial h/c/n): relative Frobenius error and max-abs error over
            max|ref|. Pre-registered gate (BUDGET below): every metric finite, rel_fro <= C[dtype],
            max_abs_scaled <= 5 C[dtype], and, where the pinned FlashRNN1 cuda_fused baseline runs the same "hist"
            loss, rel_fro <= max(1.25 x v1 rel_fro, C[dtype] / 10).
  v1        pinned FlashRNN1 cuda_fused forward+backward on the same inputs and "hist" loss (calibration); an
            unavailable baseline is recorded, not a failure
  modes     persistent (sync=grid) == one launch per step (sync=launch), bitwise, all gradients; barrier counts
            and error flag
  vs_base   (cases with "fast", not "fuse") the fast kernel variant's gradients bitwise equal to the base kernel's;
            a fused case is not bitwise to the GEMM path by construction and is gated by grads/modes/repeat only
  repeat    20 complete training steps (forward + backward) bitwise identical
Writes bw.json (status PASS only if every gated stage passes) and DONE.json.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch

from flashrnn.flashrnn2 import grad_reference, training, triton_rtiled_bw
from flashrnn.flashrnn2.execution_policy import KernelConfig, default_config

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}
BUDGET = {"float16": 1e-2, "bfloat16": 5e-2}
V1_FACTOR = 1.25


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8),
                                                                     b.contiguous().view(torch.uint8))


def inputs(case, seed):
    cell, dtype, b, t, d = (case[k] for k in ("cell", "dtype", "batch", "steps", "width"))
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    wx = torch.randn(b, t, 4, 1, d, generator=g) * 0.5
    r = torch.randn(4, 1, d, d, generator=g) * (0.5 / d ** 0.5)
    bias = torch.randn(4, 1, d, generator=g) * 0.1
    init = torch.randn(states, b, 1, 1, d, generator=g) * 0.2
    if cell == "slstm":
        if case["init"] == "zero":
            init[1:] = 0
        else:
            init[2] = init[2].abs() + 1
    cot_h = torch.randn(b, t, 1, d, generator=g)
    cot_f = torch.randn(states, b, 1, 1, d, generator=g)
    if cell == "slstm":
        cot_f[3] = 0
    leaves = [x.to(DT[dtype]).cuda() for x in (wx, r, bias, init)]
    return leaves, cot_h.cuda(), cot_f.cuda()


def run_ours(leaves, cot_h, cot_f, case, loss, sync="grid", diagnostics=None):
    cell = case["cell"]
    forward = KernelConfig(**case["forward"]) if case.get("forward") else default_config(case["width"])
    opts = dict(slice_width=case["slice_width"], num_warps=case["warps"], sync=sync, fast=case.get("fast", False),
                tf32=case.get("tf32", False), fuse_pre=case.get("fuse", 0))
    if diagnostics is not None:
        opts["diagnostics"] = diagnostics
    xs = [x.detach().clone().requires_grad_(True) for x in leaves]
    y, carry = training.recurrence(*xs, cell, output_mode="hidden", snapshot_dtype=torch.float32, config=forward,
                                   backward_options=opts)
    total = (y * cot_h).sum()
    if loss == "full":
        total = total + (carry * cot_f).sum()
    total.backward()
    torch.cuda.synchronize()
    return [x.grad for x in xs]


def run_oracle(leaves, cot_h, cot_f, case, loss):
    xs = [x.detach().double().cpu().requires_grad_(True) for x in leaves]
    hist, last = grad_reference.recurrence(*xs, case["cell"])
    total = (hist[0] * cot_h.double().cpu()).sum()
    if loss == "full":
        total = total + (last * cot_f.double().cpu()).sum()
    total.backward()
    return [x.grad for x in xs]


def run_v1(leaves, cot_h, case):
    import rtiled_m4  # pinned FlashRNN1 loader (FRNN_V1_BASE / FRNN_V1_BUILD_ROOT)

    v1 = rtiled_m4.v1_module()
    config = v1["config"](function=case["cell"], backend="cuda_fused", dtype=case["dtype"],
                          batch_size=case["batch"], hidden_dim=case["width"], num_heads=1)
    xs = [x.detach().clone().requires_grad_(True) for x in leaves]
    states, _ = v1["flashrnn"](*xs, config=config)
    (states[0].float() * cot_h).sum().backward()
    torch.cuda.synchronize()
    return [x.grad for x in xs]


def metrics(ours, ref, cell):
    names = ["d_wx", "d_r", "d_bias"]
    out = {}
    for name, a, b in zip(names, ours[:3], ref[:3]):
        out[name] = compare(a, b)
    comps = ("h", "c", "n") if cell == "slstm" else ("h", "c")
    for k, comp in enumerate(comps):
        out[f"d_init_{comp}"] = compare(ours[3][k], ref[3][k])
    return out


def compare(a, b):
    a = a.detach().double().cpu()
    b = b.detach().double().cpu()
    diff = a - b
    scale = b.abs().max().item()
    return dict(rel_fro=(diff.norm() / max(b.norm().item(), 1e-300)).item(),
                max_abs_scaled=(diff.abs().max().item() / max(scale, 1e-300)),
                finite=bool(torch.isfinite(a).all()), ref_max=scale)


def gate(m, dtype, v1m=None):
    c = BUDGET[dtype]
    checks = {}
    for name, v in m.items():
        ok = v["finite"] and v["rel_fro"] <= c and v["max_abs_scaled"] <= 5 * c
        if v1m is not None and name in v1m:
            ok = ok and v["rel_fro"] <= max(V1_FACTOR * v1m[name]["rel_fro"], c / 10)
        checks[name] = ok
    return checks


def stage_grads(case, rec):
    leaves, cot_h, cot_f = inputs(case, 1000 + case["width"] + case["batch"])
    out = {}
    v1m = None
    if os.environ.get("FRNN_V1_BASE") and case["init"] in ("zero", "warm") and not case.get("no_v1"):
        try:
            v1 = run_v1(leaves, cot_h, case)
            v1m = metrics(v1, run_oracle(leaves, cot_h, cot_f, case, "hist"), case["cell"])
            rec["v1"] = dict(status="AVAILABLE", metrics=v1m)
        except Exception:  # noqa: BLE001 - an unavailable baseline is a recorded result
            rec["v1"] = dict(status="UNAVAILABLE", error=traceback.format_exc()[-2000:])
    for loss in ("hist", "full"):
        ref = run_oracle(leaves, cot_h, cot_f, case, loss)
        m = metrics(run_ours(leaves, cot_h, cot_f, case, loss), ref, case["cell"])
        checks = gate(m, case["dtype"], v1m if loss == "hist" else None)
        out[loss] = dict(metrics=m, checks=checks, pass_=all(checks.values()))
    rec["grads"] = out
    return all(v["pass_"] for v in out.values())


def stage_modes(case, rec):
    leaves, cot_h, cot_f = inputs(case, 7 + case["width"])
    diag = {}
    grid = run_ours(leaves, cot_h, cot_f, case, "full", "grid", diag)
    launch = run_ours(leaves, cot_h, cot_f, case, "full", "launch")
    counts = diag["sync_counts"][0].cpu().tolist()
    rec["modes"] = dict(bitwise=[same(a, b) for a, b in zip(grid, launch)], sync_counts=counts,
                        expected=diag["expected_sync_counts"], error_flag=int(diag["error_flag"].item()),
                        registers=diag["registers"], spills=diag["spills"], shared_bytes=diag["shared_bytes"],
                        splits=diag["splits"], slots=diag["slots"])
    m = rec["modes"]
    return all(m["bitwise"]) and m["sync_counts"] == m["expected"] and m["error_flag"] == 0


def stage_vs_base(case, rec):
    """The fast kernel variant must reproduce the qualified base kernel bit for bit (both losses)."""
    leaves, cot_h, cot_f = inputs(case, 13 + case["width"])
    base_case = dict(case, fast=False, tf32=False, fuse=0)
    out = {}
    for loss in ("hist", "full"):
        a = run_ours(leaves, cot_h, cot_f, case, loss)
        b = run_ours(leaves, cot_h, cot_f, base_case, loss)
        out[loss] = [same(x, y) for x, y in zip(a, b)]
    rec["vs_base"] = out
    return all(all(v) for v in out.values())


def stage_repeat(case, rec):
    leaves, cot_h, cot_f = inputs(case, 11 + case["width"])
    first = run_ours(leaves, cot_h, cot_f, case, "full")
    equal = []
    for _ in range(20):
        again = run_ours(leaves, cot_h, cot_f, case, "full")
        equal.append(all(same(a, b) for a, b in zip(first, again)))
    rec["repeat"] = dict(repeats=20, bitwise=sum(equal))
    return all(equal)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--stages", default="grads,modes,repeat")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    case = json.loads(a.case)
    rec = dict(case=case, started=time.time(), gates={}, budget=dict(rel_fro=BUDGET[case["dtype"]], v1_factor=V1_FACTOR))
    for stage, fn in (("grads", stage_grads), ("vs_base", stage_vs_base), ("modes", stage_modes),
                      ("repeat", stage_repeat)):
        if stage not in a.stages.split(","):
            continue
        if stage == "vs_base" and case.get("fuse"):
            rec["gates"][stage] = "NOT_APPLICABLE_FUSED"  # pre-registered: not a gate for fused cases
            continue
        try:
            rec["gates"][stage] = "PASS" if fn(case, rec) else "FAIL"
        except Exception:  # noqa: BLE001
            rec["gates"][stage] = "ERROR"
            rec[f"{stage}_error"] = traceback.format_exc()[-3000:]
        write(Path(a.out) / "bw.json", rec)
    rec["status"] = "PASS" if rec["gates"] and all(v in ("PASS", "NOT_APPLICABLE_FUSED")
                                                   for v in rec["gates"].values()) else "FAIL"
    rec["finished"] = time.time()
    write(Path(a.out) / "bw.json", rec)
    (Path(a.out) / "DONE.json").write_text(json.dumps(dict(finished=time.time())))
    print(json.dumps(dict(case=case, gates=rec["gates"], status=rec["status"])), flush=True)


if __name__ == "__main__":
    main()
