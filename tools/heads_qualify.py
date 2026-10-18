"""Qualification of several heads in one launch (H > 1) on every single-GPU path (docs/flashrnn2/heads.md).

One case per process (a sticky CUDA error cannot leak into the next case). Every run of a case uses the same pinned
KernelConfig, so the H-head launch and the H = 1 launches run the same kernel.

Modes
  qualify --case JSON --out FILE
      forward   (no grad) for every head h of the subset: the H-head output sliced to h is bitwise equal to an
                H = 1 launch on head h's inputs (hidden output and FP32 carry); the H-head output is within the
                FP64-reference budget of tools/bw_qualify.py (rel_fro <= c, max_abs_scaled <= 5c)
      training  (case "train": true) gradients of <cot_h, h history> through training.recurrence with the same
                forward config.
                library path: every gradient within the FP64 gradient budget of tools/bw_qualify.py.
                kernel isolation: the two GEMMs outside the kernels (the gate preactivations the backward recomputes,
                and dR / dbias) take a different code path and summation order for H = 1 (addmm) and H > 1
                (einsum), so the library's d_wx cannot be bitwise per head. With the test hook
                per_head_gemms() both GEMMs of the H-head call run head by head through the H = 1 code path;
                then d_wx, d_init, d_r and d_bias must be bitwise equal per head against the H = 1 runs.
  repro --cell C --width D --heads H --batch B --steps T --arm {fwd_default,fwd_tilelang,train} --out FILE
      one arm of the scale-out phase A heads row in a fresh process, synchronizing after every call (the batch runs
      it with CUDA_LAUNCH_BLOCKING=1 and once under compute-sanitizer memcheck)
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, grad_reference, training, triton_rtiled_bw
from flashrnn.flashrnn2.execution_policy import KernelConfig, default_config


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def head_slice(leaves, h):
    wx, r, b, init = leaves
    return [wx[:, :, :, h:h + 1].contiguous(), r[:, h:h + 1].contiguous(), b[:, h:h + 1].contiguous(),
            init[:, :, :, h:h + 1].contiguous()]


def forward(leaves, cell, config):
    with torch.no_grad():
        out, carry = dispatch.recurrence(*leaves, cell, output_mode="hidden", snapshot_dtype=torch.float32,
                                         config=config)
    torch.cuda.synchronize()
    return out, carry


def train(leaves, cot_h, cell, config, backward=None):
    xs = [x.detach().clone().requires_grad_(True) for x in leaves]
    y, _ = training.recurrence(*xs, cell, output_mode="hidden", config=config, backward_options=dict(backward or {}))
    grads = torch.autograd.grad((y.float() * cot_h).sum(), xs)
    torch.cuda.synchronize()
    return y.detach(), list(grads)


class per_head_gemms:
    """Test hook: the backward's preactivation and weight-gradient GEMMs of an H-head call, head by head through the
    H = 1 code path (cuBLAS chooses its algorithm by shape, so only the same call per head is bitwise comparable)."""

    def __enter__(self):
        self.pre, self.wg = triton_rtiled_bw.preactivations, triton_rtiled_bw.weight_gradients
        orig_pre, orig_wg = self.pre, self.wg

        def pre(wx, recurrent, bias, initial, history, tf32=False):
            heads = wx.shape[3]
            if heads == 1:
                return orig_pre(wx, recurrent, bias, initial, history, tf32)
            parts = [orig_pre(wx[:, :, :, h:h + 1].contiguous(), recurrent[:, h:h + 1].contiguous(),
                              bias[:, h:h + 1].contiguous(), initial[:, :, :, h:h + 1].contiguous(),
                              history[:, :, :, h:h + 1].contiguous(), tf32) for h in range(heads)]
            return torch.cat([p[0] for p in parts], dim=3), torch.cat([p[1] for p in parts], dim=2)

        def wg(dg, h_prev, tf32=False, method="bmm"):
            heads = dg.shape[3]
            if heads == 1:
                return orig_wg(dg, h_prev, tf32, method)
            parts = [orig_wg(dg[:, :, :, h:h + 1].contiguous(), h_prev[:, :, h:h + 1].contiguous(), tf32, method)
                     for h in range(heads)]
            return torch.cat([p[0] for p in parts], dim=1), torch.cat([p[1] for p in parts], dim=1)

        triton_rtiled_bw.preactivations, triton_rtiled_bw.weight_gradients = pre, wg
        return self

    def __exit__(self, *exc):
        triton_rtiled_bw.preactivations, triton_rtiled_bw.weight_gradients = self.pre, self.wg


def mode_qualify(args):
    case = json.loads(Path(args.case).read_text())
    rec = dict(case=case, started=time.time(), gpu=torch.cuda.get_device_name())
    try:
        cfg = KernelConfig(**case["config"]) if case.get("config") else None  # None: the library's selection
        cell, heads = case["cell"], case["heads"]
        leaves, cot_h, cot_f = bw_qualify.inputs(dict(case, init="warm"), 4242 + case["width"] + heads)
        subset = case.get("head_subset") or list(range(heads))
        # Forward.
        out, carry = forward(leaves, cell, cfg)
        per_head = []
        for h in subset:
            o1, c1 = forward(head_slice(leaves, h), cell, cfg)
            per_head.append(dict(head=h, out_bitwise=torch.equal(out[:, :, h:h + 1], o1),
                                 carry_bitwise=torch.equal(carry[:, :, :, h:h + 1], c1)))
        xs = [x.detach().double().cpu() for x in leaves]
        with torch.no_grad():
            hist, _ = grad_reference.recurrence(*xs, cell)
        fwd_err = bw_qualify.compare(out.float(), hist[0])
        c = bw_qualify.BUDGET[case["dtype"]]
        fwd_ok = fwd_err["finite"] and fwd_err["rel_fro"] <= c and fwd_err["max_abs_scaled"] <= 5 * c
        rec["forward"] = dict(per_head=per_head, fp64=fwd_err, fp64_ok=fwd_ok,
                              bitwise_ok=all(p["out_bitwise"] and p["carry_bitwise"] for p in per_head))
        ok = fwd_ok and rec["forward"]["bitwise_ok"]
        # Training.
        if case.get("train"):
            bw = case.get("backward")  # e.g. {"backend": "hybrid"}; default: the library's choice for the width
            _, grads = train(leaves, cot_h, cell, cfg, bw)  # library path
            with per_head_gemms():
                _, hooked = train(leaves, cot_h, cell, cfg, bw)
            heads_rec = []
            for h in subset:
                _, g1 = train(head_slice(leaves, h), cot_h[:, :, h:h + 1].contiguous(), cell, cfg, bw)
                heads_rec.append(dict(
                    head=h, d_wx_bitwise=torch.equal(hooked[0][:, :, :, h:h + 1], g1[0]),
                    d_init_bitwise=torch.equal(hooked[3][:, :, :, h:h + 1], g1[3]),
                    d_r_bitwise=torch.equal(hooked[1][:, h:h + 1], g1[1]),
                    d_bias_bitwise=torch.equal(hooked[2][:, h:h + 1], g1[2]),
                    library_d_wx_bitwise=torch.equal(grads[0][:, :, :, h:h + 1], g1[0])))
            ref = bw_qualify.run_oracle(leaves, cot_h, cot_f, case, "hist")
            m = bw_qualify.metrics(grads, ref, cell)
            checks = bw_qualify.gate(m, case["dtype"])
            train_bitwise = all(x["d_wx_bitwise"] and x["d_init_bitwise"] and x["d_r_bitwise"] and x["d_bias_bitwise"]
                                for x in heads_rec)
            rec["training"] = dict(per_head=heads_rec, fp64=m, fp64_checks=checks, fp64_ok=all(checks.values()),
                                   bitwise_ok=train_bitwise)
            ok = ok and train_bitwise and all(checks.values())
        rec["pass_"] = bool(ok)
    except Exception:
        rec.update(pass_=False, error=traceback.format_exc())
    rec["finished"] = time.time()
    write(args.out, rec)
    print(json.dumps(dict(pass_=rec["pass_"], case=case.get("name"), error=(rec.get("error") or "")[-300:])))
    return 0 if rec["pass_"] else 1


def mode_repro(args):
    """The scale-out phase A heads row, one arm, in a fresh process."""
    rec = dict(args=vars(args), started=time.time())
    try:
        case = dict(cell=args.cell, dtype="float16", batch=args.batch, steps=args.steps, width=args.width,
                    heads=args.heads, init="warm")
        leaves, cot_h, _ = bw_qualify.inputs(case, 505 + args.width + args.heads)
        if args.arm == "fwd_default":
            cfg = default_config(args.width)
        elif args.arm == "fwd_tilelang":
            sync, s = training.TILELANG_TRAINING_FORWARD[args.width]
            cfg = KernelConfig(family="tilelang", slice_width=s, sync=sync)
        else:
            cfg = None
        rec["config"] = cfg.as_dict() if cfg is not None else "training default"
        for i in range(3):
            if args.arm == "train":
                train(leaves, cot_h, args.cell, cfg)
            else:
                forward(leaves, args.cell, cfg)
            torch.cuda.synchronize()
            rec[f"call{i}"] = "ok"
        rec["status"] = "OK"
    except Exception:
        rec.update(status="ERROR", error=traceback.format_exc())
    rec["finished"] = time.time()
    write(args.out, rec)
    print(json.dumps(dict(status=rec["status"], error=(rec.get("error") or "")[-300:])))
    return 0 if rec["status"] == "OK" else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("qualify", "repro"))
    p.add_argument("--case")
    p.add_argument("--out", required=True)
    for k in ("width", "heads", "batch", "steps"):
        p.add_argument(f"--{k}", type=int)
    p.add_argument("--cell")
    p.add_argument("--arm", choices=("fwd_default", "fwd_tilelang", "train"))
    a = p.parse_args()
    return mode_qualify(a) if a.mode == "qualify" else mode_repro(a)


if __name__ == "__main__":
    raise SystemExit(main())
