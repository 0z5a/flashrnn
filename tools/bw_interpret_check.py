"""CPU check of the R-tiled backward kernel's arithmetic with the Triton interpreter (TRITON_INTERPRET=1).

Runs the launch-per-step mode (no in-kernel barrier, so the sequential interpreter can execute it) on small shapes
with k = 1 and k > 1 CTAs per group and with ragged batch tiles, and compares every gradient with the FP64 oracle
(flashrnn2.grad_reference) on the same rounded inputs. The forward history fed to the kernel is the oracle's
forward rounded to FP32, so this checks the backward kernel and the surrounding GEMMs, not the forward kernels.
Not a GPU qualification. Writes bw-interpret.json.
"""
import os

os.environ["TRITON_INTERPRET"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from flashrnn.flashrnn2 import grad_reference, triton_rtiled_bw  # noqa: E402
from flashrnn.flashrnn2.triton_rtiled import SPIN_LIMIT, k_parts, plan  # noqa: E402

# The interpreter does not unwrap constexpr globals in Python operators; plain ints give the same (VERIFY = 0) code.
for _name in ("V_CHECK", "V_DELAY", "V_BROKEN_SINGLE", "V_BROKEN_NO_BARRIER", "V_BROKEN_ARRIVE_FIRST"):
    setattr(triton_rtiled_bw, _name, int(getattr(triton_rtiled_bw, _name)))


def run_kernel(wx, r, b, init, hist, d_hist, d_final, cell, slice_width, fast=None, fuse=False):
    batch, steps, _, heads, width = wx.shape
    states = init.shape[0]
    splits, groups, slots = plan(batch, heads, width, slice_width, sms=132)
    if fuse:
        xbuf, h_prev = triton_rtiled_bw.fused_buffer(slots, wx.contiguous(), b, init, hist)
        pre = wx.contiguous()
    else:
        pre, h_prev = triton_rtiled_bw.preactivations(wx, r, b, init, hist)
        xbuf = torch.zeros((slots, 2, 16, 4 * width), dtype=wx.dtype)
    k1, k2 = k_parts(width)
    dg = torch.zeros((batch, steps, 4, heads, width), dtype=torch.float32)
    d_init = torch.zeros((states, batch, 1, heads, width), dtype=torch.float32)
    carry = torch.zeros((2, batch, heads, width), dtype=torch.float32)
    sync = torch.zeros((slots,), dtype=torch.int32)
    err = torch.zeros((1,), dtype=torch.int32)
    for index in range(steps + 1):
        if index < steps:
            t_start, nsteps, from_final, tail = steps - 1 - index, 1, int(index == 0), 0
        else:
            t_start, nsteps, from_final, tail = 0, 0, 0, 1
        triton_rtiled_bw._rtiled_bw[(splits, slots)](
            pre, hist, init, d_hist, d_final, r, dg, d_init, carry, xbuf, sync, err,
            batch, steps, t_start, nsteps, from_final, tail, groups, slots, 0,
            heads, width, cell == "slstm", 16, slice_width, k1, k2, SPIN_LIMIT,
            **(dict(FAST=1, DS_C=int(fast[0]), DS_N=int(fast[1])) if fast else {}),
            **(dict(FUSE=int(fuse)) if fuse else {}))
    d_r, d_b = triton_rtiled_bw.weight_gradients(dg, h_prev)
    return [dg, d_r, d_b, d_init]


def case(cell, init_mode, batch, steps, heads, width, slice_width, seed):
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    wx = (torch.randn(batch, steps, 4, heads, width, generator=g) * 0.5).half()
    r = (torch.randn(4, heads, width, width, generator=g) * (0.5 / width ** 0.5)).half()
    b = (torch.randn(4, heads, width, generator=g) * 0.1).half()
    init = torch.randn(states, batch, 1, heads, width, generator=g) * 0.2
    if cell == "slstm":
        if init_mode == "zero":
            init[1:] = 0
        else:
            init[2] = init[2].abs() + 1
    init = init.half()
    d_hist = torch.randn(states, batch, steps, heads, width, generator=g)
    d_final = torch.randn(states, batch, 1, heads, width, generator=g)
    if cell == "slstm":
        d_hist[3] = 0
        d_final[3] = 0
    leaves = [x.double().requires_grad_(True) for x in (wx, r, b, init)]
    hist64, last64 = grad_reference.recurrence(*leaves, cell)
    ((hist64 * d_hist.double()).sum() + (last64 * d_final.double()).sum()).backward()
    ref = [x.grad for x in leaves]
    ours = run_kernel(wx, r, b, init, hist64.detach().float().contiguous(), d_hist, d_final, cell, slice_width)
    # The FAST variant (same values and arithmetic) must be bitwise identical, with all history components
    # and with only h carrying gradient.
    out_fast = {}
    for flags, dh_only in (((1, 1), False), ((0, 0), True)):
        dh = d_hist.clone()
        if dh_only:
            dh[1:] = 0
        base = run_kernel(wx, r, b, init, hist64.detach().float().contiguous(), dh, d_final, cell, slice_width)
        fast = run_kernel(wx, r, b, init, hist64.detach().float().contiguous(), dh, d_final, cell, slice_width, flags)
        out_fast[f"fast_bitwise_ds{flags[0]}{flags[1]}"] = all(torch.equal(x, y) for x, y in zip(base, fast))
    names = ["d_wx", "d_r", "d_bias", "d_init"]
    out = {}
    for name, a, e in zip(names, ours, ref):
        a, e = a.double(), e.double()
        if name == "d_init" and cell == "slstm":
            a, e = a[:3], e[:3]
        out[name] = dict(rel_fro=((a - e).norm() / e.norm().clamp_min(1e-300)).item(),
                         max_abs_scaled=((a - e).abs().max() / e.abs().max().clamp_min(1e-300)).item())
    # The FUSE variant recomputes the preactivations in the kernel: compared with the FP64 oracle like the base.
    for fuse in (1, 2):
        fused = run_kernel(wx, r, b, init, hist64.detach().float().contiguous(), d_hist, d_final, cell, slice_width,
                           (1, 1), fuse=fuse)
        for name, a, e in zip(["d_wx", "d_r", "d_bias", "d_init"], fused, ref):
            a, e = a.double(), e.double()
            if name == "d_init" and cell == "slstm":
                a, e = a[:3], e[:3]
            out[f"fused{fuse}_{name}"] = dict(rel_fro=((a - e).norm() / e.norm().clamp_min(1e-300)).item(),
                                              max_abs_scaled=((a - e).abs().max() / e.abs().max().clamp_min(1e-300)).item())
    out.update(out_fast)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for cell, init in (("lstm", "warm"), ("slstm", "zero"), ("slstm", "warm")):
        for batch, steps, heads, width, s in ((3, 5, 1, 32, 32), (3, 5, 1, 32, 16), (19, 4, 2, 48, 16),
                                              (5, 3, 1, 64, 32)):
            m = case(cell, init, batch, steps, heads, width, s, 7 + width + batch)
            ok = all(v is True if isinstance(v, bool) else (v["rel_fro"] < 1e-2 and v["max_abs_scaled"] < 5e-2)
                     for v in m.values())
            rows.append(dict(cell=cell, init=init, batch=batch, steps=steps, heads=heads, width=width, slice_width=s,
                             metrics=m, pass_=ok))
            print(json.dumps(dict(cell=cell, init=init, shape=[batch, steps, heads, width, s], pass_=ok,
                                  worst=max(v["rel_fro"] for v in m.values() if isinstance(v, dict)),
                                  fast={k: v for k, v in m.items() if k.startswith("fast")})), flush=True)
    status = "PASS" if all(r["pass_"] for r in rows) else "FAIL"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "bw-interpret.json").write_text(json.dumps(dict(rows=rows, status=status), indent=1))
    print("STATUS", status)


if __name__ == "__main__":
    main()
