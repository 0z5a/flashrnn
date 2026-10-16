"""Scale-out measurements on one node (docs/flashrnn2/scale-out-plan.md, phase A). One process drives all GPUs.

Modes (one JSON record per call, written to --out/<mode>.json)
  qualify   multi-GPU forward (design C, every schedule) bitwise against the single-GPU R-tiled forward of the same
            slice width (triton_rtiled, chunked-K accumulation is the same sequence of MMAs); also the stream mode
            (design B) and P = 1 against it
  time      per-step cost of every configuration in --configs: median of --reps calls at T = --steps and T = 64 (the
            slope separates per-step cost from setup), in-kernel phases (compute, publish, wait) from a PROF build
            of the same configuration, and the model prediction from --micro (micro.json of the same job)
  heads     many heads: forward and training step throughput against H on one GPU (R-tiled Triton at the table
            default and the measured TileLang configuration when installed), and heads partitioned over the GPUs
Config strings: "P:D:S:schedule:rmode[:stride]" e.g. "4:2048:8:direct:smem", "1:4096:16:direct:stream".
"""
import argparse
import json
import math
import time
import traceback
from pathlib import Path

import torch

from flashrnn.flashrnn2 import multi_gpu, reference, triton_rtiled
from flashrnn.flashrnn2.state_contract import begin_sequence


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=1, default=str))
    tmp.replace(path)


def parse(config):
    parts = config.split(":")
    peers, width, s, sched, rmode = int(parts[0]), int(parts[1]), int(parts[2]), parts[3], parts[4]
    stride = int(parts[5]) if len(parts) > 5 else 1
    return dict(peers=peers, width=width, slice_width=s, schedule=sched, rmode=rmode, stride=stride, name=config)


def inputs(cell, batch, steps, heads, width, seed, device):
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    wx = (torch.randn(batch, steps, 4, heads, width, generator=g) * 0.5).half()
    r = (torch.randn(4, heads, width, width, generator=g) * (0.5 / width ** 0.5)).half()
    b = (torch.randn(4, heads, width, generator=g) * 0.1).half()
    init = torch.randn(states, batch, 1, heads, width, generator=g) * 0.2
    if cell == "slstm":
        init[2] = init[2].abs() + 1
    return [x.to(device) for x in (wx, r, b, init.half())]


def single_gpu(leaves, cell, s, warps, device):
    wx, r, b, init = leaves
    ctx = begin_sequence(init, cell, wx.dtype)
    out, carry = triton_rtiled.recurrence(wx, r, b, init, cell, sequence_context=ctx, slice_width=s,
                                          num_warps=warps, output_mode="hidden", snapshot_dtype=torch.float32)
    torch.cuda.synchronize(device)
    return out, carry


def median(xs):
    return sorted(xs)[len(xs) // 2]


def mode_qualify(a, devices):
    rec = dict(mode="qualify", configs=[], started=time.time())
    for cell in ("lstm", "slstm"):
        for config in a.configs:
            c = parse(config)
            leaves = inputs(cell, a.batch, 48, 1, c["width"], 101 + c["width"], f"cuda:{devices[0]}")
            row = dict(c, cell=cell)
            try:
                ref = None
                if 4 * c["width"] * c["slice_width"] * 2 <= 160 * 1024 and c["width"] // c["slice_width"] <= 132:
                    ref, ref_carry = single_gpu(leaves, cell, c["slice_width"], a.warps, devices[0])
                diag = {}
                out, final, ms = multi_gpu.recurrence(*leaves, cell, devices=devices[:c["peers"]],
                                                      slice_width=c["slice_width"], schedule=c["schedule"],
                                                      stride=c["stride"], rmode=c["rmode"], num_warps=a.warps,
                                                      k_chunk=a.k_chunk, diagnostics=diag)
                row.update(kernel_ms=diag["kernel_ms"], registers=diag["registers"], spills=diag["spills"],
                           shared_bytes=diag["shared_bytes"])
                if ref is not None:
                    row["bitwise_vs_single_gpu"] = bool(torch.equal(out, ref))
                    row["max_rel_vs_single_gpu"] = ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
                    row["carry_bitwise"] = bool(torch.equal(final[0], ref_carry[0]))
                else:
                    row["bitwise_vs_single_gpu"] = None  # the single-GPU kernel cannot hold this slice
                # Every multi-GPU configuration must also equal the P = 1 direct run of the same kernel.
                base, _, _ = multi_gpu.recurrence(*leaves, cell, devices=devices[:1], slice_width=c["slice_width"],
                                                  schedule="direct", rmode=c["rmode"], num_warps=a.warps,
                                                  k_chunk=a.k_chunk) if c["peers"] > 1 else (out, None, None)
                row["bitwise_vs_p1"] = bool(torch.equal(out, base))
                row["finite"] = bool(torch.isfinite(out).all())
                # FP64 reference on the CPU (same rounded inputs), tolerance 2e-2 of the output scale.
                wx64, r64, b64, i64 = (x.double().cpu() for x in leaves)
                hist, _ = reference.recurrence(wx64, r64, b64, i64, cell)
                ref64 = hist[0].float().to(out.device)
                row["max_rel_vs_fp64"] = ((out - ref64).abs().max() / ref64.abs().max().clamp_min(1e-30)).item()
                row["pass_"] = (row["finite"] and row["bitwise_vs_p1"] and row["bitwise_vs_single_gpu"] is not False
                                and row["max_rel_vs_fp64"] < 2e-2)
            except Exception:  # noqa: BLE001
                row.update(error=traceback.format_exc()[-1500:], pass_=False)
            rec["configs"].append(row)
            print(json.dumps({k: row.get(k) for k in ("name", "cell", "bitwise_vs_single_gpu", "bitwise_vs_p1",
                                                      "kernel_ms", "pass_", "error")}), flush=True)
            write(Path(a.out) / "qualify.json", rec)
    rec["status"] = "PASS" if all(r["pass_"] for r in rec["configs"]) else "FAIL"
    rec["finished"] = time.time()
    write(Path(a.out) / "qualify.json", rec)
    return 0 if rec["status"] == "PASS" else 1


def predict(micro, c, batch_tile_bytes, r_slice_bytes, gamma_us):
    """Model prediction (us per step) from the microbenchmarks of the same job (alpha, beta per pair, HBM GB/s)."""
    if not micro or c["peers"] == 1:
        link = dict(alpha=0.0, beta=0.0)
    else:
        pairs = [v for k, v in micro["pair"].items()]
        link = dict(alpha=max(p["alpha_us"] for p in pairs), beta=max(p["beta_us_per_MB"] for p in pairs) / 1e6)
    P, m = c["peers"], batch_tile_bytes
    rounds = multi_gpu.rounds_of(c["schedule"], P)
    if P == 1:
        exchange = 0.0
    elif c["schedule"] == "direct":
        exchange = link["alpha"] + link["beta"] * m / P
    else:
        exchange = rounds * link["alpha"] + link["beta"] * m * (P - 1) / P
    stream = 0.0
    if c["rmode"] == "stream" and micro:
        hbm = min(v["copy_GBps"] for v in micro["hbm"].values()) * 1e9  # bytes per second
        stream = r_slice_bytes / hbm * 1e6
    alpha_l2 = micro.get("alpha_l2_us", 1.3) if micro else 1.3
    return dict(exchange_us=exchange, stream_us=stream, barrier_us=alpha_l2,
                total_us=gamma_us + exchange + stream + alpha_l2, rounds=rounds, link=link)


def mode_time(a, devices):
    micro = json.loads(Path(a.micro).read_text()) if a.micro and Path(a.micro).exists() else None
    rec = dict(mode="time", configs=[], started=time.time(), micro=a.micro, reps=a.reps)
    for cell in a.cells.split(","):
        for config in a.configs:
            c = parse(config)
            row = dict(c, cell=cell)
            try:
                res = {}
                for steps in (64, a.steps):
                    leaves = inputs(cell, a.batch, steps, 1, c["width"], 303 + c["width"], f"cuda:{devices[0]}")
                    run = lambda: multi_gpu.recurrence(*leaves, cell, devices=devices[:c["peers"]],  # noqa: E731
                                                       slice_width=c["slice_width"], schedule=c["schedule"],
                                                       stride=c["stride"], rmode=c["rmode"], num_warps=a.warps,
                                                       k_chunk=a.k_chunk)
                    run()
                    times = [run()[2] for _ in range(a.reps)]
                    res[steps] = dict(kernel_ms=median([max(t.values()) for t in times]),
                                      per_device_ms={str(d): median([t[d] for t in times]) for d in times[0]})
                per_step = (res[a.steps]["kernel_ms"] - res[64]["kernel_ms"]) / (a.steps - 64) * 1000.0
                diag = {}
                multi_gpu.recurrence(*leaves, cell, devices=devices[:c["peers"]], slice_width=c["slice_width"],
                                     schedule=c["schedule"], stride=c["stride"], rmode=c["rmode"], num_warps=a.warps,
                                     k_chunk=a.k_chunk, profile=True, diagnostics=diag)
                phases = diag["phases"]
                mean = [sum(phases[d]["mean_ns"][i] for d in phases) / len(phases) / 1000.0 for i in range(3)]
                worst = [max(phases[d]["max_ns"][i] for d in phases) / 1000.0 for i in range(3)]
                m = 16 * c["width"] * 2
                r_slice = 4 * c["width"] * (c["width"] // c["peers"]) * 2
                pred = predict(micro, c, m, r_slice, mean[0])
                row.update(ms=res, per_step_us=per_step, phases_mean_us=dict(zip(("compute", "publish", "wait"), mean)),
                           phases_max_us=dict(zip(("compute", "publish", "wait"), worst)),
                           profiled_kernel_ms=diag["kernel_ms"], registers=diag["registers"], spills=diag["spills"],
                           shared_bytes=diag["shared_bytes"], message_bytes=m, r_slice_bytes=r_slice, model=pred,
                           model_ratio=per_step / pred["total_us"] if pred["total_us"] else None)
            except Exception:  # noqa: BLE001
                row["error"] = traceback.format_exc()[-1500:]
            rec["configs"].append(row)
            print(json.dumps({k: row.get(k) for k in ("name", "cell", "per_step_us", "phases_mean_us", "model_ratio",
                                                      "error")}), flush=True)
            write(Path(a.out) / "time.json", rec)
    rec["finished"] = time.time()
    write(Path(a.out) / "time.json", rec)
    return 0


def mode_heads(a, devices):
    from flashrnn.flashrnn2 import dispatch, training
    from flashrnn.flashrnn2.execution_policy import KernelConfig, default_config

    rec = dict(mode="heads", rows=[], started=time.time())
    widths = [int(x) for x in a.widths.split(",")]
    heads_list = [int(x) for x in a.heads.split(",")]
    tl_ok = training._tilelang_installed()
    for cell in a.cells.split(","):
        for width in widths:
            for heads in heads_list:
                for batch in (16, 64):
                    row = dict(cell=cell, width=width, heads=heads, batch=batch, steps=a.steps)
                    try:
                        dev = f"cuda:{devices[0]}"
                        leaves = inputs(cell, batch, a.steps, heads, width, 505 + width + heads, dev)
                        configs = {"triton_default": default_config(width)}
                        if tl_ok and width in training.TILELANG_TRAINING_FORWARD:
                            sync, s = training.TILELANG_TRAINING_FORWARD[width]
                            configs["tilelang"] = KernelConfig(family="tilelang", slice_width=s, sync=sync)
                        fwd = {}
                        with torch.no_grad():
                            for name, cfg in configs.items():
                                def f(cfg=cfg):
                                    return dispatch.recurrence(*leaves, cell, output_mode="hidden", config=cfg)
                                f(); torch.cuda.synchronize(dev)
                                ts = []
                                for _ in range(a.reps):
                                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    e0.record(); f(); e1.record(); torch.cuda.synchronize(dev)
                                    ts.append(e0.elapsed_time(e1))
                                fwd[name] = median(ts)
                        row["forward_ms"] = fwd
                        # Training step (forward on the first configuration, default backward options).
                        def step():
                            xs = [x.detach().requires_grad_(True) for x in leaves]
                            y, _ = training.recurrence(*xs, cell, output_mode="hidden")
                            return torch.autograd.grad(y.float().sum(), xs)
                        step(); torch.cuda.synchronize(dev)
                        ts = []
                        for _ in range(max(3, a.reps // 2)):
                            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            e0.record(); step(); e1.record(); torch.cuda.synchronize(dev)
                            ts.append(e0.elapsed_time(e1))
                        row["train_step_ms"] = median(ts)
                        # Heads partitioned over the GPUs: each GPU runs heads / P of them concurrently.
                        if len(devices) > 1 and heads % len(devices) == 0:
                            parts = {}
                            per = heads // len(devices)
                            for i, d in enumerate(devices):
                                sl = slice(i * per, (i + 1) * per)
                                wx, r, b, init = leaves
                                parts[d] = [wx[:, :, :, sl].contiguous().to(f"cuda:{d}"), r[:, sl].contiguous().to(f"cuda:{d}"),
                                            b[:, sl].contiguous().to(f"cuda:{d}"), init[:, :, :, sl].contiguous().to(f"cuda:{d}")]
                            cfg = configs["triton_default"]

                            def all_devices():
                                evs = {}
                                for d in devices:
                                    torch.cuda.set_device(d)
                                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    e0.record()
                                    with torch.no_grad():
                                        dispatch.recurrence(*parts[d], cell, output_mode="hidden", config=cfg)
                                    e1.record()
                                    evs[d] = (e0, e1)
                                for d in devices:
                                    torch.cuda.synchronize(d)
                                return max(evs[d][0].elapsed_time(evs[d][1]) for d in devices)
                            all_devices()
                            row["forward_ms_heads_over_gpus"] = median([all_devices() for _ in range(a.reps)])
                            torch.cuda.set_device(devices[0])
                    except Exception:  # noqa: BLE001
                        row["error"] = traceback.format_exc()[-1500:]
                    rec["rows"].append(row)
                    print(json.dumps({k: row.get(k) for k in ("cell", "width", "heads", "batch", "forward_ms",
                                                              "train_step_ms", "forward_ms_heads_over_gpus", "error")}),
                          flush=True)
                    write(Path(a.out) / "heads.json", rec)
    rec["finished"] = time.time()
    write(Path(a.out) / "heads.json", rec)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("qualify", "time", "heads"))
    p.add_argument("--out", required=True)
    p.add_argument("--devices", default="0,1,2,3")
    p.add_argument("--configs", default="")
    p.add_argument("--cells", default="lstm,slstm")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--k-chunk", type=int, default=512)
    p.add_argument("--reps", type=int, default=9)
    p.add_argument("--micro", default="")
    p.add_argument("--widths", default="192,384,768")
    p.add_argument("--heads", default="1,4,16,64")
    a = p.parse_args()
    a.configs = [c for c in a.configs.split(",") if c]
    torch.backends.cuda.matmul.allow_tf32 = False
    devices = [int(x) for x in a.devices.split(",")]
    if a.mode != "heads" or len(devices) > 1:
        multi_gpu.enable_peer_access(devices)
    return {"qualify": mode_qualify, "time": mode_time, "heads": mode_heads}[a.mode](a, devices)


if __name__ == "__main__":
    raise SystemExit(main())
