"""NVSHMEM symmetric exchange buffers for the multi-GPU forward and backward, one process per GPU
(docs/flashrnn2/scale-out-plan.md section 11).

Usage: scale_nvshmem.py --out DIR --devices 0,1,2,3 --cases JSON [--reps 9]
Cases: {"kind": "forward", "config": "P:D:S:schedule:smem[:stride]", "k_chunk", "cell"} with P = the number of devices
(B = 16, T = 48), or {"kind": "backward", "cell", "width", "batch", "steps", "batch_rows"}.
The parent spawns one worker per device. Workers: unique-ID bootstrap (rank 0 writes the ID to a file), every
exchange buffer and counter array from nvshmem_malloc (one allocation per kind, sized for the largest case, made in the
same order on every PE), peer tables from nvshmem_ptr, per case: zero, NVSHMEM barrier, launch this GPU's share
(multi_gpu.recurrence_rank / hybrid_bw.mg_rank_kernel), --reps timed calls. Each worker saves its columns of the
first call's outputs. Then the parent, alone on the GPUs, runs the one-process reference (forward: the same
configuration on uva buffers; backward: the single-GPU kernel with the same batch tile) on the same seeded inputs and
compares bitwise. Per call the reported time is the slowest PE's kernel time.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import torch


def forward_inputs(cell, width, device):
    import scale_bench
    return scale_bench.inputs(cell, 16, 48, 1, width, 1201 + width, device)


def worker(rank, devices, out, cases, reps, q):
    import ctypes

    from flashrnn.flashrnn2 import hybrid_bw, multi_gpu, nvshmem_mem
    from flashrnn.flashrnn2.triton_rtiled_bw import preactivations
    import mg_bw_bench
    import scale_bench

    rec = dict(rank=rank, cases=[])
    peers = len(devices)
    try:
        dev = devices[rank]
        torch.cuda.set_device(dev)
        device = f"cuda:{dev}"
        torch.zeros(1, device=device)
        lib = nvshmem_mem.load()
        n = lib.frnn_shm_uid_bytes()
        uid = ctypes.create_string_buffer(n)
        uid_path = Path(out) / "uid.bin"
        if rank == 0:
            if lib.frnn_shm_uid(uid) != 0:
                raise RuntimeError("nvshmemx_get_uniqueid failed")
            tmp = uid_path.with_suffix(".tmp")
            tmp.write_bytes(uid.raw)
            tmp.replace(uid_path)
        else:
            t0 = time.time()
            while not uid_path.exists():
                if time.time() - t0 > 120:
                    raise RuntimeError("no unique ID from rank 0")
                time.sleep(0.05)
            ctypes.memmove(uid, uid_path.read_bytes(), n)
        rc = lib.frnn_shm_init(rank, peers, uid)
        if rc != 0:
            raise RuntimeError(f"nvshmem init failed ({rc})")
        # Sizes: the largest case of each kind (identical on every PE, allocated in the same order).
        fx = fs = bx = 8
        for c in cases:
            if c["kind"] == "forward":
                cfg = scale_bench.parse(c["config"])
                g = multi_gpu.geometry(16, 1, cfg["width"], peers, cfg["slice_width"], cfg["schedule"], c["k_chunk"])
                fx, fs = max(fx, g["xbytes"]), max(fs, g["sbytes"])
            else:
                lay = hybrid_bw.mg_layout(c["width"], c["batch"], 1, peers, c["batch_rows"])
                bx = max(bx, lay["resident"] * 2 * c["batch_rows"] * 4 * c["width"] * 2)
        bufs = {name: lib.frnn_shm_malloc(size) for name, size in (("fx", fx), ("fs", fs), ("bx", bx), ("bs", 64))}
        if not all(bufs.values()):
            raise RuntimeError(f"nvshmem_malloc failed: {bufs}")
        sizes = dict(fx=fx, fs=fs, bx=bx, bs=64)
        tables = {k: torch.tensor([lib.frnn_shm_ptr(v, pe) for pe in range(peers)], device=device, dtype=torch.int64)
                  for k, v in bufs.items()}
        rec["nvshmem_ptr_nonnull"] = {k: bool(all(t.tolist())) for k, t in tables.items()}
        for ci, c in enumerate(cases):
            row = dict(case=c, ms=[])
            try:
                if c["kind"] == "forward":
                    cfg = scale_bench.parse(c["config"])
                    if cfg["peers"] != peers:
                        raise ValueError("forward case P must equal the number of processes")
                    leaves = forward_inputs(c["cell"], cfg["width"], device)

                    def call():
                        for k in ("fx", "fs"):
                            lib.frnn_shm_zero(bufs[k], sizes[k])
                        lib.frnn_shm_barrier()
                        o, f, (e0, e1, err) = multi_gpu.recurrence_rank(
                            *leaves, c["cell"], rank=rank, peers=peers, xtable=tables["fx"], stable=tables["fs"],
                            slice_width=cfg["slice_width"], schedule=cfg["schedule"], stride=cfg["stride"],
                            rmode=cfg["rmode"], k_chunk=c["k_chunk"])
                        torch.cuda.synchronize(dev)
                        return (o, f), e0.elapsed_time(e1), int(err.item())
                    cols = slice(rank * cfg["width"] // peers, (rank + 1) * cfg["width"] // peers)
                else:
                    args = mg_bw_bench.bw_inputs(c["cell"], c["batch"], c["steps"], c["width"], 1301 + c["width"],
                                                 device)
                    wx, r, b, init, history, d_history, d_final, cell = args
                    pre, _ = preactivations(wx, r, b, init, history, False)
                    bm = c["batch_rows"]
                    krw = hybrid_bw.reg_rows(c["width"], bm)

                    def call():
                        for k in ("bx", "bs"):
                            lib.frnn_shm_zero(bufs[k], sizes[k])
                        dg = torch.zeros((c["batch"], c["steps"], 4, 1, c["width"]), device=device, dtype=torch.float32)
                        d_init = torch.zeros(init.shape, device=device, dtype=torch.float32)
                        err = torch.zeros(12, device=device, dtype=torch.int32)
                        own = torch.tensor([dg.data_ptr()] * peers, device=device, dtype=torch.int64)
                        tensors = [pre.contiguous(), history.contiguous(), init.float().contiguous(),
                                   d_history.contiguous(), d_final.float().contiguous(), r.contiguous(), dg, d_init]
                        torch.cuda.synchronize(dev)
                        lib.frnn_shm_barrier()
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        hybrid_bw.mg_rank_kernel(rank, peers, cell, wx.dtype, c["width"], c["batch"], c["steps"], 1,
                                                 bm, krw, tensors, bufs["bx"], tables["bx"], tables["bs"], own, err)
                        e1.record()
                        torch.cuda.synchronize(dev)
                        return (dg, d_init), e0.elapsed_time(e1), int(err[9].item())
                    cols = slice(rank * c["width"] // peers, (rank + 1) * c["width"] // peers)
                first, ms, err = call()
                row["first_error_flag"] = err
                torch.save([x[..., cols].cpu() for x in first], Path(out) / f"case{ci}-rank{rank}.pt")
                for _ in range(reps):
                    _, ms, err = call()
                    row["ms"].append(ms)
                    row.setdefault("error_flags", []).append(err)
            except Exception:  # noqa: BLE001
                row["error"] = traceback.format_exc()[-1500:]
            rec["cases"].append(row)
        lib.frnn_shm_barrier()
        lib.frnn_shm_finalize()
        rec["status"] = "DONE"
    except Exception:  # noqa: BLE001
        rec.update(status="ERROR", error=traceback.format_exc()[-2000:])
    (Path(out) / f"worker{rank}.json").write_text(json.dumps(rec, indent=1, default=str))
    q.put(rank)


def reference(case, devices, peers):
    """One-process reference outputs (and median uva / single-GPU kernel time) for one case."""
    from flashrnn.flashrnn2 import hybrid_bw, multi_gpu
    import mg_bw_bench
    import scale_bench

    dev0 = f"cuda:{devices[0]}"
    if case["kind"] == "forward":
        cfg = scale_bench.parse(case["config"])
        leaves = forward_inputs(case["cell"], cfg["width"], dev0)
        run = lambda: multi_gpu.recurrence(*leaves, case["cell"], devices=devices[:peers],  # noqa: E731
                                           slice_width=cfg["slice_width"], schedule=cfg["schedule"],
                                           stride=cfg["stride"], rmode=cfg["rmode"], k_chunk=case["k_chunk"])
        run()
        out, final, _ = run()
        times = [max(run()[2].values()) for _ in range(5)]
        return [out, final], sorted(times)[2], "uva one process"
    args = mg_bw_bench.bw_inputs(case["cell"], case["batch"], case["steps"], case["width"], 1301 + case["width"], dev0)
    krw = hybrid_bw.reg_rows(case["width"], case["batch_rows"])
    res = hybrid_bw.backward(*args, batch_rows=case["batch_rows"], reg_rows_per_warp=krw)
    times = []
    for _ in range(5):
        diag = {"time": True}
        hybrid_bw.backward(*args, batch_rows=case["batch_rows"], reg_rows_per_warp=krw, diagnostics=diag)
        times.append(diag["ms"]["kernel"])
    return [res[0], res[3]], sorted(times)[2], "single GPU"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--devices", default="0,1,2,3")
    p.add_argument("--cases", required=True)
    p.add_argument("--reps", type=int, default=9)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    devices = [int(x) for x in a.devices.split(",")]
    cases = json.loads(a.cases)
    rec = dict(started=time.time(), devices=devices, cases=cases, reps=a.reps,
               env={k: v for k, v in os.environ.items() if k.startswith("NVSHMEM")})
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, devices, str(out), cases, a.reps, q)) for r in range(len(devices))]
    for pr in procs:
        pr.start()
    deadline = time.time() + 1500
    for pr in procs:
        pr.join(timeout=max(1, deadline - time.time()))
    rec["worker_exit"] = [pr.exitcode for pr in procs]
    for pr in procs:
        if pr.is_alive():
            pr.kill()
            pr.join()
    workers = [json.loads((out / f"worker{r}.json").read_text()) if (out / f"worker{r}.json").exists() else {}
               for r in range(len(devices))]
    rec["workers"] = [{k: w.get(k) for k in ("status", "error", "nvshmem_ptr_nonnull")} for w in workers]
    from flashrnn.flashrnn2 import multi_gpu
    multi_gpu.enable_peer_access(devices)
    rows = []
    for ci, case in enumerate(cases):
        row = dict(case=case)
        try:
            per = [w["cases"][ci] for w in workers]
            errs = [x.get("error") for x in per if x.get("error")]
            if errs:
                raise RuntimeError(errs[0])
            with torch.cuda.device(devices[0]):  # launches on the inputs' GPU (section 14)
                ref, ref_ms, ref_kind = reference(case, devices, len(devices))
            parts = [torch.load(out / f"case{ci}-rank{r}.pt") for r in range(len(devices))]
            got = [torch.cat([parts[r][i] for r in range(len(devices))], dim=-1) for i in range(len(ref))]
            row.update(bitwise=[bool(torch.equal(g.cpu(), x.cpu())) for g, x in zip(got, ref)],
                       max_abs=[(g.cpu() - x.cpu()).abs().max().item() for g, x in zip(got, ref)],
                       error_flags=[x.get("error_flags") for x in per],
                       nvshmem_ms=[max(x["ms"][i] for x in per) for i in range(a.reps)],
                       reference_ms_median=ref_ms, reference=ref_kind)
            row["pass_"] = all(row["bitwise"]) and not any(any(f or []) for f in row["error_flags"])
        except Exception:  # noqa: BLE001
            row.update(error=traceback.format_exc()[-1500:], pass_=False)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("case", "pass_", "bitwise", "reference_ms_median")}), flush=True)
    rec.update(rows=rows, finished=time.time(),
               status="PASS" if rows and all(r["pass_"] for r in rows) else "FAIL")
    (out / "nvshmem.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(dict(status=rec["status"], workers=[w.get("status") for w in workers])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
