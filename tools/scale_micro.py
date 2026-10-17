"""Link microbenchmarks that calibrate the scale-out cost model on one node (docs/flashrnn2/scale-out-plan.md).

  pair      one-way latency alpha and inverse bandwidth beta between every GPU pair: a persistent ping-pong kernel on
            two GPUs (store m bytes into the peer's buffer, release-add the peer's flag, acquire-spin on the own flag),
            m in 16 B .. 1 MiB, N round trips per launch; alpha from m = 16 B, beta from the slope
  hbm       device-memory read bandwidth per GPU (1 GiB copy, median of 7)
  pcie      pinned-host-to-device bandwidth per GPU (512 MiB, median of 7)
  l2        intra-GPU counter-barrier round trip among k CTAs (the R-tiled group barrier, alpha_l2)
Writes micro.json. Usage: scale_micro.py --out DIR [--devices 0,1,2,3] [--allocs uva,nccl_symm] [--warmup 1]
--allocs: the pair benchmark per allocation of its buffers and flags ("uva": PyTorch allocator + peer access;
"nccl_symm": NCCL 2.32 symmetric windows, LSA addresses; results under pair_by_alloc, "pair" stays the uva rows).
--warmup 1: one untimed launch of every (size) kernel on both devices before the timed one (scale-out-plan.md
section 9: the first pair compiled every size inside its timed window).
"""
import argparse
import json
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from flashrnn.flashrnn2.multi_gpu import enable_peer_access

SIZES = [16, 256, 4096, 16384, 65536, 262144, 1048576]


@triton.jit
def _pingpong(PEER_BUF, PEER_FLAG, MY_BUF, MY_FLAG, ERR, n, first, N_BYTES: tl.constexpr, SPIN: tl.constexpr):
    """One CTA per GPU. ``first`` starts; each side stores N_BYTES into the peer buffer, bumps the peer flag,
    then waits for its own flag to reach the next value."""
    dst = tl.load(PEER_BUF).to(tl.pointer_type(tl.int32))
    pflag = tl.load(PEER_FLAG).to(tl.pointer_type(tl.int32))
    mflag = MY_FLAG
    offs = tl.arange(0, N_BYTES // 4)
    for i in range(n):
        if (first != 0) | (i > 0):
            tl.store(dst + offs, offs + i)
            tl.debug_barrier()
            tl.atomic_add(pflag, 1, sem="release", scope="sys")
        if (first == 0) | (i + 1 < n):
            target = i + 1
            seen = tl.atomic_add(mflag, 0, sem="acquire", scope="sys")
            spins = seen * 0
            while (seen < target) & (spins < SPIN):
                seen = tl.atomic_add(mflag, 0, sem="acquire", scope="sys")
                spins += 1
            if seen < target:
                tl.atomic_max(ERR, 1)
            tl.debug_barrier()


class _Raw:
    """A raw device address as a Triton pointer argument (int32 elements)."""
    dtype = torch.int32

    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


def pair_bench(a, b, nbytes, n, alloc="uva", devices=None):
    da, db = f"cuda:{a}", f"cuda:{b}"
    words = max(1, nbytes // 4)
    errs = {a: torch.zeros((1,), device=da, dtype=torch.int32), b: torch.zeros((1,), device=db, dtype=torch.int32)}
    if alloc == "uva":
        bufs = {a: torch.zeros((words + 64,), device=da, dtype=torch.int32),
                b: torch.zeros((words + 64,), device=db, dtype=torch.int32)}
        flags = {a: torch.zeros((8,), device=da, dtype=torch.int32), b: torch.zeros((8,), device=db, dtype=torch.int32)}
    else:
        from flashrnn.flashrnn2 import symm_mem
        sess = symm_mem.session(devices)
        rank = {d: devices.index(d) for d in (a, b)}
        bid, fid = sess.buffer("micro_buf", 4 * (words + 64)), sess.buffer("micro_flag", 4 * 8)
        for d in (a, b):
            torch.cuda.set_device(d)
            sess.zero(bid, rank[d], 4 * (words + 64))
            sess.zero(fid, rank[d], 4 * 8)
        btab = {d: sess.table(bid, rank[d]) for d in (a, b)}
        ftab = {d: sess.table(fid, rank[d]) for d in (a, b)}
        bufs = {d: _Raw(sess.local(bid, rank[d])) for d in (a, b)}
        flags = {d: _Raw(sess.local(fid, rank[d])) for d in (a, b)}
    for d in (a, b):
        torch.cuda.synchronize(d)
    ev = {}
    for d, peer, first in ((a, b, 1), (b, a, 0)):
        torch.cuda.set_device(d)
        if alloc == "uva":
            pb = torch.tensor([bufs[peer].data_ptr()], device=f"cuda:{d}", dtype=torch.int64)
            pf = torch.tensor([flags[peer].data_ptr()], device=f"cuda:{d}", dtype=torch.int64)
        else:  # the peer's LSA address as mapped on this device
            pb = btab[d][rank[peer]:rank[peer] + 1].clone()
            pf = ftab[d][rank[peer]:rank[peer] + 1].clone()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        _pingpong[(1,)](pb, pf, bufs[d], flags[d], errs[d], n, first, max(16, nbytes), 1 << 24, num_warps=4)
        e1.record()
        ev[d] = (e0, e1)
    for d in (a, b):
        torch.cuda.synchronize(d)
    err = int(errs[a][0].item()) | int(errs[b][0].item())
    ms = max(ev[a][0].elapsed_time(ev[a][1]), ev[b][0].elapsed_time(ev[b][1]))
    return ms * 1000.0 / n / 2.0, err  # one-way microseconds per message


def bandwidth(fn, nbytes, reps=7):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ms = sorted(ts)[len(ts) // 2]
    return nbytes / (ms / 1000.0) / 1e9  # GB/s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--devices", default="0,1,2,3")
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--allocs", default="uva")
    p.add_argument("--warmup", type=int, default=0)
    a = p.parse_args()
    devices = [int(x) for x in a.devices.split(",")]
    allocs = a.allocs.split(",")
    rec = dict(devices=devices, started=time.time(), gpu=torch.cuda.get_device_name(devices[0]), pair={}, hbm={},
               pcie={}, errors={}, allocs=allocs, warmup=a.warmup, pair_by_alloc={})
    try:
        rec["peer_probe"] = enable_peer_access(devices)
    except Exception as error:  # noqa: BLE001
        rec["errors"]["peer"] = repr(error)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "micro.json").write_text(json.dumps(rec, indent=1, default=str))
        print(json.dumps(dict(status="PEER_ACCESS_UNAVAILABLE", error=repr(error))))
        return 2
    if "nccl_symm" in allocs:
        try:
            from flashrnn.flashrnn2 import symm_mem
            sess = symm_mem.session(devices)
            rec["symm"] = dict(nccl_version=sess.nccl_version, probe=sess.probe)
        except Exception as error:  # noqa: BLE001 - reported outcome; the uva rows still run
            rec["errors"]["symm_init"] = repr(error)
            allocs = [x for x in allocs if x != "nccl_symm"]
    for alloc in allocs:
        rec["pair_by_alloc"][alloc] = {}
        for i, x in enumerate(devices):
            for y in devices[i + 1:]:
                row = {}
                for nbytes in SIZES:
                    n = a.rounds if nbytes <= 65536 else max(200, a.rounds // 8)
                    if a.warmup:
                        pair_bench(x, y, nbytes, 4, alloc, devices)
                    us, err = pair_bench(x, y, nbytes, n, alloc, devices)
                    row[str(nbytes)] = dict(one_way_us=us, barrier_error=err)
                alpha = row["16"]["one_way_us"]
                big = row[str(SIZES[-1])]["one_way_us"]
                row["alpha_us"] = alpha
                row["beta_us_per_MB"] = (big - alpha) / (SIZES[-1] / 1e6)
                row["bandwidth_GBps_at_1MB"] = SIZES[-1] / (big * 1e-6) / 1e9
                rec["pair_by_alloc"][alloc][f"{x}-{y}"] = row
                if alloc == "uva":
                    rec["pair"][f"{x}-{y}"] = row
                print(json.dumps(dict(alloc=alloc, pair=f"{x}-{y}", alpha_us=round(alpha, 2),
                                      GBps_1MB=round(row["bandwidth_GBps_at_1MB"], 1))), flush=True)
    for d in devices:
        torch.cuda.set_device(d)
        src = torch.empty((1 << 30,), device=f"cuda:{d}", dtype=torch.uint8)
        dst = torch.empty_like(src)
        rec["hbm"][str(d)] = dict(copy_GBps=bandwidth(lambda: dst.copy_(src), 2 * src.numel()),
                                  read_GBps_sum=bandwidth(lambda: src.view(torch.int32).sum(), src.numel()))
        host = torch.empty((1 << 29,), dtype=torch.uint8, pin_memory=True)
        rec["pcie"][str(d)] = dict(h2d_GBps=bandwidth(lambda: dst[: host.numel()].copy_(host, non_blocking=True), host.numel()))
        del src, dst, host
        torch.cuda.empty_cache()
    rec["finished"] = time.time()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "micro.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(dict(status="DONE", hbm={k: round(v["copy_GBps"]) for k, v in rec["hbm"].items()},
                          pcie={k: round(v["h2d_GBps"], 1) for k, v in rec["pcie"].items()})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
