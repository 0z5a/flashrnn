"""NVSwitch multicast probe (docs/flashrnn2/tmem-beta-handoff.md, "Multi-GPU"): run before the multi-GPU cluster forward.

Usage: nvls_probe.py --devices 4,5,6,7 [--block-bytes 8192] [--iters 2000] [--warm 50] --out FILE
Steps: (1) CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED per device; (2) one multicast group over the devices
(nvls_mem.MulticastGroup; prints the granularity); (3) nvls_probe.cu in both modes, one CTA per GPU concurrently:
multicast stores + multicast flag, and unicast fan-out (P stores + P flags). Per mode and rank: mismatched words
(expect 0; a timeout adds 2^40) and nanoseconds per iteration = one all-gather step of P blocks of block-bytes.
The ratio unicast / multicast per step is what the kernel's exchange would gain; the absolute multicast time is the
floor the kernel's remote wait inherits.
"""
import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from flashrnn.flashrnn2 import nvls_mem

SOURCE = Path(__file__).resolve().parents[1] / "flashrnn" / "flashrnn2" / "nvls_probe.cu"


def build(device):
    major, minor = torch.cuda.get_device_capability(device)
    cc = f"{major}{minor}"
    cuda_home = os.environ.get("FRNN_HYBRID_CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    digest = hashlib.sha256(SOURCE.read_bytes() + cc.encode()).hexdigest()[:16]
    cache = Path(os.environ.get("FRNN_HYBRID_CACHE", os.environ.get("TMPDIR", str(Path.home() / ".cache")) + "/frnn_hybrid_bw"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"nvls_probe_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        cmd = [os.path.join(cuda_home, "bin", "nvcc"), "-std=c++17", "-O3", f"-gencode=arch=compute_{cc},code=sm_{cc}",
               "-shared", "-Xcompiler=-fPIC", "-cudart=static", str(SOURCE), "-o", str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(lib)
    l = ctypes.CDLL(str(lib))
    return l


def launch(lib, device, args):
    """One CTA of 256 threads on the device's current stream, through the host shim in nvls_probe.cu.
    args: mc_base, uc_table, flag_off, rank, P, words, iters, warm, mode, out."""
    lib.nvls_probe_launch.restype = ctypes.c_int
    lib.nvls_probe_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 7 + [ctypes.c_void_p] * 2
    mc_base, uc_table, flag_off, rank, P, words, iters, warm, mode, out = args
    return lib.nvls_probe_launch(ctypes.c_void_p(mc_base), ctypes.c_void_p(uc_table), flag_off, rank, P, words, iters,
                                 warm, mode, ctypes.c_void_p(out),
                                 ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", required=True)
    ap.add_argument("--block-bytes", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    devices = [int(x) for x in a.devices.split(",")]
    P = len(devices)
    rec = dict(started=time.time(), devices=devices, block_bytes=a.block_bytes, iters=a.iters,
               gpu=[torch.cuda.get_device_name(d) for d in devices], supported={}, modes={})
    for d in devices:
        rec["supported"][d] = nvls_mem.multicast_supported(d)
        print(f"cuda:{d} {torch.cuda.get_device_name(d)} multicast_supported={rec['supported'][d]}", flush=True)
    if not all(rec["supported"].values()):
        Path(a.out).write_text(json.dumps(rec, indent=1))
        print("PROBE unsupported")
        return
    words = a.block_bytes // 4
    data_bytes = 2 * P * a.block_bytes
    flag_off = (data_bytes + 255) // 256 * 256
    grp = nvls_mem.MulticastGroup(devices, flag_off + 2 * P * 4)
    rec["granularity"] = grp.granularity
    rec["size"] = grp.size
    print(f"multicast group: granularity {grp.granularity} bytes, size {grp.size}", flush=True)
    lib = build(devices[0])
    table = torch.tensor([grp.uc(i) for i in range(P)], dtype=torch.int64)
    for mode, name in ((0, "multicast"), (1, "unicast")):
        grp.zero()
        outs, tables = [], []
        for rank, d in enumerate(devices):
            dev = torch.device(f"cuda:{d}")
            with torch.cuda.device(dev):
                tbl = table.to(dev)
                out = torch.zeros(2, dtype=torch.int64, device=dev)
                tables.append(tbl)
                outs.append(out)
                rc = launch(lib, dev, [grp.mc(0), tbl.data_ptr(), flag_off, rank, P, words, a.iters, a.warm, mode,
                                       out.data_ptr()])
                if rc != 0:
                    raise RuntimeError(f"probe launch failed on cuda:{d} (cudaError {rc})")
        for d in devices:
            torch.cuda.synchronize(d)
        res = []
        for rank, out in enumerate(outs):
            bad, ns = out.tolist()
            timeout = bad >> 40
            res.append(dict(rank=rank, device=devices[rank], mismatches=bad & ((1 << 40) - 1), timeouts=timeout,
                            ns_per_iter=ns / a.iters))
            print(f"{name:9s} rank {rank} cuda:{devices[rank]}: mismatches {bad & ((1 << 40) - 1)} timeouts {timeout} "
                  f"{ns / a.iters:8.1f} ns/iter", flush=True)
        rec["modes"][name] = res
    if "multicast" in rec["modes"] and "unicast" in rec["modes"]:
        m = max(r["ns_per_iter"] for r in rec["modes"]["multicast"])
        u = max(r["ns_per_iter"] for r in rec["modes"]["unicast"])
        rec["unicast_over_multicast"] = u / m if m else None
        print(f"step latency (slowest rank): multicast {m:.0f} ns, unicast fan-out {u:.0f} ns, ratio {u / m:.2f}")
    rec["finished"] = time.time()
    Path(a.out).write_text(json.dumps(rec, indent=1))
    ok = all(r["mismatches"] == 0 and r["timeouts"] == 0 for v in rec["modes"].values() for r in v)
    print("PROBE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
