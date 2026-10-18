"""Large-D backward (hybrid_bw.cu) through ctypes: register + shared-memory residency of R, D up to 2048.

Built per (D, register rows per warp, cell, dtype) as a standalone shared object with the CUDA 12.x toolkit
(sm_90a, static cudart), independent of PyTorch's CUDA version; tensors cross as raw device pointers on PyTorch's
current stream (the ThunderKittens path's pattern). Contract and arithmetic: the plain persistent variant of
triton_rtiled_bw.backward (same preactivation recompute and dR / dbias GEMMs, which are reused from there).
"""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("hybrid_bw.cu")
S, WARPS, SMEM_LIMIT = 16, 8, 232448
_LIBS = {}


def reg_rows(width, bm=16):
    """Register rows per warp: the fewest (multiple of 16) for which the shared-memory part of R fits."""
    kw = 4 * width // WARPS
    for krw in range(0, kw + 1, 16):
        if smem_bytes(width, krw, bm) <= SMEM_LIMIT:
            return krw
    raise ValueError("no register split fits")


def smem_bytes(width, krw, bm=16):
    kw = 4 * width // WARPS
    return WARPS * (kw - krw) * 16 * 2 + WARPS * bm * 16 * 4  # shared rows of R (16 columns, 2 bytes) + partial sums


def batch_tile(batch):
    """Batch rows per CTA: one tile for the whole batch up to 64 rows (fewer serialized groups at large D)."""
    return 16 if batch <= 16 else 32 if batch <= 32 else 64


def supported(width, sms=132):
    return width % 16 == 0 and (4 * width // WARPS) % 16 == 0 and 2 <= width // S <= sms and width >= 256


VERIFY_BITS = dict(check=1, delay=2, single=4, no_sync=8, prof=16, cl_single=32, cl_no_sync=64, mg_no_wait=128)


def verify_bits(verify):
    """verify: None or {"check": bool, "delay_seed": int | None,
    "broken": None | "single" | "no_sync" | "cl_single" | "cl_no_sync"} (the last two: cluster stage, one buffer /
    no barrier between rounds)."""
    if not verify:
        return 0, 0
    bits = VERIFY_BITS["check"] if verify.get("check", True) else 0
    if verify.get("delay_seed") is not None:
        bits |= VERIFY_BITS["delay"]
    if verify.get("broken"):
        bits |= VERIFY_BITS[verify["broken"]]
    return bits, int(verify.get("delay_seed") or 0)


def build(width, krw, slstm, bf16, cache_dir=None, verify=0, bm=16, cluster=0, mg=False):
    cuda_home = os.environ.get("FRNN_HYBRID_CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    flags = [f"-DFRNN_D={width}", f"-DFRNN_KRW={krw}", f"-DFRNN_SLSTM={int(slstm)}", f"-DFRNN_BF16={int(bf16)}",
             f"-DFRNN_VERIFY={int(verify)}", f"-DFRNN_BM={int(bm)}"]
    if cluster:
        flags.append(f"-DFRNN_CLUSTER={int(cluster)}")
    if mg:
        flags.append("-DFRNN_MG=1")
    digest = hashlib.sha256(SOURCE.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    cache = Path(cache_dir or os.environ.get("FRNN_HYBRID_CACHE",
                                             os.environ.get("TMPDIR", str(Path.home() / ".cache")) + "/frnn_hybrid_bw"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"hybrid_bw_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        cmd = [nvcc, "-std=c++17", "-O3", "-arch=sm_90a", "-shared", "-Xcompiler=-fPIC", "-cudart=static",
               *flags, str(SOURCE), "-o", str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(lib)
    return lib


def _lib(width, krw, slstm, bf16, verify=0, bm=16, cluster=0, mg=False):
    key = (width, krw, slstm, bf16, verify, bm, cluster, mg)
    if key not in _LIBS:
        lib = ctypes.CDLL(str(build(width, krw, slstm, bf16, verify=verify, bm=bm, cluster=cluster, mg=mg)))
        if mg:
            lib.frnn_hybrid_bw_mg.restype = ctypes.c_int
            lib.frnn_hybrid_bw_mg.argtypes = ([ctypes.c_void_p] * 10 + [ctypes.c_int] * 7 + [ctypes.c_void_p] * 3
                                              + [ctypes.c_int] * 2 + [ctypes.c_void_p])
        lib.frnn_hybrid_bw.restype = ctypes.c_int
        lib.frnn_hybrid_bw.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
        lib.frnn_hybrid_bw_smem.restype = ctypes.c_int
        lib.frnn_hybrid_bw_max_clusters.restype = ctypes.c_int
        _LIBS[key] = lib
    return _LIBS[key]


def backward(wx, recurrent, bias, initial, history, d_history, d_final, cell, *, reg_rows_per_warp=None, tf32=False,
             diagnostics=None, batch_rows=None, wgrad="bmm", prof=False, cluster=0, _verify=None):
    """Returns (dg FP32 [B,T,4,H,D], dR, dbias, d_initial FP32); d_history / d_final as in triton_rtiled_bw.

    cluster: 0, or 8 for the distributed-shared-memory exchange (clusters of 8 CTAs share each staged dG chunk).
    _verify (race verification only, never served): see verify_bits; the violation counters are written back into
    the dict (after_read, after_mma)."""
    from .triton_rtiled_bw import preactivations, weight_gradients

    if cell not in ("lstm", "slstm"):
        raise ValueError("the hybrid backward supports LSTM and sLSTM")
    batch, steps, gates, heads, width = wx.shape
    if wx.dtype not in (torch.float16, torch.bfloat16) or not supported(width):
        raise ValueError("unsupported shape or dtype for the hybrid backward")
    if cluster not in (0, WARPS) or (cluster and (width // S) % cluster):
        raise ValueError("cluster must be 0, or 8 with D / 16 a multiple of 8")
    device = wx.device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    bm = batch_tile(batch) if batch_rows is None else batch_rows
    krw = reg_rows(width, bm) if reg_rows_per_warp is None else reg_rows_per_warp
    k = width // S
    tiles = (batch + bm - 1) // bm
    groups = tiles * heads
    resident = max(1, min(groups, sms // k))
    slots = (groups + resident - 1) // resident
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if (diagnostics or {}).get("time") else None
    if events:
        events[0].record()
    pre, h_prev = preactivations(wx, recurrent, bias, initial, history, tf32)
    if events:
        events[1].record()
    dg = torch.empty((batch, steps, 4, heads, width), device=device, dtype=torch.float32)
    d_init = torch.empty(initial.shape, device=device, dtype=torch.float32)
    xbuf = torch.empty((resident, 2, bm, 4 * width), device=device, dtype=wx.dtype)
    init32 = initial.float().contiguous()
    bits, seed = verify_bits(_verify)
    if prof:
        bits |= VERIFY_BITS["prof"]
    lib = _lib(width, krw, cell == "slstm", wx.dtype == torch.bfloat16, bits, bm, cluster)
    if cluster:  # cap the co-resident groups by the clusters the GPU can hold at once
        fit = lib.frnn_hybrid_bw_max_clusters() * cluster // k
        if fit < 1:
            raise RuntimeError("no group of clusters fits on the GPU")
        resident = min(resident, fit)
        slots = (groups + resident - 1) // resident
    err = torch.zeros(12, device=device, dtype=torch.int32)
    err[3] = seed
    tensors = [pre.contiguous(), history.contiguous(), init32, d_history.contiguous(), d_final.float().contiguous(),
               recurrent.contiguous(), dg, d_init, xbuf, err]
    rc = lib.frnn_hybrid_bw(*[ctypes.c_void_p(x.data_ptr()) for x in tensors], batch, steps, heads, groups, resident,
                            slots, tiles, ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if rc != 0:
        raise RuntimeError(f"hybrid backward launch failed (cudaError {rc})")
    if events:
        events[2].record()
    if _verify is not None:
        counts = err.tolist()
        _verify.update(after_read=counts[1], after_mma=counts[2], after_stage=counts[8])
    d_recurrent, d_bias = weight_gradients(dg, h_prev, tf32, wgrad)
    if events:
        events[3].record()
        torch.cuda.synchronize()
        diagnostics["ms"] = dict(preactivation=events[0].elapsed_time(events[1]),
                                 kernel=events[1].elapsed_time(events[2]),
                                 weight_gradients=events[2].elapsed_time(events[3]))
    if prof and diagnostics is not None:
        c = err.tolist()
        steps = max(c[7], 1)
        diagnostics["phases_us_per_step"] = dict(reduction=c[4] * 64 / 1000 / steps, pointwise_publish=c[5] * 64 / 1000 / steps,
                                                 grid_barrier=c[6] * 64 / 1000 / steps, cta_steps=c[7])
    if diagnostics is not None:
        diagnostics.update(reg_rows_per_warp=krw, batch_rows=bm, shared_bytes=lib.frnn_hybrid_bw_smem(), ctas=k * resident,
                           groups=groups, resident=resident, slots=slots)
    return dg, d_recurrent, d_bias, d_init


def mg_layout(width, batch, heads, peers, bm, sms=132):
    """Per-GPU grid of the multi-GPU backward: CTAs per group (D / 16 / P), resident groups, slots."""
    if (width // S) % peers:
        raise ValueError("D / 16 must be a multiple of the GPU count")
    k = width // S // peers
    tiles = (batch + bm - 1) // bm
    groups = tiles * heads
    resident = max(1, min(groups, sms // k))
    return dict(k=k, tiles=tiles, groups=groups, resident=resident, slots=(groups + resident - 1) // resident)


def mg_rank_kernel(rank, peers, cell, dtype, width, batch, steps, heads, bm, krw, tensors, xbuf_ptr, xtable, stable,
                   dgtable, err, verify_bits=0, stream=None):
    """Launch one GPU's share of the multi-GPU backward kernel (current device). ``tensors``: pre, history, initial32,
    d_history, d_final32, R, dg, d_init (that GPU's copies); tables: int64 device tensors with one entry per GPU."""
    lay = mg_layout(width, batch, heads, peers, bm)
    lib = _lib(width, krw, cell == "slstm", dtype == torch.bfloat16, verify_bits, bm, 0, True)
    ptr = lambda x: ctypes.c_void_p(x if isinstance(x, int) else x.data_ptr())  # noqa: E731
    stream = stream if stream is not None else torch.cuda.current_stream().cuda_stream
    rc = lib.frnn_hybrid_bw_mg(*[ptr(x) for x in tensors], ptr(xbuf_ptr), ptr(err), batch, steps, heads,
                               lay["groups"], lay["resident"], lay["slots"], lay["tiles"], ptr(xtable), ptr(stable),
                               ptr(dgtable), rank, peers, ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"multi-GPU hybrid backward launch failed on rank {rank} (cudaError {rc})")
    return lay


def backward_mg(wx, recurrent, bias, initial, history, d_history, d_final, cell, *, devices, alloc="uva",
                batch_rows=None, reg_rows_per_warp=None, wgrad="bmm", diagnostics=None, _verify=None):
    """Multi-GPU backward, one process: the D / 16 column slices split over ``devices``; exchange buffers and arrival
    counters from the PyTorch allocator ("uva", peer access enabled) or NCCL symmetric windows ("nccl_symm").
    Returns what ``backward`` returns, on the first device. The kernel time is that of the slowest GPU."""
    from .triton_rtiled_bw import preactivations, weight_gradients

    peers = len(devices)
    batch, steps, gates, heads, width = wx.shape
    bm = batch_tile(batch) if batch_rows is None else batch_rows
    krw = reg_rows(width, bm) if reg_rows_per_warp is None else reg_rows_per_warp
    lay = mg_layout(width, batch, heads, peers, bm)
    dtype = wx.dtype
    first = f"cuda:{devices[0]}"
    pre, h_prev = preactivations(wx, recurrent, bias, initial, history, False)
    shared = [pre.contiguous(), history.contiguous(), initial.float().contiguous(), d_history.contiguous(),
              d_final.float().contiguous(), recurrent.contiguous()]
    per = {}
    for d in devices:
        dev = f"cuda:{d}"
        per[d] = dict(inputs=[x.to(dev) for x in shared],
                      dg=torch.zeros((batch, steps, 4, heads, width), device=dev, dtype=torch.float32),
                      d_init=torch.zeros(initial.shape, device=dev, dtype=torch.float32),
                      err=torch.zeros(12, device=dev, dtype=torch.int32))
    bits, seed = verify_bits(_verify)
    for d in devices:
        per[d]["err"][3] = seed
    xbytes = lay["resident"] * 2 * bm * 4 * width * wx.element_size()
    if alloc == "uva":
        for d in devices:
            per[d]["xbuf"] = torch.zeros(xbytes // wx.element_size(), device=f"cuda:{d}", dtype=dtype)
            per[d]["sync"] = torch.zeros(8, device=f"cuda:{d}", dtype=torch.int32)
        xaddr = [per[d]["xbuf"].data_ptr() for d in devices]
        saddr = [per[d]["sync"].data_ptr() for d in devices]
    elif alloc == "nccl_symm":
        from . import symm_mem
        sess = symm_mem.session(devices)
        xid, sid = sess.buffer("bw_xbuf", xbytes), sess.buffer("bw_sync", 32)
        for r, d in enumerate(devices):
            torch.cuda.set_device(d)
            sess.zero(xid, r, xbytes)
            sess.zero(sid, r, 32)
    else:
        raise ValueError("alloc uva or nccl_symm (NVSHMEM runs one process per GPU: tools/scale_nvshmem.py)")
    dgaddr = [per[d]["dg"].data_ptr() for d in devices]
    for d in devices:
        torch.cuda.synchronize(d)
    events = {}
    for r, d in enumerate(devices):
        torch.cuda.set_device(d)
        dev = f"cuda:{d}"
        if alloc == "uva":
            xt = torch.tensor(xaddr, device=dev, dtype=torch.int64)
            st = torch.tensor(saddr, device=dev, dtype=torch.int64)
            xlocal = per[d]["xbuf"].data_ptr()
        else:
            xt, st = sess.table(xid, r)[:peers].clone(), sess.table(sid, r)[:peers].clone()
            xlocal = sess.local(xid, r)
        dt = torch.tensor(dgaddr, device=dev, dtype=torch.int64)
        per[d]["tables"] = (xt, st, dt)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        mg_rank_kernel(r, peers, cell, dtype, width, batch, steps, heads, bm, krw,
                       per[d]["inputs"] + [per[d]["dg"], per[d]["d_init"]], xlocal, xt, st, dt, per[d]["err"], bits)
        e1.record()
        events[d] = (e0, e1)
    for d in devices:
        torch.cuda.synchronize(d)
    ms = {d: events[d][0].elapsed_time(events[d][1]) for d in devices}
    errs = {d: per[d]["err"].tolist() for d in devices}
    if _verify is not None:
        _verify.update(after_read=sum(e[1] for e in errs.values()), after_mma=sum(e[2] for e in errs.values()),
                       after_stage=0, wait_timeouts=sum(e[9] for e in errs.values()))
    elif any(e[9] for e in errs.values()):
        raise RuntimeError(f"multi-GPU backward: cross-GPU wait timed out {errs}")
    torch.cuda.set_device(devices[0])
    dg = per[devices[0]]["dg"].clone()
    d_init = per[devices[0]]["d_init"].clone()
    cols = width // peers
    for r, d in enumerate(devices[1:], start=1):
        sl = slice(r * cols, (r + 1) * cols)
        dg[..., sl] = per[d]["dg"][..., sl].to(first)
        d_init[..., sl] = per[d]["d_init"][..., sl].to(first)
    d_recurrent, d_bias = weight_gradients(dg, h_prev, False, wgrad)
    if diagnostics is not None:
        diagnostics.update(alloc=alloc, ms=ms, kernel_ms=max(ms.values()), batch_rows=bm, reg_rows_per_warp=krw, **lay)
    return dg, d_recurrent, d_bias, d_init
