"""BETA, UNVERIFIED: the cluster forward with the GEMM on Blackwell tensor cores (hybrid_fwc_tmem.cu, tcgen05 with the
accumulator in tensor memory). sm_100a / sm_103a only; D = 512 (and 256) with a 16-CTA cluster, B <= 16 per cluster.
See docs/flashrnn2/tmem-beta-handoff.md. Same call contract as hybrid_fwc.recurrence.

recurrence_mg: the same kernel over P GPUs of one NVSwitch domain (FRNN_P > 1), the units split across GPUs and
round(h) exchanged through NVSwitch multicast (nvls_mem.MulticastGroup); see the handoff, "Multi-GPU"."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("hybrid_fwc_tmem.cu")
MODES = {"full": 0, "hidden": 1, "last": 2}
SMEM_LIMIT = 227 * 1024
_LIBS = {}
_FLAG_EPOCH = [0]


def _geometry(width, cluster, peers=1):
    if width % (peers * cluster):
        return None
    nu = width // (peers * cluster)
    nc = 4 * nu
    smem = 2 * 16 * width * 2 + nc * width * 2 + 16 * nc * 4
    return dict(nu=nu, nc=nc, smem=smem)


def supported(width, cluster=16, peers=1):
    g = _geometry(width, cluster, peers)
    return g is not None and g["nu"] > 0 and g["nu"] % 16 == 0 and g["nc"] <= 256 and g["smem"] <= SMEM_LIMIT


def default_cluster(width, peers):
    """The largest cluster (<= 16) that gives NU = D / (P C) a multiple of 16 and fits shared memory (None: none)."""
    for c in (16, 8, 4, 2):
        if supported(width, c, peers):
            return c
    return None


def _gencode(arch):
    """nvcc flags for an arch such as sm_100a: the -gencode form, so that no plain compute_* PTX (which cannot hold
    tcgen05) is embedded (B300 session, nvcc 13.3)."""
    if arch.startswith("sm_"):
        a = arch[3:]
        return [f"-gencode=arch=compute_{a},code=sm_{a}"]
    return [f"-arch={arch}"]


def build(width, cluster, slstm, bf16, mode, snap32, verify=0, arch=None, peers=1, cache_dir=None):
    arch = arch or os.environ.get("FRNN_TMEM_ARCH", "sm_100a")
    cuda_home = os.environ.get("FRNN_HYBRID_CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    flags = [f"-DFRNN_D={width}", f"-DFRNN_C={cluster}", f"-DFRNN_SLSTM={int(slstm)}", f"-DFRNN_BF16={int(bf16)}",
             f"-DFRNN_MODE={int(mode)}", f"-DFRNN_SNAP32={int(snap32)}", f"-DFRNN_VERIFY={int(verify)}",
             f"-DFRNN_P={int(peers)}"]
    digest = hashlib.sha256(SOURCE.read_bytes() + " ".join(flags + [arch]).encode()).hexdigest()[:16]
    cache = Path(cache_dir or os.environ.get("FRNN_HYBRID_CACHE",
                                             os.environ.get("TMPDIR", str(Path.home() / ".cache")) + "/frnn_hybrid_bw"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"hybrid_fwc_tmem_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        cmd = [os.path.join(cuda_home, "bin", "nvcc"), "-std=c++17", "-O3", *_gencode(arch), "-shared",
               "-Xcompiler=-fPIC", "-cudart=static", *flags, str(SOURCE), "-o", str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(lib)
    return lib


def _lib(*key):
    if key not in _LIBS:
        lib = ctypes.CDLL(str(build(*key)))
        lib.frnn_hybrid_fwc_tmem.restype = ctypes.c_int
        lib.frnn_hybrid_fwc_tmem.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 5
                                             + [ctypes.c_int] * 2)
        lib.frnn_hybrid_fwc_tmem_smem.restype = ctypes.c_int
        lib.frnn_hybrid_fwc_tmem_xbytes.restype = ctypes.c_int
        lib.frnn_hybrid_fwc_tmem_peers.restype = ctypes.c_int
        _LIBS[key] = lib
    return _LIBS[key]


def _check(wx, recurrent, bias, initial, cell, sequence_context, output_mode, snapshot_dtype):
    from .state_contract import state_contract

    if cell not in ("lstm", "slstm") or output_mode not in MODES:
        raise ValueError("cell lstm/slstm; output mode full/hidden/last")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("CUDA FP16/BF16 inputs")
    contract = state_contract(cell, wx.dtype)
    if sequence_context is None or sequence_context.contract != contract or sequence_context.step_offset > 0:
        raise ValueError("a matching sequence context at the sequence start")
    if snapshot_dtype not in (None, wx.dtype, torch.float32):
        raise ValueError("snapshot dtype must be input dtype or FP32")
    batch, steps, gates, heads, width = wx.shape
    states = len(contract.components)
    if gates != 4 or steps < 1:
        raise ValueError("expected [B, T, 4, H, D] inputs")
    if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
        raise ValueError("expected public weight and bias layouts")
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("expected public initial state layout")
    return batch, steps, heads, width, states


def _launch(lib, tensors, zero_flag, batch, steps, heads, groups, tiles, device, mc_x=0, uc_x=0, mc_flag=0, uc_flag=0,
            rank=0, flag_base=0):
    rc = lib.frnn_hybrid_fwc_tmem(*[ctypes.c_void_p(x.data_ptr()) for x in tensors],
                                  ctypes.c_void_p(zero_flag.data_ptr()) if zero_flag is not None else ctypes.c_void_p(0),
                                  batch, steps, heads, groups, tiles,
                                  ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream),
                                  ctypes.c_void_p(mc_x), ctypes.c_void_p(uc_x), ctypes.c_void_p(mc_flag),
                                  ctypes.c_void_p(uc_flag), rank, flag_base)
    if rc != 0:
        raise RuntimeError(f"tensor-memory cluster forward launch failed (cudaError {rc})")


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, sequence_context, output_mode="hidden",
               snapshot_dtype=None, cluster=16, arch=None, prof=False, diagnostics=None):
    batch, steps, heads, width, states = _check(wx, recurrent, bias, initial, cell, sequence_context, output_mode,
                                                snapshot_dtype)
    if not supported(width, cluster):
        raise ValueError(f"unsupported shape (D = {width}, cluster {cluster})")
    device = wx.device
    tiles = (batch + 15) // 16
    groups = tiles * heads
    snap = wx.dtype if snapshot_dtype is None else snapshot_dtype
    out_shape = {"full": (states, batch, steps, heads, width), "hidden": (batch, steps, heads, width),
                 "last": (batch, 1, heads, width)}[output_mode]
    output = torch.empty(out_shape, device=device, dtype=snap)
    final = torch.empty(initial.shape, device=device, dtype=torch.float32)
    err = torch.zeros(12, device=device, dtype=torch.int32)
    lib = _lib(width, cluster, cell == "slstm", wx.dtype == torch.bfloat16, MODES[output_mode], snap == torch.float32,
               16 if prof else 0, arch, 1)
    zero_flag = (sequence_context.zero_normalizer.to(device=device, dtype=torch.bool).contiguous()
                 if cell == "slstm" else None)
    tensors = [wx.contiguous(), recurrent.contiguous(), bias.contiguous(), initial.float().contiguous(), output, final,
               err]
    _launch(lib, tensors, zero_flag, batch, steps, heads, groups, tiles, device)
    if prof and diagnostics is not None:
        c = err.tolist()
        n = max(c[9], 1)
        diagnostics["phases_us_per_step"] = dict(barrier=c[4] * 64 / 1000 / n, gemm=c[5] * 64 / 1000 / n,
                                                 pointwise=c[6] * 64 / 1000 / n, cta_steps=c[9])
    if diagnostics is not None:
        diagnostics.update(cluster=cluster, groups=groups, shared_bytes=lib.frnn_hybrid_fwc_tmem_smem())
    return output, final


def recurrence_mg(wx, recurrent, bias, initial, cell="lstm", *, sequence_context, devices, output_mode="hidden",
                  snapshot_dtype=None, cluster=None, arch=None, prof=False, diagnostics=None):
    """The cluster forward over P = len(devices) GPUs of one NVSwitch domain (one process). GPU p computes units
    [p D / P, (p + 1) D / P); round(h) is exchanged through a multicast-mapped buffer (nvls_mem). Inputs may live on
    any device (they are copied to every GPU); the outputs are returned on devices[0]. Synchronizes all P devices
    before returning. Whole logical calls only (sequence start)."""
    from . import nvls_mem

    devices = [int(d) for d in devices]
    peers = len(devices)
    if peers < 2 or len(set(devices)) != peers:
        raise ValueError("recurrence_mg needs at least two distinct devices")
    batch, steps, heads, width, states = _check(wx, recurrent, bias, initial, cell, sequence_context, output_mode,
                                                snapshot_dtype)
    cluster = default_cluster(width, peers) if cluster is None else cluster
    if cluster is None or not supported(width, cluster, peers):
        raise ValueError(f"unsupported shape for {peers} GPUs (D = {width}, cluster {cluster})")
    tiles = (batch + 15) // 16
    groups = tiles * heads
    snap = wx.dtype if snapshot_dtype is None else snapshot_dtype
    out_shape = {"full": (states, batch, steps, heads, width), "hidden": (batch, steps, heads, width),
                 "last": (batch, 1, heads, width)}[output_mode]
    lib = _lib(width, cluster, cell == "slstm", wx.dtype == torch.bfloat16, MODES[output_mode], snap == torch.float32,
               16 if prof else 0, arch, peers)
    xbytes = lib.frnn_hybrid_fwc_tmem_xbytes()
    flag_off = (xbytes + 255) // 256 * 256
    flag_bytes = 2 * peers * cluster * 4
    grp = nvls_mem.group(devices, flag_off + flag_bytes)
    # Flags carry flag_base + step + 1 and rise across calls; reset the flag words when the int32 range runs out.
    flag_base = _FLAG_EPOCH[0] * (steps + 2)
    if flag_base + steps + 2 >= 2 ** 31 - 1:
        for d in devices:
            torch.cuda.synchronize(d)
        grp.zero()
        _FLAG_EPOCH[0] = 0
        flag_base = 0
    _FLAG_EPOCH[0] += 1
    per = []
    for rank, d in enumerate(devices):
        dev = torch.device(f"cuda:{d}")
        with torch.cuda.device(dev):
            t = dict(wx=wx.to(dev).contiguous(), r=recurrent.to(dev).contiguous(), b=bias.to(dev).contiguous(),
                     init=initial.to(dev).float().contiguous(),
                     out=torch.zeros(out_shape, device=dev, dtype=snap),
                     final=torch.zeros(initial.shape, device=dev, dtype=torch.float32),
                     err=torch.zeros(12, device=dev, dtype=torch.int32))
            t["zero"] = (sequence_context.zero_normalizer.to(device=dev, dtype=torch.bool).contiguous()
                         if cell == "slstm" else None)
            per.append(t)
    for rank, d in enumerate(devices):
        dev = torch.device(f"cuda:{d}")
        t = per[rank]
        with torch.cuda.device(dev):
            _launch(lib, [t["wx"], t["r"], t["b"], t["init"], t["out"], t["final"], t["err"]], t["zero"], batch, steps,
                    heads, groups, tiles, dev, mc_x=grp.mc(0), uc_x=grp.uc(rank, 0), mc_flag=grp.mc(flag_off),
                    uc_flag=grp.uc(rank, flag_off), rank=rank, flag_base=flag_base)
    for d in devices:
        torch.cuda.synchronize(d)
    errs = [t["err"].tolist() for t in per]
    if any(e[1] for e in errs):
        raise RuntimeError(f"multi-GPU exchange timed out (err[1] per rank: {[e[1] for e in errs]})")
    dev0 = torch.device(f"cuda:{devices[0]}")
    output, final = per[0]["out"], per[0]["final"]
    dp = width // peers
    for rank in range(1, peers):
        sl = slice(rank * dp, (rank + 1) * dp)
        output[..., sl] = per[rank]["out"][..., sl].to(dev0)
        final[..., sl] = per[rank]["final"][..., sl].to(dev0)
    if diagnostics is not None:
        diagnostics.update(cluster=cluster, peers=peers, devices=devices, groups=groups,
                           shared_bytes=lib.frnn_hybrid_fwc_tmem_smem(), exchange_bytes=xbytes,
                           multicast_granularity=grp.granularity)
        if prof:
            phases = []
            for c in errs:
                n = max(c[9], 1)
                phases.append(dict(barrier=c[4] * 64 / 1000 / n, remote=c[7] * 64 / 1000 / n, gemm=c[5] * 64 / 1000 / n,
                                   pointwise=c[6] * 64 / 1000 / n, cta_steps=c[9]))
            diagnostics["phases_us_per_step"] = phases[0]
            diagnostics["phases_per_rank"] = phases
    return output, final
