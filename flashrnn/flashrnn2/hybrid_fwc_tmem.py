"""BETA, UNVERIFIED: the cluster forward with the GEMM on Blackwell tensor cores (hybrid_fwc_tmem.cu, tcgen05 with the
accumulator in tensor memory). sm_100a / sm_103a only; D = 512 (and 256) with a 16-CTA cluster, B <= 16 per cluster.
See docs/flashrnn2/tmem-beta-handoff.md. Same call contract as hybrid_fwc.recurrence."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("hybrid_fwc_tmem.cu")
MODES = {"full": 0, "hidden": 1, "last": 2}
_LIBS = {}


def supported(width, cluster=16):
    nu = width // cluster if width % cluster == 0 else 0
    nc = 4 * nu
    smem = 2 * 16 * width * 2 + nc * width * 2 + 16 * nc * 4
    return nu > 0 and nu % 16 == 0 and nc <= 256 and smem <= 227 * 1024


def build(width, cluster, slstm, bf16, mode, snap32, verify=0, arch=None, cache_dir=None):
    arch = arch or os.environ.get("FRNN_TMEM_ARCH", "sm_100a")
    cuda_home = os.environ.get("FRNN_HYBRID_CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    flags = [f"-DFRNN_D={width}", f"-DFRNN_C={cluster}", f"-DFRNN_SLSTM={int(slstm)}", f"-DFRNN_BF16={int(bf16)}",
             f"-DFRNN_MODE={int(mode)}", f"-DFRNN_SNAP32={int(snap32)}", f"-DFRNN_VERIFY={int(verify)}"]
    digest = hashlib.sha256(SOURCE.read_bytes() + " ".join(flags + [arch]).encode()).hexdigest()[:16]
    cache = Path(cache_dir or os.environ.get("FRNN_HYBRID_CACHE",
                                             os.environ.get("TMPDIR", str(Path.home() / ".cache")) + "/frnn_hybrid_bw"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"hybrid_fwc_tmem_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        cmd = [os.path.join(cuda_home, "bin", "nvcc"), "-std=c++17", "-O3", f"-arch={arch}", "-shared",
               "-Xcompiler=-fPIC", "-cudart=static", *flags, str(SOURCE), "-o", str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(lib)
    return lib


def _lib(*key):
    if key not in _LIBS:
        lib = ctypes.CDLL(str(build(*key)))
        lib.frnn_hybrid_fwc_tmem.restype = ctypes.c_int
        lib.frnn_hybrid_fwc_tmem.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        lib.frnn_hybrid_fwc_tmem_smem.restype = ctypes.c_int
        _LIBS[key] = lib
    return _LIBS[key]


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, sequence_context, output_mode="hidden",
               snapshot_dtype=None, cluster=16, arch=None, prof=False, diagnostics=None):
    from .state_contract import state_contract

    if cell not in ("lstm", "slstm") or output_mode not in MODES:
        raise ValueError("cell lstm/slstm; output mode full/hidden/last")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("CUDA FP16/BF16 inputs")
    contract = state_contract(cell, wx.dtype)
    if sequence_context is None or sequence_context.contract != contract or sequence_context.step_offset > 0:
        raise ValueError("a matching sequence context at the sequence start")
    batch, steps, gates, heads, width = wx.shape
    states = len(contract.components)
    if gates != 4 or steps < 1 or not supported(width, cluster):
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
               16 if prof else 0, arch)
    zero_flag = (sequence_context.zero_normalizer.to(device=device, dtype=torch.bool).contiguous()
                 if cell == "slstm" else None)
    tensors = [wx.contiguous(), recurrent.contiguous(), bias.contiguous(), initial.float().contiguous(), output, final,
               err]
    rc = lib.frnn_hybrid_fwc_tmem(*[ctypes.c_void_p(x.data_ptr()) for x in tensors],
                                  ctypes.c_void_p(zero_flag.data_ptr()) if zero_flag is not None else ctypes.c_void_p(0),
                                  batch, steps, heads, groups, tiles,
                                  ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if rc != 0:
        raise RuntimeError(f"tensor-memory cluster forward launch failed (cudaError {rc})")
    if prof and diagnostics is not None:
        c = err.tolist()
        n = max(c[9], 1)
        diagnostics["phases_us_per_step"] = dict(barrier=c[4] * 64 / 1000 / n, gemm=c[5] * 64 / 1000 / n,
                                                 pointwise=c[6] * 64 / 1000 / n, cta_steps=c[9])
    if diagnostics is not None:
        diagnostics.update(cluster=cluster, groups=groups, shared_bytes=lib.frnn_hybrid_fwc_tmem_smem())
    return output, final
