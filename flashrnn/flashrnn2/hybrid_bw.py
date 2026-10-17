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


def reg_rows(width):
    """Register rows per warp: the fewest (multiple of 16) for which the shared-memory part of R fits."""
    kw = 4 * width // WARPS
    for krw in range(0, kw + 1, 16):
        if smem_bytes(width, krw) <= SMEM_LIMIT:
            return krw
    raise ValueError("no register split fits")


def smem_bytes(width, krw):
    kw = 4 * width // WARPS
    return WARPS * (kw - krw) * 16 * 2 + WARPS * 256 * 4  # shared rows of R (16 columns, 2 bytes) + partial sums


def supported(width, sms=132):
    return width % 16 == 0 and (4 * width // WARPS) % 16 == 0 and 2 <= width // S <= sms and width >= 256


VERIFY_BITS = dict(check=1, delay=2, single=4, no_sync=8)


def verify_bits(verify):
    """verify: None or {"check": bool, "delay_seed": int | None, "broken": None | "single" | "no_sync"}."""
    if not verify:
        return 0, 0
    bits = VERIFY_BITS["check"] if verify.get("check", True) else 0
    if verify.get("delay_seed") is not None:
        bits |= VERIFY_BITS["delay"]
    if verify.get("broken"):
        bits |= VERIFY_BITS[verify["broken"]]
    return bits, int(verify.get("delay_seed") or 0)


def build(width, krw, slstm, bf16, cache_dir=None, verify=0):
    cuda_home = os.environ.get("FRNN_HYBRID_CUDA_HOME", "/apps/ACC/CUDA/12.8")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    flags = [f"-DFRNN_D={width}", f"-DFRNN_KRW={krw}", f"-DFRNN_SLSTM={int(slstm)}", f"-DFRNN_BF16={int(bf16)}",
             f"-DFRNN_VERIFY={int(verify)}"]
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


def _lib(width, krw, slstm, bf16, verify=0):
    key = (width, krw, slstm, bf16, verify)
    if key not in _LIBS:
        lib = ctypes.CDLL(str(build(width, krw, slstm, bf16, verify=verify)))
        lib.frnn_hybrid_bw.restype = ctypes.c_int
        lib.frnn_hybrid_bw.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
        lib.frnn_hybrid_bw_smem.restype = ctypes.c_int
        _LIBS[key] = lib
    return _LIBS[key]


def backward(wx, recurrent, bias, initial, history, d_history, d_final, cell, *, reg_rows_per_warp=None, tf32=False,
             diagnostics=None, _verify=None):
    """Returns (dg FP32 [B,T,4,H,D], dR, dbias, d_initial FP32); d_history / d_final as in triton_rtiled_bw.

    _verify (race verification only, never served): see verify_bits; the violation counters are written back into
    the dict (after_read, after_mma)."""
    from .triton_rtiled_bw import preactivations, weight_gradients

    if cell not in ("lstm", "slstm"):
        raise ValueError("the hybrid backward supports LSTM and sLSTM")
    batch, steps, gates, heads, width = wx.shape
    if wx.dtype not in (torch.float16, torch.bfloat16) or not supported(width):
        raise ValueError("unsupported shape or dtype for the hybrid backward")
    device = wx.device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    krw = reg_rows(width) if reg_rows_per_warp is None else reg_rows_per_warp
    k = width // S
    tiles = (batch + 15) // 16
    groups = tiles * heads
    resident = max(1, min(groups, sms // k))
    slots = (groups + resident - 1) // resident
    pre, h_prev = preactivations(wx, recurrent, bias, initial, history, tf32)
    dg = torch.empty((batch, steps, 4, heads, width), device=device, dtype=torch.float32)
    d_init = torch.empty(initial.shape, device=device, dtype=torch.float32)
    xbuf = torch.empty((resident, 2, 16, 4 * width), device=device, dtype=wx.dtype)
    init32 = initial.float().contiguous()
    bits, seed = verify_bits(_verify)
    lib = _lib(width, krw, cell == "slstm", wx.dtype == torch.bfloat16, bits)
    err = torch.zeros(4, device=device, dtype=torch.int32)
    err[3] = seed
    tensors = [pre.contiguous(), history.contiguous(), init32, d_history.contiguous(), d_final.float().contiguous(),
               recurrent.contiguous(), dg, d_init, xbuf, err]
    rc = lib.frnn_hybrid_bw(*[ctypes.c_void_p(x.data_ptr()) for x in tensors], batch, steps, heads, groups, resident,
                            slots, tiles, ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if rc != 0:
        raise RuntimeError(f"hybrid backward launch failed (cudaError {rc})")
    if _verify is not None:
        counts = err.tolist()
        _verify.update(after_read=counts[1], after_mma=counts[2])
    d_recurrent, d_bias = weight_gradients(dg, h_prev, tf32)
    if diagnostics is not None:
        diagnostics.update(reg_rows_per_warp=krw, shared_bytes=lib.frnn_hybrid_bw_smem(), ctas=k * resident,
                           groups=groups, resident=resident, slots=slots)
    return dg, d_recurrent, d_bias, d_init
