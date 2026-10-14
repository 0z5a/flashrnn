"""ThunderKittens R tiling (tk_rtiled.cu) through ctypes.

The kernel is built as a standalone shared object with the CUDA 12.x toolkit
(ThunderKittens needs C++20 and sm_90a), independent of PyTorch's own CUDA
version; tensors cross as raw device pointers on PyTorch's current stream.
Contract: FP16, LSTM/sLSTM, one full logical call; hidden history and FP32 carry.
"""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("tk_rtiled.cu")
SUPPORTED = {(128, 32), (128, 64), (192, 16), (192, 32), (192, 64), (256, 16), (256, 32), (256, 64), (384, 32),
             (512, 32)}
_LIB = None


def build(cache_dir=None, nvcc=None, tk_root=None):
    tk_root = Path(tk_root or os.environ["FRNN_TK_ROOT"])
    nvcc = nvcc or os.path.join(os.environ.get("FRNN_TK_CUDA_HOME", "/apps/ACC/CUDA/12.8"), "bin", "nvcc")
    digest = hashlib.sha256(SOURCE.read_bytes() + str(tk_root).encode()).hexdigest()[:16]
    cache = Path(cache_dir or os.environ.get("FRNN_TK_CACHE", Path.home() / ".cache/frnn_tk"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"tk_rtiled_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        cmd = [nvcc, "-std=c++20", "-O3", "-arch=sm_90a", "-DKITTENS_HOPPER", "-shared", "-Xcompiler=-fPIC",
               "--expt-relaxed-constexpr", "--expt-extended-lambda", "-Xcompiler=-Wno-psabi",
               "-Xcompiler=-fno-strict-aliasing", "-cudart=static", f"-I{tk_root}/include",
               f"-I{tk_root}/prototype", str(SOURCE), "-o", str(tmp), "-lcuda"]
        subprocess.run(cmd, check=True)
        tmp.replace(lib)
    return lib


def _lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build()))
        lib.frnn_tk_rtiled.restype = ctypes.c_int
        lib.frnn_tk_rtiled.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3 + [
            ctypes.c_void_p]
        _LIB = lib
    return _LIB


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, slice_width=32):
    batch, steps, gates, heads, width = wx.shape
    if wx.dtype != torch.float16 or gates != 4 or (width, slice_width) not in SUPPORTED:
        raise ValueError("ThunderKittens R tiling supports FP16 and compiled (D, S) pairs only")
    states = 4 if cell == "slstm" else 2
    zero = (torch.all(initial[2] == 0) if cell == "slstm" else torch.zeros((), device=wx.device, dtype=torch.bool))
    zero = zero.to(torch.int32).reshape(1)
    wx, recurrent, bias = wx.contiguous(), recurrent.contiguous(), bias.contiguous()
    init = initial.float().contiguous()
    out = torch.empty((batch, steps, heads, width), device=wx.device, dtype=torch.float16)
    fin = torch.empty((states, batch, 1, heads, width), device=wx.device, dtype=torch.float32)
    code = _lib().frnn_tk_rtiled(int(cell == "slstm"), width, slice_width, wx.data_ptr(), recurrent.data_ptr(),
                                 bias.data_ptr(), init.data_ptr(), zero.data_ptr(), out.data_ptr(), fin.data_ptr(),
                                 batch, steps, heads, torch.cuda.current_stream(wx.device).cuda_stream)
    if code:
        raise RuntimeError(f"ThunderKittens launch failed with code {code}")
    return out, fin
