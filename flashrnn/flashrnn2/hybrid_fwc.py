"""Small-batch forward in one thread-block cluster (hybrid_fwc.cu) through ctypes; docs/flashrnn2/max-width.md
section 80. B <= 16 per cluster (several batch tiles / heads run as independent clusters), D <= 768: the hidden state
is exchanged through distributed shared memory under one cluster barrier per step. Experimental: not a dispatch family
and not a default."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("hybrid_fwc.cu")
MODES = {"full": 0, "hidden": 1, "last": 2}
SMEM_LIMIT = 227 * 1024
_LIBS = {}


def layout(width, cluster):
    nu = width // cluster
    nc, ks = 4 * nu, width // 16
    ntw = nc // 8 // 8
    return dict(nu=nu, nc=nc, ks=ks, ntw=ntw, smem_h=2 * 16 * (width + 8) * 2, smem_p=16 * nc * 4)


def min_register_ksteps(width, cluster):
    """Smallest CR with which R, the two h buffers and the partials fit shared memory (None: never)."""
    lay = layout(width, cluster)
    for cr in range(0, lay["ks"] + 1):
        if 8 * (lay["ks"] - cr) * lay["ntw"] * 256 + lay["smem_h"] + lay["smem_p"] <= SMEM_LIMIT:
            return cr
    return None


REGISTER_BUDGET = 160  # R fragment registers per thread (2 per k-step and n-tile) that leave room for the rest


def supported(width, cluster=16):
    if not (width % cluster == 0 and (width // cluster) % 16 == 0 and 2 <= cluster <= 16):
        return False
    cr = min_register_ksteps(width, cluster)
    return cr is not None and cr * layout(width, cluster)["ntw"] * 2 <= REGISTER_BUDGET


def build(width, cluster, cr, ksplit, slstm, bf16, mode, snap32, verify=0, pull=False, cache_dir=None):
    cuda_home = os.environ.get("FRNN_HYBRID_CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    flags = [f"-DFRNN_D={width}", f"-DFRNN_C={cluster}", f"-DFRNN_CR={cr}", f"-DFRNN_KSPLIT={ksplit}",
             f"-DFRNN_SLSTM={int(slstm)}", f"-DFRNN_BF16={int(bf16)}", f"-DFRNN_MODE={int(mode)}",
             f"-DFRNN_SNAP32={int(snap32)}", f"-DFRNN_VERIFY={int(verify)}", f"-DFRNN_FWC_PULL={int(pull)}"]
    digest = hashlib.sha256(SOURCE.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    cache = Path(cache_dir or os.environ.get("FRNN_HYBRID_CACHE",
                                             os.environ.get("TMPDIR", str(Path.home() / ".cache")) + "/frnn_hybrid_bw"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"hybrid_fwc_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        cmd = [os.path.join(cuda_home, "bin", "nvcc"), "-std=c++17", "-O3", "-arch=sm_90a", "-shared",
               "-Xcompiler=-fPIC", "-cudart=static", *flags, str(SOURCE), "-o", str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(lib)
    return lib


def _lib(*key):
    if key not in _LIBS:
        lib = ctypes.CDLL(str(build(*key)))
        lib.frnn_hybrid_fwc.restype = ctypes.c_int
        lib.frnn_hybrid_fwc.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        lib.frnn_hybrid_fwc_smem.restype = ctypes.c_int
        _LIBS[key] = lib
    return _LIBS[key]


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, sequence_context, output_mode="hidden",
               snapshot_dtype=None, cluster=16, cr=None, ksplit=2, pull=False, prof=False, diagnostics=None):
    """Returns (output, FP32 compute carry). Whole logical calls only (sequence start)."""
    from .state_contract import state_contract

    if cell not in ("lstm", "slstm"):
        raise ValueError("the cluster forward supports LSTM and sLSTM")
    if output_mode not in MODES:
        raise ValueError("output mode must be full, hidden or last")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("the cluster forward requires CUDA FP16/BF16 inputs")
    contract = state_contract(cell, wx.dtype)
    if sequence_context is None or sequence_context.contract != contract or sequence_context.step_offset > 0:
        raise ValueError("the cluster forward requires a matching sequence context at the sequence start")
    if snapshot_dtype not in (None, wx.dtype, torch.float32):
        raise ValueError("snapshot dtype must be input dtype or FP32")
    batch, steps, gates, heads, width = wx.shape
    states = len(contract.components)
    if gates != 4 or steps < 1 or not supported(width, cluster):
        raise ValueError(f"unsupported shape for the cluster forward (D = {width}, cluster {cluster})")
    if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
        raise ValueError("expected public weight and bias layouts")
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("expected public initial state layout")
    if recurrent.dtype != wx.dtype or bias.dtype != wx.dtype:
        raise ValueError("weights must match the input dtype")
    device = wx.device
    cr = min_register_ksteps(width, cluster) if cr is None else cr
    tiles = (batch + 15) // 16
    groups = tiles * heads
    snap = wx.dtype if snapshot_dtype is None else snapshot_dtype
    out_shape = {"full": (states, batch, steps, heads, width), "hidden": (batch, steps, heads, width),
                 "last": (batch, 1, heads, width)}[output_mode]
    output = torch.empty(out_shape, device=device, dtype=snap)
    final = torch.empty(initial.shape, device=device, dtype=torch.float32)
    err = torch.zeros(12, device=device, dtype=torch.int32)
    lib = _lib(width, cluster, cr, ksplit, cell == "slstm", wx.dtype == torch.bfloat16, MODES[output_mode],
               snap == torch.float32, 16 if prof else 0, pull)
    zero_flag = (sequence_context.zero_normalizer.to(device=device, dtype=torch.bool).contiguous()
                 if cell == "slstm" else None)
    tensors = [wx.contiguous(), recurrent.contiguous(), bias.contiguous(), initial.float().contiguous(), output, final,
               err]
    rc = lib.frnn_hybrid_fwc(*[ctypes.c_void_p(x.data_ptr()) for x in tensors],
                             ctypes.c_void_p(zero_flag.data_ptr()) if zero_flag is not None else ctypes.c_void_p(0),
                             batch, steps, heads, groups, tiles,
                             ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if rc != 0:
        raise RuntimeError(f"cluster forward launch failed (cudaError {rc})")
    if prof and diagnostics is not None:
        c = err.tolist()
        n = max(c[9], 1)
        diagnostics["phases_us_per_step"] = dict(barrier=c[4] * 64 / 1000 / n, gemm=c[5] * 64 / 1000 / n,
                                                 pointwise=c[6] * 64 / 1000 / n, cta_steps=c[9])
    if diagnostics is not None:
        diagnostics.update(cluster=cluster, cr=cr, ksplit=ksplit, pull=pull, groups=groups,
                           shared_bytes=lib.frnn_hybrid_fwc_smem())
    return output, final
