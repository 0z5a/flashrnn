"""Large-D backward with R split in two dimensions (hybrid_bw2d.cu) through ctypes; docs/flashrnn2/max-width.md
section 20. Same contract as hybrid_bw.backward (preactivation recompute and dR / dbias GEMMs reused from the Triton
backward); the K = 4D reduction is summed in a different order (Q FP32 partials per column, fixed order), so the
outputs are not bitwise equal to the 1-D kernel's. Training backend option ``backend="hybrid2d"``; not a default.
"""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("hybrid_bw2d.cu")
# D -> (column blocks PC, K blocks Q, register k-steps CR): PC * Q CTAs per group, R block in shared memory (+ CR
# k-steps of 16 rows in registers at D = 2048); D = 1792 does not factor into a fitting grid.
CONFIGS = {1024: (8, 16, 0), 1280: (5, 16, 0), 1536: (8, 16, 0), 2048: (8, 16, 8)}
VERIFY_BITS = dict(check=1, delay=2, no_b1=4, no_b2=8, prof=16)
_LIBS = {}


def supported(width):
    return width in CONFIGS


def batch_tile(batch):
    return 16 if batch <= 16 else 32 if batch <= 32 else 64


def build(width, bm, slstm, bf16, verify=0, cache_dir=None):
    pc, q, cr = CONFIGS[width]
    cuda_home = os.environ.get("FRNN_HYBRID_CUDA_HOME", "/apps/ACC/CUDA/12.8")
    flags = [f"-DFRNN_D={width}", f"-DFRNN_PC={pc}", f"-DFRNN_Q={q}", f"-DFRNN_CR={cr}", f"-DFRNN_BM={int(bm)}",
             f"-DFRNN_SLSTM={int(slstm)}", f"-DFRNN_BF16={int(bf16)}", f"-DFRNN_VERIFY={int(verify)}"]
    digest = hashlib.sha256(SOURCE.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    cache = Path(cache_dir or os.environ.get("FRNN_HYBRID_CACHE",
                                             os.environ.get("TMPDIR", str(Path.home() / ".cache")) + "/frnn_hybrid_bw"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"hybrid_bw2d_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        cmd = [os.path.join(cuda_home, "bin", "nvcc"), "-std=c++17", "-O3", "-arch=sm_90a", "-shared",
               "-Xcompiler=-fPIC", "-cudart=static", *flags, str(SOURCE), "-o", str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(lib)
    return lib


def _lib(width, bm, slstm, bf16, verify=0):
    key = (width, bm, slstm, bf16, verify)
    if key not in _LIBS:
        lib = ctypes.CDLL(str(build(width, bm, slstm, bf16, verify)))
        lib.frnn_hybrid_bw2d.restype = ctypes.c_int
        lib.frnn_hybrid_bw2d.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
        lib.frnn_hybrid_bw2d_smem.restype = ctypes.c_int
        lib.frnn_hybrid_bw2d_ctas.restype = ctypes.c_int
        _LIBS[key] = lib
    return _LIBS[key]


def verify_bits(verify):
    """verify: None or {"check": bool, "delay_seed": int | None, "broken": None | "no_b1" | "no_b2"}."""
    if not verify:
        return 0, 0
    bits = VERIFY_BITS["check"] if verify.get("check", True) else 0
    if verify.get("delay_seed") is not None:
        bits |= VERIFY_BITS["delay"]
    if verify.get("broken"):
        bits |= VERIFY_BITS[verify["broken"]]
    return bits, int(verify.get("delay_seed") or 0)


def backward(wx, recurrent, bias, initial, history, d_history, d_final, cell, *, batch_rows=None, tf32=False,
             wgrad="bmm", prof=False, diagnostics=None, _verify=None):
    """Returns (dg FP32 [B,T,4,H,D], dR, dbias, d_initial FP32), as hybrid_bw.backward."""
    from .triton_rtiled_bw import preactivations, weight_gradients

    if cell not in ("lstm", "slstm"):
        raise ValueError("the 2-D backward supports LSTM and sLSTM")
    batch, steps, gates, heads, width = wx.shape
    if wx.dtype not in (torch.float16, torch.bfloat16) or not supported(width):
        raise ValueError(f"unsupported shape or dtype for the 2-D backward (D in {sorted(CONFIGS)})")
    device = wx.device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    bm = batch_tile(batch) if batch_rows is None else batch_rows
    pc, q, _ = CONFIGS[width]
    ctas = pc * q
    if ctas > sms:
        raise ValueError("the 2-D grid needs more CTAs than SMs")
    tiles = (batch + bm - 1) // bm
    groups = tiles * heads
    resident = max(1, min(groups, sms // ctas))
    slots = (groups + resident - 1) // resident
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if (diagnostics or {}).get("time") else None
    if events:
        events[0].record()
    pre, h_prev = preactivations(wx, recurrent, bias, initial, history, tf32)
    if events:
        events[1].record()
    dg = torch.empty((batch, steps, 4, heads, width), device=device, dtype=torch.float32)
    d_init = torch.empty(initial.shape, device=device, dtype=torch.float32)
    xbuf = torch.empty((resident, bm, 4 * width), device=device, dtype=wx.dtype)
    part = torch.empty((resident, q, bm, width), device=device, dtype=torch.float32)
    stamp = torch.zeros(resident * q, device=device, dtype=torch.int32)
    bits, seed = verify_bits(_verify)
    if prof:
        bits |= VERIFY_BITS["prof"]
    lib = _lib(width, bm, cell == "slstm", wx.dtype == torch.bfloat16, bits)
    err = torch.zeros(12, device=device, dtype=torch.int32)
    err[3] = seed
    tensors = [pre.contiguous(), history.contiguous(), initial.float().contiguous(), d_history.contiguous(),
               d_final.float().contiguous(), recurrent.contiguous(), dg, d_init, xbuf, part, stamp, err]
    rc = lib.frnn_hybrid_bw2d(*[ctypes.c_void_p(x.data_ptr()) for x in tensors], batch, steps, heads, groups, resident,
                              slots, tiles, ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if rc != 0:
        raise RuntimeError(f"2-D hybrid backward launch failed (cudaError {rc})")
    if events:
        events[2].record()
    if _verify is not None:
        c = err.tolist()
        _verify.update(after_read=c[1], stale_partials=c[2])
    d_recurrent, d_bias = weight_gradients(dg, h_prev, tf32, wgrad)
    if events:
        events[3].record()
        torch.cuda.synchronize()
        diagnostics["ms"] = dict(preactivation=events[0].elapsed_time(events[1]),
                                 kernel=events[1].elapsed_time(events[2]),
                                 weight_gradients=events[2].elapsed_time(events[3]))
    if prof and diagnostics is not None:
        c = err.tolist()
        n = max(c[9], 1)
        diagnostics["phases_us_per_step"] = dict(
            pointwise=c[4] * 64 / 1000 / n, barrier_b2=c[5] * 64 / 1000 / n, gemm=c[6] * 64 / 1000 / n,
            barrier_b1=c[7] * 64 / 1000 / n, partial_sum=c[8] * 64 / 1000 / n, cta_steps=c[9])
    if diagnostics is not None:
        diagnostics.update(batch_rows=bm, column_blocks=pc, k_blocks=q, ctas=ctas * resident, groups=groups,
                           resident=resident, slots=slots, shared_bytes=lib.frnn_hybrid_bw2d_smem())
    return dg, d_recurrent, d_bias, d_init
