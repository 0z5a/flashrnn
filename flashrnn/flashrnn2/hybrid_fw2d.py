"""Large-D forward with R split in two dimensions (hybrid_fw2d.cu) through ctypes; docs/flashrnn2/max-width.md
section 56. The transpose of hybrid_bw2d: CTA (i, j) keeps the R block of unit block i (all four gates) and hidden
K block j resident, writes FP32 partial gate preactivations, and the owner of a unit sums its Q partials in fixed
order before the pointwise update. Same contract as tilelang_cluster.recurrence (whole logical calls; output modes
full / hidden / last; snapshot in the input dtype or FP32; returns the output and the FP32 compute carry). Dispatch
family "cuda2d"; not a default.
"""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("hybrid_fw2d.cu")
# D -> (unit blocks PC, hidden K blocks Q, register k-steps CR): PC * Q CTAs per group. Each CTA's block (4 D / PC output
# columns x D / Q hidden indices) has the shape of the 2-D backward's CTA at the same D, so the exchanged and partial
# bytes per step are the backward's too.
CONFIGS = {1024: (32, 4, 0), 1280: (20, 4, 0), 1536: (32, 4, 0), 1792: (28, 4, 4), 2048: (32, 4, 8)}
# Head-concurrent grids (opt-in conc=True): a quarter of the CTAs per group, each CTA the D = 2048 block shape (256
# columns x 512 K, eight register k-steps), four groups resident at once.
CONC_CONFIGS = {1024: (16, 2, 8)}
MODES = {"full": 0, "hidden": 1, "last": 2}
VERIFY_BITS = dict(check=1, delay=2, no_b1=4, no_b2=8, prof=16)
_LIBS = {}


def supported(width):
    return width in CONFIGS


def batch_tile(batch):
    return 16 if batch <= 16 else 32 if batch <= 32 else 64


def build(width, bm, slstm, bf16, mode, snap32, verify=0, cache_dir=None, grid=None, prefetch=False, stages=2):
    pc, q, cr = grid or CONFIGS[width]
    cuda_home = os.environ.get("FRNN_HYBRID_CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    flags = [f"-DFRNN_D={width}", f"-DFRNN_PC={pc}", f"-DFRNN_Q={q}", f"-DFRNN_CR={cr}", f"-DFRNN_BM={int(bm)}",
             f"-DFRNN_SLSTM={int(slstm)}", f"-DFRNN_BF16={int(bf16)}", f"-DFRNN_MODE={int(mode)}",
             f"-DFRNN_SNAP32={int(snap32)}", f"-DFRNN_VERIFY={int(verify)}"]
    if prefetch:  # split B2 with this step's Wx loads in between (section 76)
        flags.append("-DFRNN_FW_PREFETCH=1")
    if stages != 2:  # cp.async staging depth (section 76)
        flags.append(f"-DFRNN_STAGES={int(stages)}")
    digest = hashlib.sha256(SOURCE.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    cache = Path(cache_dir or os.environ.get("FRNN_HYBRID_CACHE",
                                             os.environ.get("TMPDIR", str(Path.home() / ".cache")) + "/frnn_hybrid_bw"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"hybrid_fw2d_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        cmd = [os.path.join(cuda_home, "bin", "nvcc"), "-std=c++17", "-O3", "-arch=sm_90a", "-shared",
               "-Xcompiler=-fPIC", "-cudart=static", *flags, str(SOURCE), "-o", str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(lib)
    return lib


def _lib(width, bm, slstm, bf16, mode, snap32, verify=0, grid=None, prefetch=False, stages=2):
    key = (width, bm, slstm, bf16, mode, snap32, verify, grid, prefetch, stages)
    if key not in _LIBS:
        lib = ctypes.CDLL(str(build(width, bm, slstm, bf16, mode, snap32, verify, grid=grid, prefetch=prefetch,
                                    stages=stages)))
        lib.frnn_hybrid_fw2d.restype = ctypes.c_int
        lib.frnn_hybrid_fw2d.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 7 + [ctypes.c_void_p] * 2
        lib.frnn_hybrid_fw2d_smem.restype = ctypes.c_int
        lib.frnn_hybrid_fw2d_ctas.restype = ctypes.c_int
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


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, sequence_context, output_mode="full",
               snapshot_dtype=None, batch_rows=None, conc=False, prof=False, diagnostics=None, grid=None,
               prefetch=False, stages=2, _verify=None):
    """Returns (output, FP32 compute carry). Whole logical calls only (sequence start).

    grid, prefetch, stages (experimental, section 76): a (PC, Q, CR) grid in place of CONFIGS (Q = 1: a 1-D split with
    one grid barrier per step), the split-B2 Wx prefetch, and the cp.async staging depth."""
    from .state_contract import state_contract

    if cell not in ("lstm", "slstm"):
        raise ValueError("the 2-D forward supports LSTM and sLSTM")
    if output_mode not in MODES:
        raise ValueError("output mode must be full, hidden or last")
    if wx.dtype not in (torch.float16, torch.bfloat16) or not wx.is_cuda:
        raise ValueError("the 2-D forward requires CUDA FP16/BF16 inputs")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (wx, recurrent, bias, initial)):
        raise ValueError("the 2-D forward has no backward of its own; use training.recurrence")
    contract = state_contract(cell, wx.dtype)
    if sequence_context is None or sequence_context.contract != contract:
        raise ValueError("the 2-D forward requires a matching sequence context")
    if sequence_context.step_offset > 0:
        raise ValueError("the 2-D forward runs whole logical calls only")
    if snapshot_dtype not in (None, wx.dtype, torch.float32):
        raise ValueError("snapshot dtype must be input dtype or FP32")
    batch, steps, gates, heads, width = wx.shape
    states = len(contract.components)
    if gates != 4 or (not supported(width) and grid is None) or steps < 1:
        raise ValueError(f"unsupported shape for the 2-D forward (D in {sorted(CONFIGS)})")
    if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
        raise ValueError("expected public weight and bias layouts")
    if initial.shape != (states, batch, 1, heads, width):
        raise ValueError("expected public initial state layout")
    if recurrent.dtype != wx.dtype or bias.dtype != wx.dtype:
        raise ValueError("weights must match the input dtype")
    device = wx.device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    bm = batch_tile(batch) if batch_rows is None else batch_rows
    if conc and width not in CONC_CONFIGS:
        raise ValueError(f"no head-concurrent grid at D = {width} (D in {sorted(CONC_CONFIGS)})")
    grid = tuple(grid) if grid is not None else (CONC_CONFIGS[width] if conc else None)
    pc, q, _ = grid or CONFIGS[width]
    ctas = pc * q
    if ctas > sms:
        raise ValueError("the 2-D grid needs more CTAs than SMs")
    tiles = (batch + bm - 1) // bm
    groups = tiles * heads
    resident = max(1, min(groups, sms // ctas))
    slots = (groups + resident - 1) // resident
    snap = wx.dtype if snapshot_dtype is None else snapshot_dtype
    out_shape = {"full": (states, batch, steps, heads, width), "hidden": (batch, steps, heads, width),
                 "last": (batch, 1, heads, width)}[output_mode]
    output = torch.empty(out_shape, device=device, dtype=snap)
    final = torch.empty(initial.shape, device=device, dtype=torch.float32)
    xbuf = torch.empty((resident, bm, width), device=device, dtype=wx.dtype)
    part = torch.empty((resident, q, bm, 4 * width), device=device, dtype=torch.float32)
    stamp = torch.zeros(resident * pc * q + (1 if prefetch else 0), device=device, dtype=torch.int32)
    bits, seed = verify_bits(_verify)
    if _verify and output_mode == "last":
        raise ValueError("verification reads the stored history: output mode full or hidden")
    if prof:
        bits |= VERIFY_BITS["prof"]
    lib = _lib(width, bm, cell == "slstm", wx.dtype == torch.bfloat16, MODES[output_mode], snap == torch.float32,
               bits, grid=grid, prefetch=prefetch, stages=stages)
    err = torch.zeros(12, device=device, dtype=torch.int32)
    err[3].fill_(seed)  # in place, no host copy: CUDA-graph safe
    # sLSTM: the kernel reads the device flag itself (no .item(): no host synchronization, CUDA-graph capturable).
    zero_flag = (sequence_context.zero_normalizer.to(device=device, dtype=torch.bool).contiguous()
                 if cell == "slstm" else None)
    zero_first = ctypes.c_void_p(zero_flag.data_ptr()) if zero_flag is not None else ctypes.c_void_p(0)
    tensors = [wx.contiguous(), recurrent.contiguous(), bias.contiguous(), initial.float().contiguous(), output, final,
               xbuf, part, stamp, err]
    rc = lib.frnn_hybrid_fw2d(*[ctypes.c_void_p(x.data_ptr()) for x in tensors], batch, steps, heads, groups,
                              resident, slots, tiles, zero_first,
                              ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if rc != 0:
        raise RuntimeError(f"2-D hybrid forward launch failed (cudaError {rc})")
    if _verify is not None:
        c = err.tolist()
        _verify.update(after_read=c[1], stale_partials=c[2])
    if prof and diagnostics is not None:
        c = err.tolist()
        n = max(c[9], 1)
        diagnostics["phases_us_per_step"] = dict(barrier_b2=c[4] * 64 / 1000 / n, gemm=c[5] * 64 / 1000 / n,
                                                 barrier_b1=c[6] * 64 / 1000 / n, pointwise=c[7] * 64 / 1000 / n,
                                                 cta_steps=c[9])
    if diagnostics is not None:
        diagnostics.update(batch_rows=bm, unit_blocks=pc, k_blocks=q, ctas=ctas * resident, groups=groups,
                           resident=resident, slots=slots, shared_bytes=lib.frnn_hybrid_fw2d_smem())
    return output, final
