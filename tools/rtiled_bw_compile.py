"""GPU-free sm_90a compile record of the R-tiled backward kernel (login-node preflight).

Usage: rtiled_bw_compile.py --out DIR [--widths 64,128,...] [--slices 16,32,64] [--warps 4,8]
Shared bytes from Triton metadata; registers/spills from ptxas -v of the emitted PTX.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget

from flashrnn.flashrnn2.triton_rtiled import SPIN_LIMIT, k_parts
from flashrnn.flashrnn2.triton_rtiled_bw import _rtiled_bw

H100_SMEM_OPTIN = 232448


def compile_one(cell, dtype, width, s, warps):
    t = {"float16": "fp16", "bfloat16": "bf16"}[dtype]
    k1, k2 = k_parts(width)
    sig = dict(PRE="*fp32", STATES="*fp32", INITIAL="*fp32", DSTATES="*fp32", DFINAL="*fp32", R=f"*{t}",
               DG="*fp32", DINIT="*fp32", CARRY="*fp32", XBUF=f"*{t}", SYNC="*i32", ERR="*i32",
               B="i32", T="i32", T_START="i32", NSTEPS="i32", FROM_FINAL="i32", TAIL="i32", GROUPS="i32",
               SLOTS="i32", PERSISTENT="i32", H="constexpr", D="constexpr", SLSTM="constexpr", BM="constexpr",
               S="constexpr", K1="constexpr", K2="constexpr", SPIN="constexpr", VERIFY="constexpr")
    const = dict(H=1, D=width, SLSTM=cell == "slstm", BM=16, S=s, K1=k1, K2=k2, SPIN=SPIN_LIMIT, VERIFY=0)
    src = triton.compiler.ASTSource(fn=_rtiled_bw, signature=sig, constexprs=const)
    k = triton.compile(src, target=GPUTarget("cuda", 90, 32),
                       options=dict(num_warps=warps, enable_fp_fusion=False, launch_cooperative_grid=True))
    ptx = k.asm["ptx"]
    ptxas = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k.ptx")
        Path(path).write_text(ptx)
        r = subprocess.run([ptxas, "-v", "--gpu-name=sm_90a", path, "-o", os.path.join(d, "k.cubin")],
                           capture_output=True, text=True)
    regs = re.search(r"Used (\d+) registers", r.stderr)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", r.stderr)
    return dict(cell=cell, dtype=dtype, width=width, slice_width=s, warps=warps, k_parts=[k1, k2],
                splits=triton.cdiv(width, s), shared_bytes=k.metadata.shared,
                fits_smem=k.metadata.shared <= H100_SMEM_OPTIN, ptxas_ok=r.returncode == 0,
                registers=int(regs.group(1)) if regs else None,
                spill_store_bytes=int(spill.group(1)) if spill else None,
                spill_load_bytes=int(spill.group(2)) if spill else None,
                mma=dict(wgmma=ptx.count("wgmma"), mma_sync=ptx.count("mma.sync")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--widths", default="64,128,192,256,384,512,768")
    p.add_argument("--slices", default="16,32,64")
    p.add_argument("--warps", default="4,8")
    p.add_argument("--dtypes", default="float16,bfloat16")
    a = p.parse_args()
    rows = []
    for cell in ("lstm", "slstm"):
        for dtype in a.dtypes.split(","):
            for width in map(int, a.widths.split(",")):
                for s in map(int, a.slices.split(",")):
                    if s > triton.next_power_of_2(width) or triton.cdiv(width, s) > 132:
                        continue
                    for warps in map(int, a.warps.split(",")):
                        t0 = time.time()
                        try:
                            row = compile_one(cell, dtype, width, s, warps)
                        except Exception as error:  # noqa: BLE001
                            row = dict(cell=cell, dtype=dtype, width=width, slice_width=s, warps=warps,
                                       error=str(error)[-600:])
                        row["seconds"] = round(time.time() - t0, 1)
                        rows.append(row)
                        print(json.dumps({k: row.get(k) for k in ("cell", "dtype", "width", "slice_width", "warps",
                                                                  "registers", "spill_store_bytes", "shared_bytes",
                                                                  "error")}), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "bw-compile.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
