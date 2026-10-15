"""GPU-free sm_90a compile sweep of the R-tiled kernel (login-node resource preflight).

Shared bytes come from Triton metadata; registers/spills from ptxas -v of the
emitted PTX. This is a compile record, not a launch or co-residency check.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import triton
from triton.backends.compiler import GPUTarget

from flashrnn.flashrnn2.triton_rtiled import SPIN_LIMIT, _rtiled, k_parts

H100_SMEM_OPTIN = 232448


def compile_one(cell, dtype, width, s, warps, pad, mode=0):
    t = {"float16": "fp16", "bfloat16": "bf16"}[dtype]
    k1, k2 = k_parts(width, pad)
    sig = dict(
        WX=f"*{t}", R=f"*{t}", BIAS=f"*{t}", INITIAL="*fp32", OUTPUT="*fp32", FINAL="*fp32",
        XBUF=f"*{t}", SYNC="*i32", ERR="*i32", ZERO_NORMALIZER="*i1",
        B="i32", T="i32", TOTAL_T="i32", STEP_BASE="i32", SEQUENCE_START="i32", GROUPS="i32", SLOTS="i32",
        PERSISTENT="i32", H="constexpr", D="constexpr", SLSTM="constexpr", BM="constexpr", S="constexpr", K1="constexpr",
        K2="constexpr", OUTPUT_MODE="constexpr", SPIN="constexpr",
    )
    const = dict(H=1, D=width, SLSTM=cell == "slstm", BM=16, S=s, K1=k1, K2=k2, OUTPUT_MODE=mode,
                 SPIN=SPIN_LIMIT)
    src = triton.compiler.ASTSource(fn=_rtiled, signature=sig, constexprs=const)
    k = triton.compile(src, target=GPUTarget("cuda", 90, 32),
                       options=dict(num_warps=warps, enable_fp_fusion=False, launch_cooperative_grid=True))
    ptx = k.asm["ptx"]
    ptxas = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.ptx")
        open(p, "w").write(ptx)
        r = subprocess.run([ptxas, "-v", "--gpu-name=sm_90a", p, "-o", os.path.join(d, "k.cubin")],
                           capture_output=True, text=True)
    log = r.stderr
    regs = re.search(r"Used (\d+) registers", log)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    feats = {f: ptx.count(f) for f in ("mma.sync", "wgmma", "ldmatrix", "ld.global.cg", "atom.global.acquire",
                                       "atom.global.release", "atom.acquire", "atom.release", "bar.sync",
                                       "fence.acq_rel", "membar")}
    return dict(k_parts=[k1, k2], splits=triton.cdiv(width, s), shared_bytes=k.metadata.shared,
                fits_smem=k.metadata.shared <= H100_SMEM_OPTIN,
                registers=int(regs.group(1)) if regs else None,
                spill_store_bytes=int(spill.group(1)) if spill else None,
                spill_load_bytes=int(spill.group(2)) if spill else None,
                ptx=feats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--widths", default="64,96,128,192,256,384,512,768")
    p.add_argument("--cells", default="lstm,slstm")
    p.add_argument("--warps", default="4,8")
    p.add_argument("--slices", default="16,32,64,128")
    a = p.parse_args()
    rows = []
    for cell in a.cells.split(","):
        for width in map(int, a.widths.split(",")):
            for s in map(int, a.slices.split(",")):
                if s > triton.next_power_of_2(width):
                    continue
                for warps in map(int, a.warps.split(",")):
                    for pad in ((False, True) if k_parts(width)[1] else (False,)):
                        rec = dict(cell=cell, dtype="float16", width=width, slice_width=s, warps=warps, pad_k=pad)
                        start = time.time()
                        try:
                            rec.update(compile_one(cell, "float16", width, s, warps, pad), status="COMPILED")
                        except Exception as e:  # noqa: BLE001
                            rec.update(status="FAILED", error=f"{type(e).__name__}: {str(e)[:400]}")
                        rec["seconds"] = round(time.time() - start, 2)
                        rows.append(rec)
                        print(json.dumps(rec), flush=True)
    json.dump(dict(triton=triton.__version__, python=sys.version, rows=rows), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
