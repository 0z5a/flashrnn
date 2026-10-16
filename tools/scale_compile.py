"""GPU-free sm_90a compile record of the multi-GPU forward kernel (login-node preflight for the scale-out batch).

Usage: scale_compile.py --out DIR [--cases "P:D:S:KC:SCHED:RMODE,..."]
Default cases cover phase A of docs/flashrnn2/scale-out-plan.md. Shared bytes from Triton metadata; registers and
spills from ptxas -v of the emitted PTX.
"""
import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget

from flashrnn.flashrnn2.multi_gpu import SCHEDULES, _rtiled_mg, rounds_of
from flashrnn.flashrnn2.triton_rtiled import SPIN_LIMIT

H100_SMEM_OPTIN = 232448
DEFAULT = []
for width in (1024, 2048, 4096):
    for peers in (1, 2, 3, 4):
        for s in (8, 16):
            if width % (peers * s):
                continue
            for sched in ("direct", "ring", "doubling"):
                if sched == "doubling" and peers & (peers - 1):
                    continue
                if sched != "direct" and peers == 1:
                    continue
                for rmode in ("smem", "stream"):
                    if rmode == "smem" and 4 * width * s * 2 > 160 * 1024:
                        continue  # R slice alone exceeds shared memory; only the stream mode is planned there
                    DEFAULT.append((peers, width, s, 512, sched, rmode))


def compile_one(peers, width, s, kc, sched, rmode, cell="lstm", warps=4, prof=0):
    kc = min(kc, triton.next_power_of_2(width))
    nch = triton.cdiv(width, kc)
    sig = dict(WX="*fp16", R="*fp16", BIAS="*fp16", INITIAL="*fp32", OUTPUT="*fp32", FINAL="*fp32", PEERS="*i64",
               SYNCS="*i64", ERR="*i32", ZERO_NORMALIZER="*i1", B="i32", T="i32", GROUPS="i32", SLOTS="i32",
               RANK="i32", COL0="i32", SPLITS_PER="i32", TOTAL_SPLITS="i32",
               H="constexpr", D="constexpr", SLSTM="constexpr", BM="constexpr", S="constexpr", KC="constexpr",
               NCH="constexpr", OUTPUT_MODE="constexpr", SPIN="constexpr", NP="constexpr", SCHED="constexpr",
               STRIDE="constexpr", ROUNDS="constexpr", DG="constexpr", RMODE="constexpr", PROF="constexpr")
    const = dict(H=1, D=width, SLSTM=cell == "slstm", BM=16, S=s, KC=kc, NCH=nch, OUTPUT_MODE=1, SPIN=SPIN_LIMIT,
                 NP=peers, SCHED=SCHEDULES[sched], STRIDE=1, ROUNDS=rounds_of(sched, peers), DG=width // peers,
                 RMODE=0 if rmode == "smem" else 1, PROF=prof)
    src = triton.compiler.ASTSource(fn=_rtiled_mg, signature=sig, constexprs=const)
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
    return dict(peers=peers, width=width, slice_width=s, k_chunk=kc, chunks=nch, schedule=sched, rmode=rmode,
                cell=cell, warps=warps, prof=prof, shared_bytes=k.metadata.shared,
                fits_smem=k.metadata.shared <= H100_SMEM_OPTIN, ptxas_ok=r.returncode == 0,
                registers=int(regs.group(1)) if regs else None,
                spill_store_bytes=int(spill.group(1)) if spill else None,
                spill_load_bytes=int(spill.group(2)) if spill else None,
                sys_atomics=ptx.count(".sys"), mma=ptx.count("mma.sync"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cases", default="")
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--prof", type=int, default=0)
    a = p.parse_args()
    cases = DEFAULT
    if a.cases:
        cases = []
        for c in a.cases.split(","):
            peers, width, s, kc, sched, rmode = c.split(":")
            cases.append((int(peers), int(width), int(s), int(kc), sched, rmode))
    rows = []
    for case in cases:
        t0 = time.time()
        try:
            row = compile_one(*case, warps=a.warps, prof=a.prof)
        except Exception as error:  # noqa: BLE001
            row = dict(zip(("peers", "width", "slice_width", "k_chunk", "schedule", "rmode"), case),
                       error=str(error)[-800:])
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("peers", "width", "slice_width", "schedule", "rmode", "registers",
                                                  "spill_store_bytes", "shared_bytes", "fits_smem", "error")}),
              flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "scale-compile.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
