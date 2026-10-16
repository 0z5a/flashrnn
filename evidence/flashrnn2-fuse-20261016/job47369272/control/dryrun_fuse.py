"""Login-node preflight for the fused backward batch: every planned backward kernel (cell, D, slice, warps, fused
level; FP16 signature, the fused kernels with the Wx operand in the input dtype) compiles for sm_90a within shared
memory, and the tools import. Runs in parts (--part i --parts n) and merges them (--merge) into dryrun-fuse.json."""
import argparse
import json
import sys
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget

CONTROL = Path(__file__).resolve().parent
sys.path.insert(0, str(CONTROL.parent / "source" / "tools"))
import bw_bench  # noqa: E402,F401
import bw_profile  # noqa: E402,F401
import bw_qualify  # noqa: E402,F401
import race_verify  # noqa: E402,F401
from flashrnn.flashrnn2.triton_rtiled import SPIN_LIMIT, k_parts  # noqa: E402
from flashrnn.flashrnn2.triton_rtiled_bw import _rtiled_bw  # noqa: E402


def keys():
    plan = json.loads((CONTROL / "plan.json").read_text())
    out = set()
    for case in plan["select_cases"]:
        for spec in case["roles"]["frnn2"]:
            out.add((case["cell"], case["width"], spec["slice_width"], spec["warps"], spec.get("fuse", 0)))
    for q in plan["qualify"]:
        c = q["case"]
        out.add((c["cell"], c["width"], c["slice_width"], c["warps"], c["fuse"]))
    for r in plan["race"]:
        c = r["task"]["config"]
        out.add((r["task"]["cell"], c["width"], c["slice_width"], c["warps"], c["fuse"]))
    for c in plan["sanitizer"]:
        c = c["case"]
        out.add((c["cell"], c["width"], c["slice_width"], c["warps"], c["fuse"]))
    return sorted(out)


def compile_key(cell, width, s, warps, fuse):
    k1, k2 = k_parts(width)
    sig = dict(PRE="*fp16" if fuse else "*fp32", STATES="*fp32", INITIAL="*fp32", DSTATES="*fp32", DFINAL="*fp32",
               R="*fp16", DG="*fp32", DINIT="*fp32", CARRY="*fp32", XBUF="*fp16", SYNC="*i32", ERR="*i32",
               B="i32", T="i32", T_START="i32", NSTEPS="i32", FROM_FINAL="i32", TAIL="i32", GROUPS="i32",
               SLOTS="i32", PERSISTENT="i32", H="constexpr", D="constexpr", SLSTM="constexpr", BM="constexpr",
               S="constexpr", K1="constexpr", K2="constexpr", SPIN="constexpr", VERIFY="constexpr",
               FAST="constexpr", DS_C="constexpr", DS_N="constexpr", FUSE="constexpr")
    const = dict(H=1, D=width, SLSTM=cell == "slstm", BM=16, S=s, K1=k1, K2=k2, SPIN=SPIN_LIMIT, VERIFY=0,
                 FAST=1, DS_C=0, DS_N=0, FUSE=fuse)
    k = triton.compile(triton.compiler.ASTSource(fn=_rtiled_bw, signature=sig, constexprs=const),
                       target=GPUTarget("cuda", 90, 32),
                       options=dict(num_warps=warps, enable_fp_fusion=False, launch_cooperative_grid=True))
    return k.metadata.shared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--merge", action="store_true")
    a = p.parse_args()
    all_keys = keys()
    if a.merge:
        rec = dict(compiled=0, failed=[], distinct=len(all_keys), covered=[])
        for i in range(a.parts):
            part = json.loads((CONTROL / f"dryrun-fuse-part{i}.json").read_text())
            rec["compiled"] += part["compiled"]
            rec["failed"] += part["failed"]
            rec["covered"] += part["keys"]
        rec["all_keys_covered"] = sorted(map(tuple, rec["covered"])) == all_keys
        rec["status"] = "PASS" if not rec["failed"] and rec["all_keys_covered"] else "FAIL"
        del rec["covered"]
        (CONTROL / "dryrun-fuse.json").write_text(json.dumps(rec, indent=1))
        print("DRYRUN_FUSE " + json.dumps(rec))
        return
    mine = all_keys[a.part::a.parts]
    rec = dict(compiled=0, failed=[], keys=mine, shared={})
    for key in mine:
        try:
            shared = compile_key(*key)
            rec["shared"][str(key)] = shared
            if shared > 232448:
                rec["failed"].append(list(key) + ["smem", shared])
            else:
                rec["compiled"] += 1
        except Exception as error:  # noqa: BLE001
            rec["failed"].append(list(key) + [str(error)[-300:]])
    (CONTROL / f"dryrun-fuse-part{a.part}.json").write_text(json.dumps(rec, indent=1))
    print(f"DRYRUN_FUSE_PART {a.part} compiled={rec['compiled']} failed={len(rec['failed'])} of {len(mine)}")


if __name__ == "__main__":
    main()
