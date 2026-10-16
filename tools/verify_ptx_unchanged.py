"""Login-node check: with race verification off (VERIFY=0), the instrumented kernels compile to the same PTX as the
uninstrumented ones (debug line/file directives and DWARF sections stripped).

Usage: verify_ptx_unchanged.py --baseline-dir DIR --out DIR. DIR holds the uninstrumented module files
(triton_rtiled.py, triton_rtiled_bw.py) from the reference commit.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ptx(fn, sig, const, warps):
    src = triton.compiler.ASTSource(fn=fn, signature=sig, constexprs=const)
    k = triton.compile(src, target=GPUTarget("cuda", 90, 32),
                       options=dict(num_warps=warps, enable_fp_fusion=False, launch_cooperative_grid=True))
    text = re.split(r"\.section\s+\.debug", k.asm["ptx"])[0]  # instructions only; DWARF line tables differ
    return "\n".join(l for l in text.splitlines()
                     if not l.strip().startswith((".loc", ".file", "//")) and "debug" not in l)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import flashrnn.flashrnn2.triton_rtiled as new
    old = load(Path(a.baseline_dir) / "triton_rtiled.py", "flashrnn.flashrnn2.triton_rtiled_baseline")
    rows = []
    for cell in ("lstm", "slstm"):
        for width, s, mode in ((256, 32, 0), (768, 32, 1), (192, 64, 0)):
            k1, k2 = new.k_parts(width)
            sig = dict(WX="*fp16", R="*fp16", BIAS="*fp16", INITIAL="*fp32", OUTPUT="*fp32", FINAL="*fp32",
                       XBUF="*fp16", SYNC="*i32", ERR="*i32", ZERO_NORMALIZER="*i1", B="i32", T="i32", TOTAL_T="i32",
                       STEP_BASE="i32", SEQUENCE_START="i32", GROUPS="i32", SLOTS="i32", PERSISTENT="i32",
                       H="constexpr", D="constexpr", SLSTM="constexpr", BM="constexpr", S="constexpr",
                       K1="constexpr", K2="constexpr", OUTPUT_MODE="constexpr", SPIN="constexpr")
            const = dict(H=1, D=width, SLSTM=cell == "slstm", BM=16, S=s, K1=k1, K2=k2, OUTPUT_MODE=mode,
                         SPIN=new.SPIN_LIMIT)
            old_v = "VERIFY" in old._rtiled.arg_names  # baseline may already carry the (off) instrumentation
            a_ptx = ptx(old._rtiled, dict(sig, VERIFY="constexpr") if old_v else sig,
                        dict(const, VERIFY=0) if old_v else const, 8)
            sig_v = dict(sig, VERIFY="constexpr")
            c_ptx = ptx(new._rtiled, sig_v, dict(const, VERIFY=0), 8)
            on = ptx(new._rtiled, sig_v, dict(const, VERIFY=3), 8)
            rows.append(dict(kernel="forward", cell=cell, width=width, slice_width=s, mode=mode,
                             explicit_zero_equal=a_ptx == c_ptx,
                             verify_on_differs=on != a_ptx))
    import flashrnn.flashrnn2.triton_rtiled_bw as new_bw
    old_bw = load(Path(a.baseline_dir) / "triton_rtiled_bw.py", "flashrnn.flashrnn2.triton_rtiled_bw_baseline")
    for cell in ("lstm", "slstm"):
        for width, s in ((256, 32), (768, 16), (64, 64)):
            k1, k2 = new.k_parts(width)
            sig = dict(PRE="*fp32", STATES="*fp32", INITIAL="*fp32", DSTATES="*fp32", DFINAL="*fp32", R="*fp16",
                       DG="*fp32", DINIT="*fp32", CARRY="*fp32", XBUF="*fp16", SYNC="*i32", ERR="*i32",
                       B="i32", T="i32", T_START="i32", NSTEPS="i32", FROM_FINAL="i32", TAIL="i32", GROUPS="i32",
                       SLOTS="i32", PERSISTENT="i32", H="constexpr", D="constexpr", SLSTM="constexpr",
                       BM="constexpr", S="constexpr", K1="constexpr", K2="constexpr", SPIN="constexpr")
            const = dict(H=1, D=width, SLSTM=cell == "slstm", BM=16, S=s, K1=k1, K2=k2, SPIN=new.SPIN_LIMIT)
            old_names = old_bw._rtiled_bw.arg_names
            extra = {k: v for k, v in (("VERIFY", 0), ("FAST", 0), ("DS_C", 1), ("DS_N", 1), ("FUSE", 0), ("SPLITG", 0),
                                       ("PROF", 0)) if k in old_names}
            old_sig = dict(sig, **{k: "constexpr" for k in extra})
            old_const = dict(const, **extra)
            a_ptx = ptx(old_bw._rtiled_bw, old_sig, old_const, 8)
            sig_v = dict(sig, VERIFY="constexpr", FAST="constexpr", DS_C="constexpr", DS_N="constexpr", FUSE="constexpr",
                         SPLITG="constexpr", PROF="constexpr")
            off = dict(SPLITG=0, PROF=0)
            c_ptx = ptx(new_bw._rtiled_bw, sig_v, dict(const, VERIFY=0, FAST=0, DS_C=1, DS_N=1, FUSE=0, **off), 8)
            on = ptx(new_bw._rtiled_bw, sig_v, dict(const, VERIFY=3, FAST=0, DS_C=1, DS_N=1, FUSE=0, **off), 8)
            fast = ptx(new_bw._rtiled_bw, sig_v, dict(const, VERIFY=0, FAST=1, DS_C=0, DS_N=0, FUSE=0, **off), 8)
            if "FUSE" in old_names:  # the fused variant (Wx operand in the input dtype) must also be unchanged
                sig_f = dict(sig_v, PRE="*fp16")
                old_sig_f = dict(old_sig, PRE="*fp16")
                new_fused = ptx(new_bw._rtiled_bw, sig_f, dict(const, VERIFY=0, FAST=1, DS_C=0, DS_N=0, FUSE=2, **off), 8)
                old_fused = ptx(old_bw._rtiled_bw, old_sig_f, dict(old_const, FAST=1, DS_C=0, DS_N=0, FUSE=2), 8)
                fused_equal = new_fused == old_fused
            else:
                fused_equal = None
            if "FAST" in old_names:  # the fast variant of the baseline must also be unchanged
                old_fast = ptx(old_bw._rtiled_bw, dict(old_sig), dict(old_const, FAST=1, DS_C=0, DS_N=0), 8)
                rows_fast_equal = old_fast == fast
            else:
                rows_fast_equal = None
            rows.append(dict(kernel="backward", cell=cell, width=width, slice_width=s,
                             explicit_zero_equal=a_ptx == c_ptx, verify_on_differs=on != a_ptx,
                             fast_compiles_and_differs=fast != a_ptx, fast_equal_to_baseline_fast=rows_fast_equal,
                             fused_equal_to_baseline_fused=fused_equal))
    status = "PASS" if all(r["explicit_zero_equal"] and r["verify_on_differs"]
                           and r.get("fast_equal_to_baseline_fast") is not False
                           and r.get("fused_equal_to_baseline_fused") is not False for r in rows) \
        else "FAIL"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "verify-ptx.json").write_text(json.dumps(dict(rows=rows, status=status), indent=1))
    print(json.dumps(dict(status=status, rows=rows)))


if __name__ == "__main__":
    main()
