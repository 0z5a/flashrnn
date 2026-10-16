"""M3 GPU spike: TileLang cluster+DSMEM R tiling vs the Triton L2-barrier path.

Numerics against the frozen FP64 oracle (fixture_math, CELL_BUDGET) and against
Triton R tiling on identical inputs; exploratory (unpaired) timing only.
"""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
import triton

import fixture_math as fm
from flashrnn.flashrnn2 import tilelang_rtiled, triton_rtiled
from flashrnn.flashrnn2.state_contract import begin_sequence

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}
SOURCES = ["fixture_math.py", "flashrnn/flashrnn2/tilelang_rtiled.py", "flashrnn/flashrnn2/triton_rtiled.py",
           "tools/rtiled_m3_tilelang.py"]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def identity():
    import tilelang
    p = torch.cuda.get_device_properties(0)
    return dict(gpu=p.name, sms=p.multi_processor_count, triton=triton.__version__, tilelang=tilelang.__version__,
                torch=torch.__version__, host=os.uname().nodename, uuid=os.environ.get("FRNN_GPU_UUID"),
                cuda_home=os.environ.get("CUDA_HOME"),
                source={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in SOURCES})


def triton_call(values, cell, s, mode="hidden"):
    ctx = begin_sequence(values[3], cell, values[0].dtype)
    return triton_rtiled.recurrence(*values, cell, sequence_context=ctx, slice_width=s, output_mode=mode)


def numerics(out, args):
    rows = []
    for cell in args.cells:
        for dtype in ("float16", "bfloat16"):
            for width, s in args.shapes:
                for batch, steps, heads in ((3, 17, 2), (16, 128, 1)):
                    for mode in (("warm",) if cell == "lstm" else ("zero", "mixed", "warm")):
                        case = dict(family="m3", cell=cell, dtype=dtype, batch=batch, steps=steps, width=width,
                                    heads=heads, mode=mode)
                        rec = dict(case=fm.case_name(case), slice_width=s, k=width // s)
                        try:
                            fx = fm.make_fixture(case)
                            values = tuple(torch.tensor(fx[k], device="cuda", dtype=DT[dtype])
                                           for k in ("wx", "r", "bias", "initial"))
                            y, carry = tilelang_rtiled.recurrence(*values, cell, slice_width=s)
                            ty, tcarry = triton_call(values, cell, s)
                            torch.cuda.synchronize()
                            ref = fx["reference_fp64"]
                            rec["oracle"] = dict(
                                hidden=fm.compare(y.float().cpu().numpy(), ref[0], fm.CELL_BUDGET),
                                carry=fm.compare(carry.cpu().numpy(), fx["reference_final_fp64"], fm.CELL_BUDGET))
                            rec["vs_triton"] = dict(
                                hidden_bitwise=bool(torch.equal(y.view(torch.int16), ty.view(torch.int16))),
                                carry_bitwise=bool(torch.equal(carry, tcarry)),
                                hidden_max_abs=float((y.float() - ty.float()).abs().max()),
                                carry_max_abs=float((carry - tcarry).abs().max()))
                            rec["status"] = "PASS" if all(v["pass_"] for v in rec["oracle"].values()) \
                                else "NUMERICAL_FAIL"
                        except Exception:  # noqa: BLE001
                            rec.update(status="ERROR", error=traceback.format_exc()[-1500:])
                        rows.append(rec)
                        print(json.dumps({k: rec[k] for k in ("case", "slice_width", "status")}), flush=True)
    write(out / "m3-numerics.json", dict(identity=identity(), budget="fixture_math.CELL_BUDGET", rows=rows))
    return rows


def timed(fn, n):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def explore(out, args):
    rows = []
    for width, s in args.shapes:
        for batch in (16, 64):
            steps = 256
            fx = fm.make_fixture(dict(family="m3t", cell="lstm", dtype="float16", batch=batch, steps=1, width=width,
                                      heads=1, mode="warm"))
            g = torch.Generator(device="cuda").manual_seed(width + batch)
            wx = (torch.randn((batch, steps, 4, 1, width), generator=g, device="cuda") * 0.1).half()
            values = (wx,) + tuple(torch.tensor(fx[k], device="cuda", dtype=torch.float16)
                                   for k in ("r", "bias", "initial"))
            ctx = begin_sequence(values[3], "lstm", torch.float16)
            arms = {
                "tilelang_cluster": lambda: tilelang_rtiled.recurrence(*values, "lstm", slice_width=s),
                "triton_grid": lambda: triton_rtiled.recurrence(*values, "lstm", sequence_context=ctx, slice_width=s,
                                                                output_mode="hidden"),
            }
            rec = dict(width=width, slice_width=s, k=width // s, batch=batch, steps=steps, rounds=[])
            try:
                for fn in arms.values():
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()
                for r in range(8):
                    order = list(arms) if r % 2 == 0 else list(arms)[::-1]
                    rec["rounds"].append({name: timed(arms[name], 10) for name in order})
                med = {name: sorted(x[name] for x in rec["rounds"])[len(rec["rounds"]) // 2] for name in arms}
                rec.update(median_ms=med, us_per_step={k: v * 1000 / steps for k, v in med.items()},
                           status="MEASURED")
            except Exception:  # noqa: BLE001
                rec.update(status="ERROR", error=traceback.format_exc()[-1500:])
            rows.append(rec)
            print(json.dumps({k: rec[k] for k in ("width", "slice_width", "batch", "status")}
                             | ({"us_per_step": rec["us_per_step"]} if "us_per_step" in rec else {})), flush=True)
    write(out / "m3-explore-timing.json", dict(
        identity=identity(), rows=rows,
        note="exploratory, unpaired; both arms allocate outputs per call; TileLang arm has no chunk/full-state "
             "contract yet; not a speedup claim"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cells", default="lstm,slstm")
    p.add_argument("--shapes", default="128:64,192:64,256:64,384:64,384:32,512:32")
    p.add_argument("--stages", default="numerics,explore")
    args = p.parse_args()
    args.cells = args.cells.split(",")
    args.shapes = [tuple(map(int, x.split(":"))) for x in args.shapes.split(",")]
    out = Path(args.out)
    torch.backends.cuda.matmul.allow_tf32 = False
    write(out / "identity.json", identity())
    if "numerics" in args.stages:
        numerics(out, args)
    if "explore" in args.stages:
        explore(out, args)
    write(out / "DONE.json", dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


if __name__ == "__main__":
    main()
