"""M1 GPU checks: R tiling through PreparedSequence (fixed buffers, chunks, CUDA Graphs)."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
import triton

from flashrnn.flashrnn2 import triton_rtiled
from flashrnn.flashrnn2.execution_policy import KernelConfig, default_config
from flashrnn.flashrnn2.prepared import PreparedSequence
from flashrnn.flashrnn2.state_contract import begin_sequence

SOURCES = ["flashrnn/flashrnn2/prepared.py", "flashrnn/flashrnn2/execution_policy.py",
           "flashrnn/flashrnn2/triton_rtiled.py", "tools/rtiled_m1_prepared.py"]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8),
                                                                      b.contiguous().view(torch.uint8))


def fixture(cell, dtype, batch, steps, heads, width, seed):
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    wx = torch.randn(batch, steps, 4, heads, width, generator=g) * 0.5
    r = torch.randn(4, heads, width, width, generator=g) / width ** 0.5
    bias = torch.randn(4, heads, width, generator=g) * 0.1
    initial = torch.randn(states, batch, 1, heads, width, generator=g) * 0.1
    if cell == "slstm":
        initial[2] = initial[2].abs() + 1
    return tuple(x.to(dtype).cuda().contiguous() for x in (wx, r, bias, initial))


def chunk_plan(steps):
    out, acc = [], 0
    for n in (1, 3, 13, 7, 9, 64):
        if acc + n >= steps:
            break
        out.append(n)
        acc += n
    return out + [steps - acc]


def case(cell, dtype, batch, steps, heads, width, cfg):
    values = fixture(cell, dtype, batch, steps, heads, width, width + batch)
    ctx = begin_sequence(values[3], cell, dtype)
    checks = {}
    for mode in ("full", "hidden", "last"):
        ref = triton_rtiled.recurrence(*values, cell, sequence_context=ctx, slice_width=cfg.slice_width,
                                       num_warps=cfg.warps, output_mode=mode, snapshot_dtype=torch.float32)
        p = PreparedSequence(*values, cell, output_mode=mode, snapshot_dtype=torch.float32, config=cfg)
        y, carry = p.run()
        p.check()
        checks[f"{mode}_eager_bitwise"] = same(y, ref[0]) and same(carry, ref[1])
        pc = PreparedSequence(*values, cell, output_mode=mode, snapshot_dtype=torch.float32, config=cfg,
                              chunks=chunk_plan(steps))
        yc, cc = pc.run()
        pc.check()
        checks[f"{mode}_chunks_bitwise"] = same(yc, ref[0]) and same(cc, ref[1])
        pg = PreparedSequence(*values, cell, output_mode=mode, snapshot_dtype=torch.float32, config=cfg,
                              chunks=chunk_plan(steps))
        try:
            pg.capture()
            ok = True
            for _ in range(20):
                yg, cg = pg.run()
                ok &= same(yg, ref[0]) and same(cg, ref[1])
            pg.check()
            checks[f"{mode}_graph_replay_bitwise"] = ok
        except Exception as e:  # noqa: BLE001 - capture support is a recorded result
            checks[f"{mode}_graph_replay_bitwise"] = False
            checks[f"{mode}_graph_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            pg.close()
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    ident = dict(gpu=torch.cuda.get_device_name(), triton=triton.__version__, torch=torch.__version__,
                 uuid=os.environ.get("FRNN_GPU_UUID"),
                 source={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in SOURCES})
    rows = []
    for cell in ("lstm", "slstm"):
        for dtype in (torch.float16, torch.bfloat16):
            for width in (128, 192, 256, 384, 512, 768):
                for batch, steps, heads in ((3, 37, 2), (40, 64, 1)):
                    cfg = default_config(width)
                    if cfg.family != "mma_rtiled":
                        cfg = KernelConfig(family="mma_rtiled", warps=8, slice_width=64, sync="grid")
                    rec = dict(cell=cell, dtype=str(dtype), width=width, batch=batch, steps=steps, heads=heads,
                               config=cfg.as_dict())
                    try:
                        rec["checks"] = case(cell, dtype, batch, steps, heads, width, cfg)
                        rec["status"] = "PASS" if all(v is True for k, v in rec["checks"].items()
                                                      if not k.endswith("_error")) else "FAIL"
                    except Exception:  # noqa: BLE001
                        rec.update(status="ERROR", error=traceback.format_exc()[-1500:])
                    rows.append(rec)
                    print(json.dumps({k: rec[k] for k in ("cell", "dtype", "width", "batch", "status")}), flush=True)
    write(out / "m1-prepared.json", dict(identity=ident, rows=rows))
    write(out / "DONE.json", dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


if __name__ == "__main__":
    main()
