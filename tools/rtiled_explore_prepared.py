"""Exploratory prepared-graph timing for R tiling (candidate screening, not a speedup claim).

Every arm is a captured PreparedSequence (fixed buffers, CUDA Graph replay),
hidden output, FP16, H=1, T=256. Arms of one shape are interleaved in rotating
order; medians are reported. The formal E2/E3 protocol is separate.
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

from flashrnn.flashrnn2.execution_policy import KernelConfig
from flashrnn.flashrnn2.prepared import PreparedSequence

SOURCES = ["flashrnn/flashrnn2/prepared.py", "flashrnn/flashrnn2/triton_rtiled.py",
           "flashrnn/flashrnn2/triton_persistent.py", "tools/rtiled_explore_prepared.py"]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def arms_for(width, configs):
    arms = {}
    if width <= 128:
        arms["single_mma4"] = KernelConfig()
        arms["single_mma8"] = KernelConfig(warps=8)
    for c in configs:
        if c["width"] == width and c["cell"] == "lstm" and not c["pad_k"]:
            name = f"rtiled_s{c['slice_width']}_w{c['warps']}"
            arms[name] = KernelConfig(family="mma_rtiled", warps=c["warps"], slice_width=c["slice_width"],
                                      sync="grid")
    return arms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--configs", required=True)
    p.add_argument("--widths", default="64,96,128,192,256,384,512,768")
    p.add_argument("--cell", default="lstm")
    args = p.parse_args()
    out = Path(args.out)
    configs = json.loads(Path(args.configs).read_text())
    ident = dict(gpu=torch.cuda.get_device_name(), triton=triton.__version__, torch=torch.__version__,
                 uuid=os.environ.get("FRNN_GPU_UUID"),
                 source={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in SOURCES})
    rows = []
    for width in map(int, args.widths.split(",")):
        for batch in (16, 64):
            steps = 256
            g = torch.Generator().manual_seed(width * 7 + batch)
            states = 4 if args.cell == "slstm" else 2
            values = (
                (torch.randn(batch, steps, 4, 1, width, generator=g) * 0.1).half().cuda(),
                (torch.randn(4, 1, width, width, generator=g) * 0.1 / width ** 0.5).half().cuda(),
                (torch.randn(4, 1, width, generator=g) * 0.03).half().cuda(),
                (torch.randn(states, batch, 1, 1, width, generator=g) * 0.05).half().cuda(),
            )
            if args.cell == "slstm":
                values[3][2].abs_().add_(1)
            rec = dict(cell=args.cell, width=width, batch=batch, steps=steps, arms={}, rounds=[])
            prepared = {}
            try:
                for name, cfg in arms_for(width, configs).items():
                    try:
                        prepared[name] = PreparedSequence(*values, args.cell, output_mode="hidden",
                                                          config=cfg).capture()
                        rec["arms"][name] = cfg.as_dict()
                    except Exception as e:  # noqa: BLE001
                        rec["arms"][name] = dict(cfg.as_dict(), error=f"{type(e).__name__}: {str(e)[:200]}")
                names = list(prepared)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for r in range(12):
                    order = names[r % len(names):] + names[:r % len(names)]
                    row = {}
                    for name in order:
                        prepared[name].run()
                        torch.cuda.synchronize()
                        start.record()
                        for _ in range(10):
                            prepared[name].run()
                        end.record()
                        torch.cuda.synchronize()
                        row[name] = start.elapsed_time(end) / 10
                    rec["rounds"].append(row)
                for name in names:
                    prepared[name].check()
                med = {n: sorted(x[n] for x in rec["rounds"])[len(rec["rounds"]) // 2] for n in names}
                rec.update(median_ms=med, us_per_step={n: v * 1000 / steps for n, v in med.items()},
                           fastest=min(med, key=med.get), status="MEASURED")
            except Exception:  # noqa: BLE001
                rec.update(status="ERROR", error=traceback.format_exc()[-1500:])
            finally:
                for p_ in prepared.values():
                    p_.close()
            rows.append(rec)
            print(json.dumps({k: rec.get(k) for k in ("width", "batch", "status", "fastest", "us_per_step")}),
                  flush=True)
    write(out / "explore-prepared.json", dict(identity=ident, rows=rows,
                                              note="exploratory screening; not the paired E2/E3 protocol"))
    write(out / "DONE.json", dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


if __name__ == "__main__":
    main()
