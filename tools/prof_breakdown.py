"""max-width 74.2(a): kernel breakdown of one LSTM FP16 layer's training step at D = 1024, B = 16, for cuDNN and
FlashRNN2 (torch.profiler / CUPTI): per kernel name, launches and total device time per step."""
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import e2e_bench

# The forward FlashRNN2 selects on its measured runtime (training.TILELANG_TRAINING_FORWARD), pinned so that a
# runtime without the packaged tables does not fall back to the untuned R-tiled default.
FWD = {512: dict(family="tilelang", slice_width=32, sync="cluster"), 768: dict(family="tilelang", slice_width=16,
       sync="grid"), 1024: dict(family="tilelang", slice_width=16, sync="grid")}
case = dict(mode="layer", cell="lstm", dtype="float16", width=int(sys.argv[2]), batch=int(sys.argv[3]), steps=256,
            arms=["frnn2", "cudnn"], calls=3, config=FWD[int(sys.argv[2])])
ctx = e2e_bench.contexts(case, case["arms"])
state = e2e_bench.build(case, case["arms"], 4242)
out = dict(case=case, config=ctx["cfg"].as_dict(), arms={})
STEPS = 5
for arm in case["arms"]:
    step = e2e_bench.make_step(case, state, arm, ctx)
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(STEPS):
            step()
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        dev = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        if dev and e.count:
            rows.append(dict(name=e.key[:120], launches_per_step=e.count / STEPS, us_per_step=dev / STEPS))
    rows.sort(key=lambda r: -r["us_per_step"])
    out["arms"][arm] = dict(total_us_per_step=sum(r["us_per_step"] for r in rows), kernels=rows[:25])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
for arm, v in out["arms"].items():
    print(arm, round(v["total_us_per_step"]), "us/step")
    for r in v["kernels"][:8]:
        print("   ", round(r["us_per_step"]), "us", r["launches_per_step"], "x", r["name"][:90])
