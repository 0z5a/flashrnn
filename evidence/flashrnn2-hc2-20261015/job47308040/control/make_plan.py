"""Write plan.json for HC round 2: profiling, the occupancy plane (CTA size) screen and the launch plane (CUDA Graph).

Run once while preparing; the output is sealed with the batch.
"""
import json
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
R1 = json.loads((CONTROL / "r1best.json").read_text())  # best per (cell, D, C) from job 47307383's frozen table
CELLS, STEPS = ("lstm", "slstm"), 128
NCU = ("/apps/ACC/CUDA/12.8/bin/ncu --target-processes application-only --launch-skip 3 --launch-count 1 "
       "-k regex:main_kernel|_rtiled --section SpeedOfLight --section Occupancy --section LaunchStats "
       "--section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis")
NSYS = "/apps/ACC/CUDA/12.8/bin/nsys profile -t cuda --stats=true --force-overwrite true -o {out}/trace"


def tl(sync, s, bm, warps=4):
    return dict(family="tilelang", batch_tile=bm, warps=warps, weight_operand="b", slice_width=s, sync=sync,
                pad_k=False)


def tri(s, w):
    return dict(family="mma_rtiled", batch_tile=16, warps=w, weight_operand="b", slice_width=s, sync="grid",
                pad_k=False)


def main():
    profiles = [dict(name="d256-tri-s64-w8", cell="lstm", width=256, config=tri(64, 8)),
                dict(name="d256-tl-cluster-s16-bm16", cell="lstm", width=256, config=tl("cluster", 16, 16)),
                dict(name="d256-tl-cluster-s32-bm64", cell="lstm", width=256, config=tl("cluster", 32, 64)),
                dict(name="d768-tri-s32-w4", cell="lstm", width=768, config=tri(32, 4)),
                dict(name="d768-tl-grid-s16-bm64", cell="lstm", width=768, config=tl("grid", 16, 64))]

    wide = {256: [("cluster", 16, 32), ("cluster", 16, 64), ("cluster", 32, 32), ("cluster", 32, 64), ("cluster", 64, 32)],
            768: [("grid", 16, 32), ("grid", 16, 64)]}
    cases, lane = [], 0
    for width in (256, 768):
        for cell in CELLS:
            for conc in (256, 1024):
                best = R1[f"{cell}/{width}/{conc}"]
                ref = dict(best["spec"], label="r1_best")
                ref.pop("wide", None)
                cands = [ref]
                labels128, labels256 = ["r1_best"], []
                for sync, s, bm in wide[width]:
                    a, b = f"tl_{sync}_s{s}_bm{bm}_w4", f"tl_{sync}_s{s}_bm{bm}_w8"
                    cands.append(dict(label=a, strategy="s0p", group=conc, config=tl(sync, s, bm, 4)))
                    cands.append(dict(label=b, strategy="s0p", group=conc, config=tl(sync, s, bm, 8), twin=a))
                    labels128.append(a)
                    labels256.append(b)
                cases.append(dict(name=f"hc2-{cell}-d{width}-c{conc}-t{STEPS}", lane=lane, cell=cell, width=width,
                                  steps=STEPS, concurrency=conc, reference="r1_best", candidates=cands,
                                  roles=dict(best=labels128 + labels256, best_w4=labels128, best_w8=labels256)))
                lane = (lane + 1) % 4

    comps, lane = [], 0
    for width in (256, 768):
        for cell in CELLS:
            for conc in (256, 1024):
                spec = dict(R1[f"{cell}/{width}/{conc}"]["spec"])
                spec.pop("wide", None)
                spec.pop("label", None)
                comps.append(dict(name=f"hc2-{cell}-d{width}-c{conc}-t{STEPS}-graph-vs-eager", experiment="HC_GRAPH_VS_EAGER",
                                  cell=cell, width=width, batch=conc, concurrency=conc, steps=STEPS, lane=lane,
                                  baseline=spec, candidate=dict(spec, graph=True),
                                  baseline_label=R1[f"{cell}/{width}/{conc}"]["label"],
                                  candidate_label=R1[f"{cell}/{width}/{conc}"]["label"] + "+graph"))
                lane = (lane + 1) % 4

    plan = dict(
        batch="flashrnn2-hc2-20261015-v1",
        purpose=("High-concurrency round 2. Profiling of five recurrence configurations at B = 1024, T = 128 (ncu if "
                 "counters are permitted, else an nsys timeline of the s0p forward); occupancy plane screen: TileLang "
                 "R tiling with two warpgroups per CTA (256 threads) vs one, at the round-1 batch tiles, bitwise twin "
                 "checks and split-0 makespans; launch plane: the round-1 best configuration served through a CUDA Graph "
                 "of the whole 4-layer forward vs eager, paired split-1 timing"),
        stop_criterion=("at C = 1024 for D = 256 and D = 768, LSTM and sLSTM: throughput >= 2x max(s0, s0p) and P90 "
                        "no worse than s0"),
        minutes=150, starts=5, blocks=20, select_reps=5, tk_root="", tk_cache="/scratch/tmp/tueb043681-hc2-{job}-tk",
        ncu=NCU, nsys=NSYS, profiles=profiles, select_cases=cases, comparisons=comps,
        node_reason="One ACC node (4 H100): profiling lanes, then screening lanes, then paired timing lanes",
        not_claimed=["256-thread variants are screened, not qualified: no paired claim for them in this round",
                     "arrival processes other than a closed batch", "multi-GPU", "model quality"],
    )
    (CONTROL / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(len(profiles), len(cases), len(comps))


if __name__ == "__main__":
    main()
