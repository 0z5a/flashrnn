"""Write plan.json for HC round 3: input-projection prefetch (latency plane) and two-warpgroup grid CTAs.

Run once while preparing; the output is sealed with the batch.
"""
import json
import sys
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else CONTROL.parent / "source"
R1 = json.loads((CONTROL / "r1best.json").read_text())  # best per (cell, D, C), job 47307383
CELLS, STEPS, GROUP0 = ("lstm", "slstm"), 128, 32
STAGES = "e1,protocol,stress,vs_ref"
NEW = {256: [("cluster", 16, 16, 4, True), ("cluster", 16, 32, 4, True), ("cluster", 32, 16, 4, True),
             ("cluster", 32, 32, 4, True)],
       768: [("grid", 16, 32, 8, False), ("grid", 16, 32, 8, True), ("grid", 16, 64, 8, False),
             ("grid", 16, 64, 8, True)]}
QUALIFIED = {256: [("cluster", 16, 16), ("cluster", 32, 16), ("cluster", 16, 32), ("cluster", 32, 32)],  # M6, HC1 v2
             768: [("grid", 32, 16), ("grid", 16, 32), ("grid", 16, 64)]}  # M7, HC1 v2


def tl(sync, s, bm, warps=4, prefetch=False):
    return dict(family="tilelang", batch_tile=bm, warps=warps, weight_operand="b", slice_width=s, sync=sync,
                pad_k=False, prefetch=prefetch)


def tri(s, w):
    return dict(family="mma_rtiled", batch_tile=16, warps=w, weight_operand="b", slice_width=s, sync="grid",
                pad_k=False, prefetch=False)


def policy_config(cell, width, batch):
    key = f"{cell}/float16/b{batch}/t{STEPS}/h1/d{width}"
    for name in ("h100-rtiled-m6-20261014.json", "h100-rtiled-m7-20261014.json"):
        table = json.loads((SOURCE / "flashrnn/flashrnn2/tables" / name).read_text())["table"]
        if key in table:
            e = table[key]
            if e["family"] == "tilelang":
                return tl(e["sync"], e["slice_width"], e.get("batch_tile", 16))
            return tri(e["slice_width"], e["warps"])
    return tri(64 if width <= 256 else 32, 8)


def with_prefetch_field(cfg):
    return dict(cfg, prefetch=cfg.get("prefetch", False))


def main():
    first = ["lstm", "bfloat16", 768, 16, "grid", 64, 8, True, STAGES]
    quals = [first]
    for width, variants in NEW.items():
        for sync, s, bm, warps, pf in variants:
            for cell in CELLS:
                for dtype in ("bfloat16", "float16"):
                    q = [cell, dtype, width, s, sync, bm, warps, pf, STAGES]
                    if q != first:
                        quals.append(q)

    cases, comps, lane = [], [], 0
    for width in (256, 768):
        for cell in CELLS:
            for conc in (256, 1024):
                r1 = R1[f"{cell}/{width}/{conc}"]
                cands = [dict(label="s0", strategy="s0", group=GROUP0, config=policy_config(cell, width, GROUP0)),
                         dict(label="s0p_default", strategy="s0p", group=conc, config=policy_config(cell, width, conc)),
                         dict(label="r1_best", strategy="s0p", group=conc, config=with_prefetch_field(r1["spec"]["config"]))]
                tris = [(64, 4)] if width == 256 else [(32, 4)]
                for s_, w in tris:
                    cands.append(dict(label=f"tri_s{s_}_w{w}", strategy="s0p", group=conc, config=tri(s_, w)))
                for sync, s_, bm in QUALIFIED[width]:
                    cands.append(dict(label=f"tl_{sync}_s{s_}_bm{bm}_w4", strategy="s0p", group=conc,
                                      config=tl(sync, s_, bm)))
                for sync, s_, bm, warps, pf in NEW[width]:
                    cands.append(dict(label=f"tl_{sync}_s{s_}_bm{bm}_w{warps}" + ("_pf" if pf else ""), strategy="s0p",
                                      group=conc, config=tl(sync, s_, bm, warps, pf), wide=True))
                labels = [c["label"] for c in cands]
                name = f"hc3-{cell}-d{width}-c{conc}-t{STEPS}"
                cases.append(dict(name=name, lane=lane, cell=cell, width=width, steps=STEPS, concurrency=conc,
                                  reference="s0p_default", candidates=cands,
                                  roles=dict(best=[l for l in labels if l != "s0"], s0p_default=["s0p_default"],
                                             r1_best=["r1_best"], s0=["s0"])))
                base = dict(cell=cell, width=width, batch=conc, concurrency=conc, steps=STEPS, select_case=name, lane=lane)
                comps.append(dict(base, name=f"{name}-best-vs-s0p", experiment="HC_BEST_VS_S0P",
                                  baseline_role="s0p_default", candidate_role="best"))
                comps.append(dict(base, name=f"{name}-best-vs-r1", experiment="HC_BEST_VS_R1",
                                  baseline_role="r1_best", candidate_role="best"))
                if conc == 1024:
                    comps.append(dict(base, name=f"{name}-best-vs-s0", experiment="HC_BEST_VS_S0",
                                      baseline_role="s0", candidate_role="best"))
                lane = (lane + 1) % 4

    cs = "/apps/ACC/CUDA/12.8/bin/compute-sanitizer --target-processes application-only --error-exitcode 99"
    runs = []
    for case in (["lstm", "float16", 256, 16, "cluster", 16, 4, True], ["lstm", "float16", 768, 16, "grid", 64, 8, True]):
        tag = f"{case[4]}-d{case[2]}-bm{case[5]}-w{case[6]}-pf"
        runs += [dict(tool=f"memcheck-{tag}", prefix=f"{cs} --tool memcheck", stages="sanity", blocking=True, case=case),
                 dict(tool=f"synccheck-{tag}", prefix=f"{cs} --tool synccheck", stages="sanity", blocking=True,
                      case=case)]
    plan = dict(
        batch="flashrnn2-hc3-20261015-v2",
        purpose=("High-concurrency round 3. Latency plane: TileLang kernels that load the next step's input "
                 "projections into registers after each epilogue (ncu, job 47308040: long-scoreboard stalls 40-55%); "
                 "occupancy plane: two-warpgroup CTAs on the grid path (screened 1.26x at D768 C1024). Qualified at the "
                 "M0 + M6 standard plus bitwise equality with the reference kernel; split-0 selection; split-1 paired "
                 "timing of the best against s0p_default, against the round-1 best and against s0 at C = 1024"),
        stop_criterion=("at C = 1024 for D = 256 and D = 768, LSTM and sLSTM: throughput >= 2x max(s0, s0p) and P90 "
                        "no worse than s0"),
        minutes=160, starts=5, blocks=20, qualification_timeout_s=2400, select_reps=5, tk_root="",
        tk_cache="/scratch/tmp/tueb043681-hc3-{job}-tk",
        qualifications=quals, sanitizer_runs=runs, select_cases=cases, comparisons=comps,
        node_reason="One ACC node (4 H100): qualification lanes in parallel, then selection and paired timing per lane",
        not_claimed=["Triton kernels unchanged (no prefetch)", "arrival processes other than a closed batch",
                     "multi-GPU", "model quality", "backward"],
    )
    (CONTROL / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(len(quals), len(cases), len(comps), sum(len(c["candidates"]) for c in cases))


if __name__ == "__main__":
    main()
