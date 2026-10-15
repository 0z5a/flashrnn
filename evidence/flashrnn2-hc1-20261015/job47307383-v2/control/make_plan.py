"""Write plan.json for HC round 1: the SM-cost plane (TileLang batch tile BM > 16) at high concurrency.

Run once while preparing; the output is sealed with the batch.
"""
import json
import sys
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else CONTROL.parent / "source"
M0 = json.loads((CONTROL / "m0-configs.json").read_text())
CELLS = ("lstm", "slstm")
STEPS, GROUP0 = 128, 32
# (sync, D, S, BM) variants the shared memory admits (tilelang_cluster.supported / supported_grid).
# Cluster BM = 128 is excluded (v1, job 47306925: the swizzled h^T layout breaks the per-peer bulk copy).
WIDE = {256: [("cluster", 16, 32), ("cluster", 16, 64), ("cluster", 32, 32),
              ("cluster", 32, 64), ("cluster", 64, 32)],
        768: [("grid", 16, 32), ("grid", 16, 64)]}


def tl(sync, s, bm):
    return dict(family="tilelang", batch_tile=bm, warps=4, weight_operand="b", slice_width=s, sync=sync, pad_k=False)


def tri(s, w, pad):
    return dict(family="mma_rtiled", batch_tile=16, warps=w, weight_operand="b", slice_width=s, sync="grid", pad_k=pad)


def policy_config(cell, width, batch):
    """What ExecutionPolicy.select serves for this exact shape from the M6 + M7 tables (default otherwise)."""
    key = f"{cell}/float16/b{batch}/t{STEPS}/h1/d{width}"
    for name in ("h100-rtiled-m6-20261014.json", "h100-rtiled-m7-20261014.json"):
        table = json.loads((SOURCE / "flashrnn/flashrnn2/tables" / name).read_text())["table"]
        if key in table:
            e = table[key]
            if e["family"] == "tilelang":
                return tl(e["sync"], e["slice_width"], e.get("batch_tile", 16))
            return tri(e["slice_width"], e["warps"], e.get("pad_k", False))
    return tri(64 if width <= 256 else 32, 8, False)  # default_config(width)


def main():
    quals, first = [], ["lstm", "bfloat16", 256, 16, "cluster", 64, "e1,protocol,stress,vs_bm16"]
    quals.append(first)
    for width, variants in WIDE.items():
        for sync, s, bm in variants:
            for cell in CELLS:
                for dtype in ("bfloat16", "float16"):
                    q = [cell, dtype, width, s, sync, bm, "e1,protocol,stress,vs_bm16"]
                    if q != first:
                        quals.append(q)

    cases, comps, lane = [], [], 0
    for width in (256, 768):
        for cell in CELLS:
            for conc in (256, 1024):
                cands = [dict(label="s0", strategy="s0", group=GROUP0, config=policy_config(cell, width, GROUP0)),
                         dict(label="s0p_default", strategy="s0p", group=conc, config=policy_config(cell, width, conc))]
                bm16 = ["s0p_default"]
                for c in M0:
                    if c["cell"] == cell and c["width"] == width:
                        lab = f"tri_s{c['slice_width']}_w{c['warps']}" + ("_pad" if c["pad_k"] else "")
                        cfg = tri(c["slice_width"], c["warps"], c["pad_k"])
                        if cfg != cands[1]["config"]:
                            cands.append(dict(label=lab, strategy="s0p", group=conc, config=cfg))
                            bm16.append(lab)
                for sync, s in ((("cluster", 16), ("cluster", 32), ("cluster", 64)) if width == 256 else
                                (("grid", 16), ("grid", 32))):
                    lab = f"tl_{sync}_s{s}_bm16"
                    cands.append(dict(label=lab, strategy="s0p", group=conc, config=tl(sync, s, 16)))
                    bm16.append(lab)
                wide = []
                for sync, s, bm in WIDE[width]:
                    lab = f"tl_{sync}_s{s}_bm{bm}"
                    cands.append(dict(label=lab, strategy="s0p", group=conc, config=tl(sync, s, bm), wide=True))
                    wide.append(lab)
                name = f"hc1-{cell}-d{width}-c{conc}-t{STEPS}"
                cases.append(dict(name=name, lane=lane, cell=cell, width=width, steps=STEPS, concurrency=conc,
                                  reference="s0p_default", candidates=cands,
                                  roles=dict(best=bm16 + wide, best_bm16=bm16, s0=["s0"], s0p_default=["s0p_default"])))
                base = dict(cell=cell, width=width, batch=conc, concurrency=conc, steps=STEPS, select_case=name,
                            lane=lane)
                comps.append(dict(base, name=f"{name}-best-vs-s0p", experiment="HC_BEST_VS_S0P",
                                  baseline_role="s0p_default", candidate_role="best"))
                comps.append(dict(base, name=f"{name}-best-vs-bm16", experiment="HC_BEST_VS_BEST_BM16",
                                  baseline_role="best_bm16", candidate_role="best"))
                if conc == 1024:
                    comps.append(dict(base, name=f"{name}-best-vs-s0", experiment="HC_BEST_VS_S0",
                                      baseline_role="s0", candidate_role="best"))
                lane = (lane + 1) % 4

    cs = "/apps/ACC/CUDA/12.8/bin/compute-sanitizer --target-processes application-only --error-exitcode 99"
    sanitizer_cases = [["lstm", "float16", 256, 16, "cluster", 64], ["lstm", "float16", 768, 16, "grid", 64]]
    runs = []
    for case in sanitizer_cases:
        tag = f"{case[4]}-d{case[2]}-bm{case[5]}"
        runs += [dict(tool=f"memcheck-{tag}", prefix=f"{cs} --tool memcheck", stages="sanity", blocking=True, case=case),
                 dict(tool=f"synccheck-{tag}", prefix=f"{cs} --tool synccheck", stages="sanity", blocking=True,
                      case=case)]
    plan = dict(
        batch="flashrnn2-hc1-20261015-v2",
        purpose=("High-concurrency round 1, SM-cost plane: TileLang R tiling with batch tiles BM > 16 (rows per CTA "
                 "per step, WGMMA N), qualified at the M0 + M6 standard plus bitwise agreement with BM = 16; then, for "
                 "C requests of 128 tokens arriving together (C = 256, 1024; D = 256, 768; LSTM and sLSTM), split-0 "
                 "selection among s0, s0p_default, every BM = 16 configuration at B = C and the qualified BM > 16 "
                 "configurations, and split-1 paired timing of the best against s0p_default (headline), against the "
                 "best BM = 16 configuration (plane attribution) and against s0 at C = 1024 (latency)"),
        deviations=["v2 of job 47306925: cluster BM = 128 removed; sanitizer on the largest remaining cluster tile; a "
                    "sanitizer finding now excludes its configuration instead of stopping the batch"],
        stop_criterion=("at C = 1024 for D = 256 and D = 768, LSTM and sLSTM: throughput >= 2x max(s0, s0p) and P90 "
                        "no worse than s0"),
        minutes=160, starts=5, blocks=20, qualification_timeout_s=2400, select_reps=5, tk_root="",
        tk_cache="/scratch/tmp/tueb043681-hc1-{job}-tk",
        qualifications=quals, sanitizer_runs=runs, select_cases=cases, comparisons=comps,
        node_reason="One ACC node (4 H100): qualification lanes in parallel, then selection and paired timing per lane",
        not_claimed=["arrival processes other than a closed batch", "multi-GPU", "model quality", "backward"],
    )
    (CONTROL / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(len(quals), len(cases), len(comps), sum(len(c["candidates"]) for c in cases))


if __name__ == "__main__":
    main()
