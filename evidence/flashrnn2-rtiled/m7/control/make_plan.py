"""Write plan.json for M7: the TileLang cooperative-grid path (sync_grid) for D = 768.

Run once while preparing; the output is sealed with the batch.
"""
import json
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
M0 = json.loads((CONTROL / "m0-configs.json").read_text())
CELLS = ("lstm", "slstm")
D = 768


def triton_specs(cell):
    return [dict(kind="rtiled", slice_width=c["slice_width"], warps=c["warps"], pad_k=c["pad_k"])
            for c in M0 if c["cell"] == cell and c["width"] == D]


def main():
    quals = [["lstm", "bfloat16", D, 32, "grid", "e1,protocol,stress"]]  # first: largest shared-memory footprint
    for cell in CELLS:
        for dtype in ("bfloat16", "float16"):
            for s in (32, 16):
                q = [cell, dtype, D, s, "grid", "e1,protocol,stress"]
                if q not in quals:
                    quals.append(q)
    # Widths the cluster path also covers: the grid exchange must reproduce it bit for bit.
    for cell in CELLS:
        for dtype in ("bfloat16", "float16"):
            for width, s in ((256, 32), (512, 32), (256, 16)):
                quals.append([cell, dtype, width, s, "grid", "e1,protocol,stress,vs_cluster"])

    tilelang = [dict(kind="tilelang_full", slice_width=s, sync="grid") for s in (16, 32)]
    cases, comps, lane = [], [], 0
    shapes = [("kernel", b, b, 1024) for b in (16, 64)] + [("model", b, c, t) for b, c, t in
                                                           ((1, 8, 32), (16, 32, 32), (32, 64, 128))]
    for cell in CELLS:
        for level, batch, conc, steps in shapes:
            name = (f"k7-{cell}-d{D}-b{batch}-t{steps}" if level == "kernel" else
                    f"m7-{cell}-d{D}-b{batch}-c{conc}-t{steps}")
            tri = triton_specs(cell)
            case = dict(name=name, lane=lane, cell=cell, width=D, batch=batch, concurrency=conc, steps=steps,
                        roles=dict(default=[dict(kind="rtiled", slice_width=32, warps=8, pad_k=False)],
                                   policy=tri + tilelang, triton=tri, tilelang=tilelang))
            if level == "kernel":
                case["level"] = "kernel"
            cases.append(case)
            prefix = "K_" if level == "kernel" else ""
            base = dict(cell=cell, width=D, batch=batch, concurrency=conc, steps=steps, select_case=name, lane=lane)
            if level == "kernel":
                base["level"] = "kernel"
            comps.append(dict(base, name=f"{name}-policy-vs-default", experiment=f"{prefix}POLICY_VS_DEFAULT",
                              baseline=dict(from_role="default"), candidate_role="policy"))
            comps.append(dict(base, name=f"{name}-tlgrid-vs-triton", experiment=f"{prefix}TLGRID_VS_TRITON",
                              baseline=dict(from_role="triton"), candidate_role="tilelang"))
            lane = (lane + 1) % 4

    cs = "CUDA 12.8 compute-sanitizer --target-processes application-only --error-exitcode 99"
    plan = dict(
        batch="flashrnn2-rtiled-m7-20261014-v1",
        purpose=("D = 768 with the TileLang cooperative-grid path (T.sync_grid, k = 24 or 48 CTAs per group): "
                 "full-contract qualification at the M0 + M6 standard, bitwise agreement with the cluster path "
                 "where both exist, then split-0 selection and split-1 paired timing against the conservative "
                 "default and against the best Triton R-tiled configuration"),
        protocol=("phase 0: fixture_math CELL_BUDGET E1 (FP16/BF16; full/hidden/last; sLSTM zero/mixed/warm), bitwise "
                  "per-step launches = ragged chunks = public chunked calls = one call, hidden/last = full, 20 repeats, "
                  "side stream with producer, prepared CUDA Graph replay and bound-initial tracking, 50-repeat stress "
                  "with grid slots, grid = cluster bitwise at D = 256/512; compute-sanitizer memcheck and synccheck "
                  "blocking, racecheck recorded; phase 0 stops the batch when the first configuration ERRORs, a "
                  "blocking sanitizer is not clean, or no FP16 D = 768 TileLang configuration qualifies; phase 1: "
                  "split-0 selection; frozen comparison + policy tables; phase 2: split-1 5 starts x 20 ABBA/BAAB "
                  "blocks, policy served through ExecutionPolicy.select"),
        minutes=160, starts=5, blocks=20, qualification_timeout_s=2400, tk_root="",
        tk_cache="<node-local scratch>",
        qualifications=quals,
        sanitizer_case=["lstm", "float16", D, 32, "grid"],
        sanitizer_runs=[
            dict(tool="memcheck", prefix=f"{cs} --tool memcheck", stages="sanity", blocking=True),
            dict(tool="synccheck", prefix=f"{cs} --tool synccheck", stages="sanity", blocking=True),
            dict(tool="racecheck", prefix=f"{cs} --tool racecheck --racecheck-report all", stages="sanity",
                 blocking=False),
        ],
        select_cases=cases, comparisons=comps,
        node_reason=("One H100 node (4 H100): a D = 768 group needs 24 or 48 SMs of one GPU; qualification lanes in "
                     "parallel, then selection and paired timing pinned to GPU-local cores"),
        deviations=["racecheck counts as clean when its RACECHECK SUMMARY reports 0 errors and 0 warnings (M6 v2 "
                    "receipt mislabelled it because the rule only looked for ERROR SUMMARY)"],
        not_claimed=["E2E model quality", "backward", "multi-GPU", "shapes outside the timed set"],
    )
    (CONTROL / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(len(quals), len(cases), len(comps))


if __name__ == "__main__":
    main()
