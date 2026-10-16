"""Write plan.json for the fused-preactivation backward batch (see run_fuse.py for the phases).

The B = 16 confirmation (job 47347074) left FlashRNN2's training step 9-13% behind FlashRNN1 at D = 384/512, with
the T-proportional preactivation GEMM and copies as the largest remaining parts. This batch qualifies the backward
kernel that recomputes the gate preactivations itself (FUSE = 1: one MMA per gate; FUSE = 2: one MMA over the four
gates interleaved), race-verifies and sanitizes it, profiles D = 384/512, and times the training step against
FlashRNN1 with fused and unfused backward candidates side by side.
v2: v1 (job 47365377) stopped after 6 min in the FlashRNN1 prebuild with EDQUOT: the group scratch quota was at its
hard limit. Same source, plan and criteria; the batch directory moved to home (quota gate in seal.py).
"""
import json
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
WIDTHS, STEPS = (192, 256, 384, 512, 768), 256
TL_FORWARD = {192: [("cluster", 16), ("cluster", 32)], 256: [("cluster", 16), ("cluster", 32)],
              384: [("cluster", 32)], 512: [("cluster", 32)], 768: [("grid", 16), ("grid", 32)]}
UNFUSED_SLICES = (32, 16)
# Fused kernels hold both R slices in shared memory: S32 fits up to D = 384 (212992 B), S16 everywhere (login sweep).
FUSED_SLICES = {192: (16, 32), 256: (16, 32), 384: (16, 32), 512: (16,), 768: (16,)}
FUSE_LEVELS = (1, 2)


def fused_slices(width, level):
    """FUSE=2 (fewer MMAs, fewer in-loop spills in the login sweep) gets every fitting slice; FUSE=1 only S16."""
    return FUSED_SLICES[width] if level == 2 else (16,)


def triton_forwards(width):
    return [None, dict(family="mma_rtiled", warps=4, slice_width=32, sync="grid")]


def roles_for(width, forwards):
    unfused = [dict(kind="frnn2", forward=f, slice_width=s, warps=w, fast=True, tf32=True)
               for f in forwards for s in UNFUSED_SLICES for w in (4, 8)]
    fused = [dict(kind="frnn2", forward=f, slice_width=s, warps=w, fast=True, tf32=True, fuse=level)
             for f in forwards for level in FUSE_LEVELS for s in fused_slices(width, level) for w in (4, 8)]
    return dict(v1=[dict(kind="v1")], unfused=unfused, fused=fused, frnn2=unfused + fused)


def main():
    cases, comps, prebuild, lane = [], [], [], 0
    for env, batch in (("tl", 16), ("mod", 16), ("mod", 64)):
        tl_env = env == "tl"
        for width in WIDTHS:
            for cell in ("lstm", "slstm"):
                forwards = ([dict(family="tilelang", slice_width=s, sync=sync) for sync, s in TL_FORWARD[width]]
                            if tl_env else triton_forwards(width))
                name = f"fuse-{env}-{cell}-float16-d{width}-b{batch}-t{STEPS}"
                case = dict(name=name, lane=lane, cell=cell, dtype="float16", width=width, batch=batch,
                            concurrency=batch, steps=STEPS, tilelang=tl_env, roles=roles_for(width, forwards))
                cases.append(case)
                common = dict(level="train_step", cell=cell, dtype="float16", width=width, batch=batch,
                              concurrency=batch, steps=STEPS, select_case=name, lane=lane, tilelang=tl_env)
                comps.append(dict(common, name=f"{name}-frnn2-vs-v1", experiment="TRAIN_STEP_FRNN2_VS_V1",
                                  baseline_role="v1", candidate_role="frnn2"))
                comps.append(dict(common, name=f"{name}-fused-vs-unfused", experiment="TRAIN_STEP_FUSED_VS_UNFUSED",
                                  baseline_role="unfused", candidate_role="fused"))
                if env == "mod":
                    prebuild.append(dict(case, name=f"prebuild-{cell}-d{width}-b{batch}",
                                         roles=dict(v1=[dict(kind="v1")])))
                lane = (lane + 1) % 4
    qualify, lane = [], 0
    inits = (("lstm", "warm"), ("slstm", "zero"), ("slstm", "warm"))
    for width in WIDTHS:
        for dtype in ("float16", "bfloat16"):
            for cell, init in inits:
                for level in FUSE_LEVELS:
                    for s in fused_slices(width, level):
                        for warps in (8, 4):
                            shapes = [(32, 64)]
                            if level == 2 and s == 16 and warps == 8:
                                shapes = [(3, 17), (32, 64)]  # the primary variant also on a ragged short shape
                            for batch, steps in shapes:
                                # FlashRNN1 calibration where the gradient path changes most: FUSE=2, S16, 8 warps.
                                v1 = level == 2 and s == 16 and warps == 8 and batch == 32
                                c = dict(cell=cell, dtype=dtype, width=width, slice_width=s, warps=warps, batch=batch,
                                         steps=steps, init=init, fast=True, tf32=True, fuse=level, no_v1=not v1)
                                qualify.append(dict(
                                    name=f"q-fuse{level}-{cell}-{init}-{dtype}-d{width}-s{s}-w{warps}-b{batch}-t{steps}",
                                    lane=lane, fuse=level, stages="grads,vs_base,modes,repeat", case=c))
                                lane = (lane + 1) % 4
    race = []
    for level, width in ((2, 256), (2, 512), (2, 768), (1, 384)):
        for cell in ("lstm", "slstm"):
            for group in ("correct", "broken_single", "broken_no_barrier", "broken_arrive_first"):
                race.append(dict(name=f"race-fuse{level}-{cell}-d{width}-{group}", lane=lane, fuse=level,
                                 variants=group, gating=group == "correct",
                                 task=dict(path="triton_bw", cell=cell,
                                           config=dict(width=width, slice_width=16, warps=8, batch=40, steps=48,
                                                       fast=True, fuse=level))))
                lane = (lane + 1) % 4
    sanitizer = []
    for tool in ("memcheck", "synccheck"):
        for level in FUSE_LEVELS:
            for cell in ("lstm", "slstm"):
                for width in (192, 256):
                    sanitizer.append(dict(name=f"sanitizer-{tool}-fuse{level}-{cell}-d{width}", tool=tool, lane=lane,
                                          fuse=level, case=dict(cell=cell, dtype="float16", width=width,
                                                                slice_width=16, warps=8, batch=3, steps=5,
                                                                init="warm", fast=True, tf32=True, fuse=level)))
                    lane = (lane + 1) % 4
    profile = []
    for width in (384, 512):
        for cell in ("lstm", "slstm"):
            for steps in (64, 256):
                profile.append(dict(name=f"profile-{cell}-d{width}-b16-t{steps}", lane=lane,
                                    case=dict(cell=cell, dtype="float16", width=width, batch=16, steps=steps)))
                lane = (lane + 1) % 4
    plan = dict(
        batch="flashrnn2-fuse-20261016-v2",
        purpose=("Fused preactivation recompute in the R-tiled backward: qualify FUSE=1/2 (gradient gate against "
                 "the FP64 oracle with FlashRNN1 calibration, persistent == launch bitwise, 20 repeats), race-verify "
                 "and sanitize them, profile D = 384/512 at B = 16, then select among unfused (fast + TF32) and fused "
                 "backward candidates per forward and time the training step against FlashRNN1 (TileLang env B = 16, "
                 "module env B = 16 and 64) and fused against unfused"),
        gates=("Pre-registered admission of a fused level f to selection: every qualification case of level f "
               "passes (grads, modes, repeat; vs_base is not applicable: a fused kernel is not bitwise to the GEMM "
               "path by construction), the race 'correct' groups of level f are clean (0 violations, outputs bitwise "
               "equal to the uninstrumented kernel; broken-variant controls are reported, not gating), and every "
               "sanitizer run of level f reports 0 errors. A level that fails is removed from every selection role "
               "before selection; the unfused candidates (qualified in jobs 47331012/47347074) always stay."),
        criterion=("Pre-registered: B = 16 counts as fixed if, in at least one environment, the selected FlashRNN2 "
                   "training step (role frnn2 = unfused + admitted fused candidates)'s 95% interval lower bound "
                   "against FlashRNN1 is >= 1.0 for every D in 192-768 and both cells; B = 64 must not regress "
                   "(every shape keeps lower bound >= 1.0). The fusion effect is the fused-vs-unfused comparison "
                   "(best admitted fused vs best unfused, same forward set) per shape. Default rule: fuse_pre becomes "
                   "the default for a width D only if fused-vs-unfused has lower bound >= 1.0 at B = 16 in the "
                   "TileLang environment for both cells; it applies to every batch size if the B = 64 fused-vs-unfused "
                   "upper bound is >= 1.0 for both cells, otherwise only to B <= 16."),
        minutes=165, starts=5, blocks=20, v1_build_root="/scratch/tmp/tueb043681-fuse2-{job}-v1builds",
        prebuild=prebuild, qualify=qualify, race=race, sanitizer=sanitizer, profile=profile,
        select_cases=cases, comparisons=comps,
        expected_minutes=dict(total=125, basis=("v3 (job 47347074) took 60 min for 28 prebuilds, 80 phase-0 tasks, "
                                                "39 selections and 40 comparisons; here 20 prebuilds, about 200 "
                                                "phase-0 tasks plus 8 profiles, 30 selections with about twice the "
                                                "candidates, 60 comparisons x 5 starts")),
        node_reason="One H100 node (4 H100): independent single-GPU lanes; no cross-GPU topology needed",
        not_claimed=["BF16 timing", "T other than 256 (profile: 64/256)", "heads > 1", "D < 192 fused"],
    )
    (CONTROL / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(len(prebuild), len(qualify), len(race), len(sanitizer), len(profile), len(cases), len(comps),
          sum(len(c["roles"]["frnn2"]) for c in cases))


if __name__ == "__main__":
    main()
