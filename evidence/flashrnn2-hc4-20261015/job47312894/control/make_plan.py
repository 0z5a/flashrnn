"""Write plan.json for HC round 4: structural ablation ceilings and the whole-node NUMA experiment.

Run once while preparing; the output is sealed with the batch.
"""
import json
from pathlib import Path

CONTROL = Path(__file__).resolve().parent


def tl(sync, s, bm, warps):
    return dict(family="tilelang", batch_tile=bm, warps=warps, weight_operand="b", slice_width=s, sync=sync,
                pad_k=False, prefetch=False)


def tri(s, w):
    return dict(family="mma_rtiled", batch_tile=16, warps=w, weight_operand="b", slice_width=s, sync="grid",
                pad_k=False, prefetch=False)


VARIANTS = "exchange,gemm,stage,wx,epilogue,out,exchange+stage+wx+out+epilogue"


def main():
    ablation = [dict(name=f"{cell}-d256-tl-cluster-s16-bm16", cell=cell, width=256, config=tl("cluster", 16, 16, 4))
                for cell in ("lstm", "slstm")]
    ablation += [dict(name=f"{cell}-d768-tl-grid-s16-bm64-w8", cell=cell, width=768, config=tl("grid", 16, 64, 8))
                 for cell in ("lstm", "slstm")]
    numa = [dict(name="lstm-d768-c1024", cell="lstm", width=768, concurrency=1024, config=tl("grid", 16, 64, 8),
                 bursts=200, note="round-3 best at C = 1024"),
            dict(name="lstm-d256-c1024", cell="lstm", width=256, concurrency=1024, config=tri(64, 8),
                 bursts=1000, note="round-3 best at C = 1024 (Triton default)")]
    plan = dict(
        batch="flashrnn2-hc4-20261015-v1",
        purpose=("High-concurrency round 4. (A) Timing-only ablation ceilings of the best TileLang kernels: each "
                 "variant removes one component (h exchange and its synchronization, the WGMMA, the FP32 gate stage, "
                 "the input-projection loads, the gate nonlinearities, the output stores; and all but the WGMMA) to "
                 "bound what a structural rewrite could gain. Ablated results are wrong by design and never served. "
                 "(B) Whole-node NUMA plane: four serving processes, one per GPU, C = 1024 per GPU, started together, "
                 "with CPU and memory bound to the GPU's NUMA node, to the farthest node, or unbound; 5 rounds with the "
                 "conditions rotated"),
        ablation=ablation, variants=VARIANTS, ablation_batches="1024,256", ablation_reps=15,
        numa=numa, numa_rounds=5, numa_conditions=["local", "remote", "unbound"], numa_cores_per_worker=4,
        steps=128, minutes=150, tk_root="", tk_cache="/scratch/tmp/tueb043681-hc4-{job}-tk",
        node_reason="One ACC node (4 H100): (A) one ablation task per GPU lane; (B) all four GPUs together",
        not_claimed=["ablated timings are ceilings for single components, not achievable speedups",
                     "Triton kernels are not ablated", "NUMA results for other hosts or arrival processes"],
    )
    (CONTROL / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(len(ablation), len(numa))


if __name__ == "__main__":
    main()
