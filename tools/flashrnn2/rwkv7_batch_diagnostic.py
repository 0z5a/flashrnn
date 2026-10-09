"""Locate RWKV7 batch-dependent errors; FP64 projection probes are not new budgets."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from export_rwkv7 import CONTRACT, packed_cache
from mamba_script_gate import verified_metadata
from torch.nn import functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    model_meta, oracle = verified_metadata(args.model), verified_metadata(args.goldens)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    cases = torch.load(args.goldens, weights_only=True, map_location="cpu")[:2]
    ids = torch.stack([c["input_ids"] for c in cases])
    cache_rows, projections = [], []
    with torch.inference_mode():
        layer = next(model.layers.children())
        x = F.layer_norm(
            model.embedding[ids[:, 0]],
            (model.hidden,),
            layer.attn_norm_w,
            layer.attn_norm_b,
        )
        for name, weight, mix in (
            ("receptance", layer.rw, layer.x_r),
            ("key", layer.kw, layer.x_k),
            ("value", layer.vw, layer.x_v),
        ):
            mixed = x + (-x) * mix
            single = F.linear(mixed[:1], weight)[0]
            batched = F.linear(mixed, weight)[0]
            fp64 = F.linear(mixed[:1].double(), weight.double())[0]
            projections.append(
                {
                    "projection": name,
                    "batch1_vs_batch2_max_abs": float((single - batched).abs().max()),
                    "different_elements": int((single != batched).sum()),
                    "batch1_vs_fp64_max_abs": float(
                        (single.double() - fp64).abs().max()
                    ),
                    "batch2_vs_fp64_max_abs": float(
                        (batched.double() - fp64).abs().max()
                    ),
                    "batch1_vs_fp64_relative_l2": float(
                        (single.double() - fp64).norm() / fp64.norm()
                    ),
                    "batch2_vs_fp64_relative_l2": float(
                        (batched.double() - fp64).norm() / fp64.norm()
                    ),
                }
            )
        shifts, states = packed_cache(cases, "prefill_cache")
        shifts, states = torch.zeros_like(shifts), torch.zeros_like(states)
        for step in range(oracle["generated_tokens"]):
            logits, shifts, states = model(ids, shifts, states)
            ids = logits[:, -1].argmax(-1, keepdim=True)
            if step not in (0, oracle["generated_tokens"] - 1):
                continue
            key = "prefill_cache" if step == 0 else "final_cache"
            for name, actual, expected in zip(
                ("shifts", "matrices"), (shifts, states), packed_cache(cases, key)
            ):
                budget = CONTRACT["cache_contract"]
                failed = ~torch.isclose(actual, expected, **budget)
                ratio = (actual - expected).abs() / (
                    budget["atol"] + budget["rtol"] * expected.abs()
                )
                coordinate = tuple(
                    int(i) for i in torch.unravel_index(ratio.argmax(), ratio.shape)
                )
                cache_rows.append(
                    {
                        "step": step,
                        "state": name,
                        "failed_elements": int(failed.sum()),
                        "max_abs": float((actual - expected).abs().max()),
                        "worst_budget_ratio": float(ratio[coordinate]),
                        "worst_coordinate": list(coordinate),
                        "actual_at_worst": float(actual[coordinate]),
                        "reference_at_worst": float(expected[coordinate]),
                        "failed_by_layer": (
                            failed.flatten(2).sum(-1)
                            if name == "shifts"
                            else failed.flatten(1).sum(-1)
                        ).tolist(),
                    }
                )
    result = {
        "scope": "FIRST_TWO_NATIVE_PROMPTS_B2_CACHE_ERRORS_AND_FIRST_LAYER_SAME_INPUT_LINEAR_PROBES",
        "torch": torch.__version__,
        "model_sha256": model_meta["artifact_sha256"],
        "oracle_sha256": oracle["artifact_sha256"],
        "contracts": CONTRACT,
        "projection_reference_scope": "FP64 for three isolated first-layer projections only; not full-model FP64 truth",
        "projections": projections,
        "cache_errors": cache_rows,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(packed_cache.__code__.co_filename),
                Path(verified_metadata.__code__.co_filename),
            )
        },
        "performance_claim": False,
        "revised_acceptance": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"projections": projections, "cache_errors": cache_rows}), flush=True
    )


if __name__ == "__main__":
    main()
