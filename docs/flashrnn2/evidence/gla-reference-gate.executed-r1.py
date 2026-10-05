"""Exercise complete GLA cached generation against full-prefix recomputation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from gla_torch_reference import GLAReference
from tokenizers import Tokenizer

PROMPTS = (
    "The history of recurrent neural networks began with attempts to model memory.",
    "A small research team compared the predictions of several language models.",
    "The train crossed the bridge before arriving at the central station.",
    "Scientists collected samples from the river and measured the water temperature.",
    "A computer program can process a sequence one element at a time.",
    "The library opened its doors early on a quiet Monday morning.",
    "An engineer checked the measurements before designing the new circuit.",
    "The next chapter explains how the experiment was conducted and evaluated.",
    "A mountain path led the hikers through a forest toward the lake.",
    "The teacher asked the students to explain their answers in complete sentences.",
    "The performance of a system depends on both computation and communication.",
    "A musician practiced the final movement before the evening concert began.",
)


def compare(actual: torch.Tensor, expected: torch.Tensor, budget: float) -> dict:
    difference = (actual - expected).abs()
    allowed = budget + budget * expected.abs()
    failed = difference > allowed
    return {
        "pass": bool(torch.isfinite(actual).all() and not failed.any()),
        "max_abs": difference.max().item(),
        "worst_normalized": (difference / allowed).max().item(),
        "failed_elements": failed.sum().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    started = time.time()
    metadata = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": started,
        "device": "cpu",
        "torch": torch.__version__,
        "scope": "FULL_GLA_TORCH_REFERENCE_CACHED_VS_FULL_PREFIX_P5_G4",
        "logits_budget": {"atol": 1e-3, "rtol": 1e-3},
        "state_budget": {"atol": 1e-5, "rtol": 1e-5},
        "performance_claim": False,
        "native_accelerated_fla_qualified": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("gla_torch_reference.py"),
            )
        },
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenized = [tokenizer.encode(text).ids for text in PROMPTS]
    assert all(len(ids) >= 5 for ids in tokenized)
    model = GLAReference(args.model, args.source)
    metadata["model"] = model.provenance
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    snapshots, records = [], []
    with torch.inference_mode(), args.output.open("x") as handle:
        for batch in (1, 2, 4):
            for case in range(3):
                ids = torch.tensor(
                    [tokenized[4 * case + row][:5] for row in range(batch)]
                )
                full_ids = ids.clone()
                logits, cache = model(ids)
                saved_logits, saved_tokens = [], []
                for step in range(4):
                    full_logits, full_cache = model(full_ids)
                    left, right = logits[:, -1], full_logits[:, -1]
                    checks = {"logits": compare(left, right, 1e-3)}
                    state_checks = [
                        compare(a["recurrent_state"], b["recurrent_state"], 1e-5)
                        for a, b in zip(cache.states, full_cache.states, strict=True)
                    ]
                    assert len(state_checks) == 24
                    checks["state"] = {
                        "pass": all(x["pass"] for x in state_checks),
                        "max_abs": max(x["max_abs"] for x in state_checks),
                        "worst_normalized": max(
                            x["worst_normalized"] for x in state_checks
                        ),
                        "failed_elements": sum(
                            x["failed_elements"] for x in state_checks
                        ),
                    }
                    next_ids = left.argmax(-1, keepdim=True)
                    match = torch.equal(next_ids, right.argmax(-1, keepdim=True))
                    row = {
                        "batch": batch,
                        "case": case,
                        "step": step,
                        "checks": checks,
                        "token_equal": match,
                        "pass": match and all(x["pass"] for x in checks.values()),
                    }
                    records.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(json.dumps(row), flush=True)
                    saved_logits.append(left.clone())
                    saved_tokens.append(next_ids[:, 0].clone())
                    if step < 3:
                        full_ids = torch.cat((full_ids, next_ids), dim=1)
                        logits, cache = model(next_ids, cache)
                snapshots.append(
                    {
                        "batch": batch,
                        "case": case,
                        "input_ids": ids,
                        "logits": torch.stack(saved_logits),
                        "generated_ids": torch.stack(saved_tokens),
                    }
                )
    tensor_path = args.output.with_suffix(".pt")
    torch.save(snapshots, tensor_path)
    with tensor_path.open("rb") as handle:
        metadata["tensor_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    passed = all(row["pass"] for row in records)
    metadata.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        cases=len(snapshots),
        batch_steps=len(records),
        token_choices=sum(row["batch"] for row in records),
        prompts=PROMPTS,
        snapshots="Full-vocabulary logits and token IDs; state error summaries only",
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
