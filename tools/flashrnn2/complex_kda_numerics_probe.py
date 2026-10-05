"""Compare the author's full-prefix and cached CPU paths for one pinned prompt."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

PROMPT = "The history of recurrent networks begins with memory."


def compare(actual: torch.Tensor, expected: torch.Tensor, budget: float) -> dict:
    difference = (actual.float() - expected.float()).abs()
    allowed = budget * (1 + expected.float().abs())
    failed = (
        ~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)
    )
    return {
        "pass": not failed.any().item(),
        "max_abs": difference.max().item(),
        "failed_elements": int(failed.sum()),
        "worst_normalized": (difference / allowed).max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dtype", required=True, choices=("bf16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert args.batch_size in (1, 2, 4)
    assert args.prompt_tokens in (5, 16)
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    with args.checkpoint.open("rb") as handle:
        assert (
            hashlib.file_digest(handle, "sha256").hexdigest() == args.checkpoint_sha256
        )
    os.environ["COMPLEX_KDA_BACKEND"] = "torch"
    sys.path.insert(0, str(args.model_source.resolve()))
    from configuration_complex_kda import ComplexKDAConfig
    from modeling_complex_kda import ComplexKDAForCausalLM

    torch.set_num_threads(4)
    started = time.monotonic()
    config = ComplexKDAConfig.from_pretrained(args.model_source)
    with torch.device("meta"):
        model = ComplexKDAForCausalLM(config)
    weights = load_file(str(args.checkpoint), device="cpu")
    assert len(weights) == 507 and set(weights) == set(model.state_dict())
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    if args.dtype == "fp32":
        model.float()
    model.eval()
    tokenizer = Tokenizer.from_file(str(args.model_source / "tokenizer.json"))
    prompt_ids = tokenizer.encode(PROMPT * 4, add_special_tokens=False).ids[
        : args.prompt_tokens + args.batch_size - 1
    ]
    ids = torch.tensor(
        [
            prompt_ids[index : index + args.prompt_tokens]
            for index in range(args.batch_size)
        ],
        dtype=torch.long,
    )
    assert ids.shape == (args.batch_size, args.prompt_tokens)
    loaded_seconds = time.monotonic() - started
    budget = 0.02 if args.dtype == "bf16" else 1e-4
    with torch.inference_mode():
        full = model(ids, use_cache=True)
        cache = None
        for position in range(ids.shape[1]):
            step = model(
                ids[:, position : position + 1], past_key_values=cache, use_cache=True
            )
            cache = step.past_key_values
    full_logits = full.logits[:, -1].clone()
    cached_logits = step.logits[:, -1].clone()
    logits = compare(cached_logits, full_logits, budget)
    pairs = []
    for layer in range(24):
        full_state = full.past_key_values.states[layer]
        cached_state = cache.states[layer]
        pairs.append(
            (
                f"layer_{layer}_recurrent",
                cached_state["recurrent_state"].clone(),
                full_state["recurrent_state"].clone(),
            )
        )
        for index in range(3):
            pairs.append(
                (
                    f"layer_{layer}_conv_{index}",
                    cached_state["conv_state"][index].clone(),
                    full_state["conv_state"][index].clone(),
                )
            )
    checks = [
        (name, compare(actual, expected, budget)) for name, actual, expected in pairs
    ]
    tokens_equal = torch.equal(full_logits.argmax(-1), cached_logits.argmax(-1))
    passed = (
        logits["pass"] and all(check["pass"] for _, check in checks) and tokens_equal
    )
    snapshot = {
        "input_ids": ids,
        "full_logits": full_logits,
        "cached_logits": cached_logits,
        "state_pairs": pairs,
    }
    snapshot_path = args.output.with_suffix(".pt")
    torch.save(snapshot, snapshot_path)
    with snapshot_path.open("rb") as handle:
        snapshot_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {
        "status": "PASS" if passed else "NUMERICAL_FAILED",
        "dtype": args.dtype,
        "checkpoint_sha256": args.checkpoint_sha256,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "budget": {"atol": budget, "rtol": budget},
        "input_ids": ids.tolist(),
        "checkpoint_tensors": 507,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "layers": 24,
        "logits": logits,
        "tokens_equal": tokens_equal,
        "state_pairs": len(checks),
        "state_pass": sum(check["pass"] for _, check in checks),
        "state_max_abs": max(check["max_abs"] for _, check in checks),
        "state_failed_elements": sum(check["failed_elements"] for _, check in checks),
        "first_failed_states": [
            (name, check) for name, check in checks if not check["pass"]
        ][:5],
        "snapshot_sha256": snapshot_sha256,
        "loaded_seconds": round(loaded_seconds, 3),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "native_gpu_executed": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
