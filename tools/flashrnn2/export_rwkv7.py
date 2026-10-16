"""Qualify batched full RWKV7 against native oracles, then save a Torch-only model."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from mamba_script_gate import errors, verified_metadata
from rwkv7_batched import RWKV7Batched
from rwkv7_native import NativeRWKV7

CONTRACT = {
    "logits_contract": {"atol": 0.001, "rtol": 0.001},
    "cache_contract": {"atol": 0.00001, "rtol": 0.00001},
}


def packed_cache(cases: list[dict], name: str) -> tuple[torch.Tensor, torch.Tensor]:
    states = [case[name] for case in cases]
    attention = torch.stack([torch.stack(s[0::3]) for s in states], dim=1)
    channel = torch.stack([torch.stack(s[2::3]) for s in states], dim=1)
    matrices = torch.stack([torch.stack(s[1::3]) for s in states], dim=1)
    return torch.stack((attention, channel)), matrices


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--converter-source", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.results.exists()
    torch.set_num_threads(1)
    started = time.time()
    oracle = verified_metadata(args.goldens)
    assert oracle["status"] == "NATIVE_CPU_ORACLES_COMPLETE_ACCELERATOR_PARITY_UNTESTED"
    cases = torch.load(args.goldens, map_location="cpu", weights_only=True)
    native = NativeRWKV7(args.model, args.native_source, args.converter_source)
    model = RWKV7Batched(native).eval()
    buffer_elements = sum(value.numel() for value in model.buffers())
    assert buffer_elements == native.checkpoint_elements - 2 * native.n_embd
    # Official inference preparation folds the two layer0 pre-norm vectors into embedding.
    scripted = torch.jit.script(model)
    meta = {
        "status": "RUNNING",
        "model_family": "rwkv7",
        "started": started,
        "torch": torch.__version__,
        "oracles": oracle,
        "contracts": CONTRACT,
        "tolerance_source": "Same pre-existing logits/state budgets as the Mamba script qualification; fixed before RWKV execution",
        "checkpoint_elements": native.checkpoint_elements,
        "inference_buffer_elements": buffer_elements,
        "model_layers": native.n_layer,
        "cache_layout": {
            "shifts": "[attention_or_channel,L,B,D]",
            "matrices": "[L,B,H,V,K]",
        },
        "cache_policy": "Functional outputs; incoming shift/matrix tensors cloned",
        "dtype": "float32",
        "batches": [1, 2, 4],
        "prompt_length": oracle["prompt_length"],
        "generated_tokens": oracle["generated_tokens"],
        "selection": "Three groups per B, circularly selecting the three independent native prompts; B4 repeats one prompt in a distinct row",
        "performance_claim": False,
    }
    meta_path = args.results.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    failed, compared = 0, 0
    with torch.inference_mode(), args.results.open("x") as handle:
        for batch in meta["batches"]:
            for group in range(len(cases)):
                selected = [cases[(group + i) % len(cases)] for i in range(batch)]
                reference_shifts, reference_states = packed_cache(
                    selected, "prefill_cache"
                )
                shifts, states = (
                    torch.zeros_like(reference_shifts),
                    torch.zeros_like(reference_states),
                )
                ids = torch.stack([case["input_ids"] for case in selected])
                for step in range(oracle["generated_tokens"]):
                    method = scripted.prefill if step == 0 else scripted.decode
                    logits, shifts, states = method(ids, shifts, states)
                    actual = (logits[:, -1],)
                    expected = (
                        torch.stack([case["logits"][step] for case in selected]),
                    )
                    if step in (0, oracle["generated_tokens"] - 1):
                        key = "prefill_cache" if step == 0 else "final_cache"
                        actual += (shifts, states)
                        expected += packed_cache(selected, key)
                    comparisons = errors(actual, expected, CONTRACT)
                    for entry, name in zip(
                        comparisons, ("logits", "shift_cache", "matrix_cache")
                    ):
                        entry["tensor"] = name
                    ids = logits[:, -1].argmax(-1, keepdim=True)
                    token_match = torch.equal(
                        ids[:, 0],
                        torch.stack([case["generated_ids"][step] for case in selected]),
                    )
                    passed = token_match and all(e["pass"] for e in comparisons)
                    failed += int(not passed)
                    compared += 1
                    row = {
                        "batch": batch,
                        "group": group,
                        "step": step,
                        "errors": comparisons,
                        "token_match": token_match,
                        "status": "PASS" if passed else "FAIL",
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                print(
                    json.dumps(
                        {"batch": batch, "group": group, "failed_steps_so_far": failed}
                    ),
                    flush=True,
                )
    scripted.save(str(args.output))
    meta.update(
        status="CPU_PARITY_FAIL_GPU_UNTESTED"
        if failed
        else "CPU_PARITY_PASS_GPU_UNTESTED",
        failed_steps=failed,
        compared_steps=compared,
        finished=time.time(),
        artifact_sha256=digest(args.output),
        artifact_bytes=args.output.stat().st_size,
        source_sha256={
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path(NativeRWKV7.step.__code__.co_filename),
                Path(RWKV7Batched.forward.__code__.co_filename),
                Path(errors.__code__.co_filename),
            )
        },
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
