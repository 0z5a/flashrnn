"""Full-checkpoint pure GDN-2 CPU generation and cache comparison."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from gdn2_pure_torch_reference import load_reference
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


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual - expected).abs()
    allowed = 1e-4 + 1e-4 * expected.abs()
    failed = (
        ~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)
    )
    return {
        "pass": not failed.any().item(),
        "max_abs": difference.max().item(),
        "failed_elements": int(failed.sum()),
    }


def prefill(model, ids: torch.Tensor):
    state = None
    for position in range(ids.shape[1]):
        logits, state = model(ids[:, position : position + 1], state)
    return logits[:, -1], state


def state_pairs(batch_state: list, serial_state: list, row: int):
    for layer_index, (batch_layer, serial_layer) in enumerate(
        zip(batch_state, serial_state, strict=True)
    ):
        yield (
            f"layer_{layer_index}_recurrent",
            batch_layer[0][row : row + 1],
            serial_layer[0],
        )
        for conv_index in range(3):
            yield (
                f"layer_{layer_index}_conv_{conv_index}",
                batch_layer[1][conv_index][row : row + 1],
                serial_layer[1][conv_index],
            )


def full_state_pairs(batch_state: list, full_state: list):
    for layer_index, (cached_layer, full_layer) in enumerate(
        zip(batch_state, full_state, strict=True)
    ):
        yield f"layer_{layer_index}_recurrent", cached_layer[0], full_layer[0]
        for conv_index in range(3):
            yield (
                f"layer_{layer_index}_conv_{conv_index}",
                cached_layer[1][conv_index],
                full_layer[1][conv_index],
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--fla-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    inputs = [
        [1, *tokenizer.encode(prompt, add_special_tokens=False).ids[:4]]
        for prompt in PROMPTS
    ]
    assert len(inputs) == 12 and all(len(ids) == 5 for ids in inputs)
    started = time.time()
    meta_path = args.output.with_suffix(".meta.json")
    meta = {
        "status": "RUNNING",
        "scope": "PURE_GDN2_305M_12_LAYER_FULL_CPU_BATCH_SERIAL_CACHE_P5_G4",
        "pid": os.getpid(),
        "started": started,
        "torch": torch.__version__,
        "model_revision": "2d98aff017d7a9e125a14dd02e8c4b6f631930a8",
        "source_revision": "7efb255b69411084e7e3c0e13d5083910d3258a7",
        "tokenizer_revision": "ff3c701f2424c7625fdefb9dd470f45ef18b02d6",
        "input_ids": inputs,
        "budget": {"atol": 1e-4, "rtol": 1e-4},
        "performance_claim": False,
        "model_card_printed_parameters": 239272896,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    model = load_reference(args.checkpoint, args.fla_source)
    meta.update(checkpoint_tensors=267, actual_parameters=304809920)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    rows, snapshots = [], []
    with torch.inference_mode(), args.output.open("x") as handle:
        for batch in (1, 2, 4):
            for case in range(3):
                ids = torch.tensor(inputs[4 * case : 4 * case + batch])
                step_logits, batch_state = prefill(model, ids)
                serial_results = [
                    prefill(model, ids[index : index + 1]) for index in range(batch)
                ]
                serial_states = [result[1] for result in serial_results]
                for generation in range(4):
                    full_output, full_state = model(ids)
                    full = full_output[:, -1]
                    serial = torch.cat(
                        [
                            model(ids[index : index + 1])[0][:, -1]
                            for index in range(batch)
                        ]
                    )
                    full_check = compare(full, serial)
                    step_check = compare(step_logits, full)
                    tokens = full.argmax(-1)
                    token_equal = torch.equal(
                        tokens, serial.argmax(-1)
                    ) and torch.equal(tokens, step_logits.argmax(-1))
                    checks = []
                    full_checks = []
                    if generation == 3:
                        for index in range(batch):
                            for name, actual, expected in state_pairs(
                                batch_state, serial_states[index], index
                            ):
                                checks.append((name, actual.clone(), expected.clone()))
                        for name, actual, expected in full_state_pairs(
                            batch_state, full_state
                        ):
                            full_checks.append((name, actual.clone(), expected.clone()))
                    state_pass = all(
                        compare(actual, expected)["pass"]
                        for _, actual, expected in checks
                    )
                    full_state_pass = all(
                        compare(actual, expected)["pass"]
                        for _, actual, expected in full_checks
                    )
                    row = {
                        "batch": batch,
                        "case": case,
                        "generation": generation,
                        "full_vs_serial": full_check,
                        "cached_vs_full": step_check,
                        "tokens_equal": token_equal,
                        "state_pairs": len(checks),
                        "state_pass": state_pass,
                        "full_state_pairs": len(full_checks),
                        "full_state_pass": full_state_pass,
                        "pass": full_check["pass"]
                        and step_check["pass"]
                        and token_equal
                        and state_pass
                        and full_state_pass,
                    }
                    rows.append(row)
                    snapshots.append(
                        {
                            "row": row,
                            "input_ids": ids.clone(),
                            "full_logits": full.clone(),
                            "serial_logits": serial.clone(),
                            "cached_logits": step_logits.clone(),
                            "tokens": tokens.clone(),
                            "state_pairs": checks,
                            "full_state_pairs": full_checks,
                        }
                    )
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(json.dumps(row), flush=True)
                    if generation < 3:
                        ids = torch.cat((ids, tokens[:, None]), dim=1)
                        step_logits, batch_state = model(tokens[:, None], batch_state)
                        step_logits = step_logits[:, -1]
                        for index in range(batch):
                            _, serial_states[index] = model(
                                tokens[index : index + 1, None], serial_states[index]
                            )
    tensor_path = args.output.with_suffix(".pt")
    torch.save(snapshots, tensor_path)
    with tensor_path.open("rb") as handle:
        tensor_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    passed = all(row["pass"] for row in rows)
    meta.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        cases=9,
        batch_steps=len(rows),
        token_choices=sum(row["batch"] for row in rows),
        final_state_pairs=sum(row["state_pairs"] for row in rows),
        full_state_pairs=sum(row["full_state_pairs"] for row in rows),
        tensor_sha256=tensor_sha256,
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
