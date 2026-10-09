"""Complete-checkpoint xLSTM/sLSTM CPU reference for the author's LM forward path."""

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

REVISION = "34d473516f5391ffb59df94313b4acb6fd0745b6"
WEIGHT_SHA256 = "1ed83701a3164eab20a2abb09278e8a2635d7ea8000253d4faec9cd6b6b397d1"
SOURCE_REVISION = "ab22eadbd245f293dd8dec38ed29963d73758a12"
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


def load_model(model_dir: Path, source_dir: Path):
    sys.path.insert(0, str(source_dir))
    from xlstm import (
        mLSTMBlockConfig,
        sLSTMBlockConfig,
        xLSTMLMModel,
        xLSTMLMModelConfig,
    )
    from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
    from xlstm.blocks.slstm.layer import sLSTMLayerConfig

    config = json.loads((model_dir / "config.json").read_text())["_xlstm_config"]
    assert config["num_blocks"] == 2 and config["slstm_at"] == [1]
    assert config["embedding_dim"] == 64 and config["vocab_size"] == 32000
    assert config["mlstm_block"]["mlstm"]["num_heads"] == 1
    assert config["slstm_block"]["slstm"]["num_heads"] == 1
    model_config = xLSTMLMModelConfig(
        num_blocks=2,
        embedding_dim=64,
        mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(num_heads=1)),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(num_heads=1, backend="vanilla", dtype="float32")
        ),
        slstm_at=[1],
        context_length=256,
        vocab_size=32000,
        tie_weights=True,
    )
    model = xLSTMLMModel(model_config).eval()
    weights = {
        key.removeprefix("model."): value
        for key, value in load_file(
            str(model_dir / "model.safetensors"), device="cpu"
        ).items()
    }
    assert len(weights) == 29
    cell = model.xlstm_block_stack.blocks[1].xlstm.slstm_cell
    recurrent_key = "xlstm_block_stack.blocks.1.xlstm.slstm_cell._recurrent_kernel_"
    original = weights[recurrent_key]
    weights[recurrent_key] = cell._recurrent_kernel_ext2int(original)
    assert torch.equal(
        cell._recurrent_kernel_int2ext(weights[recurrent_key]).reshape_as(original),
        original,
    )
    weights["lm_head.weight"] = weights["token_embedding.weight"]
    model.load_state_dict(weights, strict=True)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    return model


def full_forward(model, ids: torch.Tensor) -> torch.Tensor:
    # The pinned Hugging Face author's xLSTMBlockStack.forward omits post_blocks_norm.
    hidden = model.emb_dropout(model.token_embedding(ids))
    for block in model.xlstm_block_stack.blocks:
        hidden = block(hidden)
    return model.lm_head(hidden)[:, -1, :]


def recurrent_step(model, ids: torch.Tensor, state: dict | None):
    hidden = model.emb_dropout(model.token_embedding(ids))
    state = {} if state is None else state
    next_state = {}
    for index, block in enumerate(model.xlstm_block_stack.blocks):
        key = f"block_{index}"
        hidden, next_state[key] = block.step(hidden, **state.get(key, {}))
    return model.lm_head(hidden)[:, -1, :], next_state


def state_pairs(batch_state: dict, serial_state: dict, row: int):
    batch_mlstm = batch_state["block_0"]
    serial_mlstm = serial_state["block_0"]
    for index, value in enumerate(batch_mlstm["mlstm_state"]):
        yield f"mlstm_{index}", value[row : row + 1], serial_mlstm["mlstm_state"][index]
    yield (
        "mlstm_conv",
        batch_mlstm["conv_state"][0][row : row + 1],
        serial_mlstm["conv_state"][0],
    )
    batch_slstm = batch_state["block_1"]
    serial_slstm = serial_state["block_1"]
    yield (
        "slstm_conv",
        batch_slstm["conv_state"][0][row : row + 1],
        serial_slstm["conv_state"][0],
    )
    yield (
        "slstm_hcnm",
        batch_slstm["slstm_state"][:, row : row + 1],
        serial_slstm["slstm_state"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(4)
    weight_path = args.model / "model.safetensors"
    with weight_path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == WEIGHT_SHA256
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    inputs = [
        [1, *tokenizer.encode(prompt, add_special_tokens=False).ids[:4]]
        for prompt in PROMPTS
    ]
    assert len(inputs) == 12 and all(len(ids) == 5 for ids in inputs)
    started = time.time()
    meta_path = args.output.with_suffix(".meta.json")
    meta = {
        "status": "RUNNING",
        "scope": "LOVERCRAFT_XLSTM_MLSTM_SLSTM_FULL_CPU_AUTHOR_FORWARD_P5_G4",
        "pid": os.getpid(),
        "started": started,
        "torch": torch.__version__,
        "model_revision": REVISION,
        "source_revision": SOURCE_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "input_ids": inputs,
        "budget": {"atol": 1e-4, "rtol": 1e-4},
        "performance_claim": False,
        "forward_semantics": "author HF wrapper omits post_blocks_norm",
        "cpu_cell_backend": "official xlstm vanilla float32; CUDA layout converted losslessly",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    model = load_model(args.model, args.source)
    meta.update(
        parameters=sum(p.numel() for p in model.parameters()), checkpoint_tensors=29
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    rows, snapshots = [], []
    with torch.inference_mode(), args.output.open("x") as handle:
        for batch in (1, 2, 4):
            for case in range(3):
                ids = torch.tensor(inputs[4 * case : 4 * case + batch])
                batch_state = None
                serial_states = [None] * batch
                for position in range(ids.shape[1]):
                    step_logits, batch_state = recurrent_step(
                        model, ids[:, position : position + 1], batch_state
                    )
                    for index in range(batch):
                        _, serial_states[index] = recurrent_step(
                            model,
                            ids[index : index + 1, position : position + 1],
                            serial_states[index],
                        )
                for generation in range(4):
                    full = full_forward(model, ids)
                    serial = torch.cat(
                        [
                            full_forward(model, ids[index : index + 1])
                            for index in range(batch)
                        ]
                    )
                    state_checks = []
                    for index in range(batch):
                        for name, actual, expected in state_pairs(
                            batch_state, serial_states[index], index
                        ):
                            state_checks.append(
                                (name, actual.clone(), expected.clone())
                            )
                    full_check = compare(full, serial)
                    step_check = compare(step_logits, full)
                    state_check = all(
                        compare(actual, expected)["pass"]
                        for _, actual, expected in state_checks
                    )
                    tokens = full.argmax(-1)
                    token_equal = torch.equal(
                        tokens, serial.argmax(-1)
                    ) and torch.equal(tokens, step_logits.argmax(-1))
                    row = {
                        "batch": batch,
                        "case": case,
                        "generation": generation,
                        "full_vs_serial": full_check,
                        "step_vs_full": step_check,
                        "state_pairs": len(state_checks),
                        "state_pass": state_check,
                        "tokens_equal": token_equal,
                        "pass": full_check["pass"]
                        and step_check["pass"]
                        and state_check
                        and token_equal,
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(json.dumps(row), flush=True)
                    snapshots.append(
                        {
                            "row": row,
                            "input_ids": ids.clone(),
                            "full_logits": full.clone(),
                            "serial_logits": serial.clone(),
                            "step_logits": step_logits.clone(),
                            "tokens": tokens.clone(),
                            "state_pairs": state_checks,
                        }
                    )
                    ids = torch.cat((ids, tokens[:, None]), dim=1)
                    step_logits, batch_state = recurrent_step(
                        model, tokens[:, None], batch_state
                    )
                    for index in range(batch):
                        _, serial_states[index] = recurrent_step(
                            model, tokens[index : index + 1, None], serial_states[index]
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
        state_tensor_pairs=sum(row["state_pairs"] for row in rows),
        tensor_sha256=tensor_sha256,
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
