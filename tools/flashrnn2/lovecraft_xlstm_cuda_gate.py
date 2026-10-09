"""Qualify the pinned mixed xLSTM checkpoint against the author's CUDA cell."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from lovecraft_xlstm_reference_gate import (
    PROMPTS,
    REVISION,
    SOURCE_REVISION,
    WEIGHT_SHA256,
    compare,
    full_forward,
    load_model,
    recurrent_step,
)
from safetensors.torch import load_file
from tokenizers import Tokenizer

SOURCE_ARCHIVE_SHA256 = (
    "dc919005b21352a0b65f075c2dc81012303ee23b78d9bc091c2b76e48a871025"
)
CONFIG_SHA256 = "479ccd87fa94595e3da45ac307b5ac917279813807aba2a89fb9d182b197de58"
TOKENIZER_SHA256 = "bf467c9e0f536bda271283c6ef85eb1a943e3196b621c8a912d64953b205df83"
CPU_META_SHA256 = "a041175afe8fbb61f9cd999136f44c3b0d567a751502b7a6623a5ebee0f6ec5e"


def load_cuda_model(model_dir: Path, source_dir: Path):
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
            slstm=sLSTMLayerConfig(num_heads=1, backend="cuda", dtype="float32")
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
    prefix = "xlstm_block_stack.blocks.1.xlstm.slstm_cell."
    assert weights[prefix + "_recurrent_kernel_"].shape == cell._recurrent_kernel_.shape
    assert weights[prefix + "_bias_"].shape == cell._bias_.shape
    weights["lm_head.weight"] = weights["token_embedding.weight"]
    model.load_state_dict(weights, strict=True)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    return model


def state_tensors(state: dict) -> dict[str, torch.Tensor]:
    mlstm = state["block_0"]
    slstm = state["block_1"]
    assert len(mlstm["mlstm_state"]) == 3
    return {
        "mlstm_0": mlstm["mlstm_state"][0],
        "mlstm_1": mlstm["mlstm_state"][1],
        "mlstm_2": mlstm["mlstm_state"][2],
        "mlstm_conv": mlstm["conv_state"][0],
        "slstm_conv": slstm["conv_state"][0],
        "slstm_hcnm": slstm["slstm_state"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--cpu-reference-meta", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--batch-set", choices=("small", "large"), default="small")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batches = (1, 2, 4) if args.batch_set == "small" else (16, 32, 64)
    raw = args.output.with_suffix(".pt")
    meta_path = args.output.with_suffix(".meta.json")
    if any(path.exists() for path in (args.output, raw, meta_path)):
        parser.error("output already exists")
    assert (
        hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest()
        == WEIGHT_SHA256
    )
    assert (
        hashlib.sha256(args.source_archive.read_bytes()).hexdigest()
        == SOURCE_ARCHIVE_SHA256
    )
    assert (
        hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
        == CONFIG_SHA256
    )
    assert (
        hashlib.sha256((args.model / "tokenizer.json").read_bytes()).hexdigest()
        == TOKENIZER_SHA256
    )
    assert (
        hashlib.sha256(args.cpu_reference_meta.read_bytes()).hexdigest()
        == CPU_META_SHA256
    )
    cpu_meta = json.loads(args.cpu_reference_meta.read_text())
    assert cpu_meta["status"] == "PASS" and cpu_meta["weight_sha256"] == WEIGHT_SHA256
    if torch.cuda.device_count() != 1 or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("expected exactly one assigned GPU")
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    device_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if device_uuid != args.expected_gpu_uuid:
        raise ValueError("GPU identity differs from admission")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = {
        "evidence_version": 1,
        "status": "RUNNING",
        "started": time.time(),
        "scope": "PINNED_COMPLETE_MIXED_XLSTM_AUTHOR_CUDA_VS_VANILLA_FP32_NO_TIMING",
        "model_revision": REVISION,
        "source_revision": SOURCE_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "cpu_reference_meta_sha256": CPU_META_SHA256,
        "device_uuid": device_uuid,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "budget": {"atol": 1e-4, "rtol": 1e-4},
        "batch_set": args.batch_set,
        "batches": list(batches),
        "performance_claim": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    vanilla = load_model(args.model, args.source).to("cuda").eval()
    native = load_cuda_model(args.model, args.source).to("cuda").eval()
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    inputs = [
        [1, *tokenizer.encode(prompt, add_special_tokens=False).ids[:4]]
        for prompt in PROMPTS
    ]
    assert len(inputs) == 12 and all(len(ids) == 5 for ids in inputs)
    assert inputs == cpu_meta["input_ids"]
    pairs = []
    tokens = []

    def record(
        batch: int, case: int, phase: str, step: int, name: str, expected, actual
    ):
        reference = expected.detach().cpu()
        candidate = actual.detach().cpu()
        failed = (
            compare(candidate, reference)["failed_elements"]
            if candidate.shape == reference.shape
            else max(candidate.numel(), reference.numel())
        )
        pairs.append(
            {
                "batch": batch,
                "case": case,
                "phase": phase,
                "step": step,
                "name": name,
                "reference": reference,
                "native": candidate,
                "producer_failed": failed,
            }
        )

    def record_step(batch, case, phase, step, reference, candidate):
        reference_logits, reference_state = reference
        native_logits, native_state = candidate
        record(batch, case, phase, step, "step_logits", reference_logits, native_logits)
        base_states = state_tensors(reference_state)
        cuda_states = state_tensors(native_state)
        for name in base_states:
            record(
                batch,
                case,
                phase,
                step,
                name,
                base_states[name],
                cuda_states[name],
            )

    with torch.inference_mode():
        for batch in batches:
            for case in range(3):
                ids = torch.tensor(
                    [
                        inputs[(4 * case + index) % len(inputs)]
                        for index in range(batch)
                    ],
                    device="cuda",
                )
                reference_state = native_state = None
                for position in range(5):
                    reference_step = recurrent_step(
                        vanilla, ids[:, position : position + 1], reference_state
                    )
                    native_step = recurrent_step(
                        native, ids[:, position : position + 1], native_state
                    )
                    reference_state, native_state = reference_step[1], native_step[1]
                    record_step(
                        batch, case, "prefill", position, reference_step, native_step
                    )
                for generation in range(4):
                    reference_full = full_forward(vanilla, ids)
                    native_full = full_forward(native, ids)
                    record(
                        batch,
                        case,
                        "forecast",
                        generation,
                        "full_logits",
                        reference_full,
                        native_full,
                    )
                    record(
                        batch,
                        case,
                        "forecast",
                        generation,
                        "vanilla_cached_vs_full",
                        reference_full,
                        reference_step[0],
                    )
                    record(
                        batch,
                        case,
                        "forecast",
                        generation,
                        "cuda_cached_vs_full",
                        native_full,
                        native_step[0],
                    )
                    expected_token = reference_full.argmax(-1)
                    actual_token = native_full.argmax(-1)
                    tokens.append(
                        {
                            "batch": batch,
                            "case": case,
                            "generation": generation,
                            "reference": expected_token.cpu(),
                            "native": actual_token.cpu(),
                        }
                    )
                    ids = torch.cat((ids, expected_token[:, None]), dim=1)
                    reference_step = recurrent_step(
                        vanilla, expected_token[:, None], reference_state
                    )
                    native_step = recurrent_step(
                        native, expected_token[:, None], native_state
                    )
                    reference_state, native_state = reference_step[1], native_step[1]
                    record_step(
                        batch, case, "decode", generation, reference_step, native_step
                    )
    assert len(pairs) == 675 and len(tokens) == 36
    torch.save({"pairs": pairs, "tokens": tokens}, raw)
    rows = []
    for batch in batches:
        for case in range(3):
            tensor_rows = [
                pair for pair in pairs if (pair["batch"], pair["case"]) == (batch, case)
            ]
            token_rows = [
                row for row in tokens if (row["batch"], row["case"]) == (batch, case)
            ]
            assert len(tensor_rows) == 75 and len(token_rows) == 4
            rows.append(
                {
                    "batch": batch,
                    "case": case,
                    "tensor_pairs": len(tensor_rows),
                    "failed_elements": sum(
                        row["producer_failed"] for row in tensor_rows
                    ),
                    "token_rows": len(token_rows),
                    "token_mismatches": sum(
                        int(not torch.equal(row["reference"], row["native"]))
                        for row in token_rows
                    ),
                }
            )
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    passed = all(pair["producer_failed"] == 0 for pair in pairs) and all(
        torch.equal(row["reference"], row["native"]) for row in tokens
    )
    meta.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        tensor_pairs=len(pairs),
        token_rows=len(tokens),
        rows_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"status": meta["status"], "tensor_pairs": len(pairs)}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
