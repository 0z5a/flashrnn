"""Qualify the pinned full xLSTM checkpoint with the FlashRNN2 sLSTM cell."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from lovecraft_xlstm_cuda_gate import (
    SOURCE_ARCHIVE_SHA256,
    load_cuda_model,
    state_tensors,
)
from lovecraft_xlstm_reference_gate import (
    PROMPTS,
    REVISION,
    WEIGHT_SHA256,
    compare,
    full_forward,
    recurrent_step,
)
from tokenizers import Tokenizer

from flashrnn.flashrnn2.tirex_layout import recurrent_layout
from flashrnn.flashrnn2.triton_persistent import recurrence
from flashrnn.flashrnn2.xlstm_adapter import install_xlstm_cell, slstm_cell

SCOPE = "PINNED_XLSTM_FLASHRNN2_FP32_FULL_MODEL_VS_AUTHOR_CUDA_NO_TIMING"
BUDGET = {"atol": 1e-4, "rtol": 1e-4}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def qualified(
    raw: Path, audit: Path, expected_audit_sha256: str, pairs: int, scope: str
) -> dict:
    meta = json.loads(raw.with_suffix(".meta.json").read_text())
    report = json.loads(audit.read_text())
    if (
        digest(audit) != expected_audit_sha256
        or meta["status"] != "PASS"
        or meta["scope"] != scope
        or report["status"] != "PASS"
        or meta["raw_sha256"] != digest(raw)
        or report["raw_sha256"] != digest(raw)
        or report["rows_sha256"] != meta["rows_sha256"]
        or report["tensor_pairs"] != pairs
        or meta["weight_sha256"] != WEIGHT_SHA256
        or meta["model_revision"] != REVISION
        or meta["source_archive_sha256"] != SOURCE_ARCHIVE_SHA256
        or meta["budget"] != BUDGET
    ):
        raise ValueError("prior independent numerical qualification is incomplete")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--author-raw", type=Path, required=True)
    parser.add_argument("--author-audit", type=Path, required=True)
    parser.add_argument("--author-audit-sha256", required=True)
    parser.add_argument("--cell-raw", type=Path, required=True)
    parser.add_argument("--cell-audit", type=Path, required=True)
    parser.add_argument("--cell-audit-sha256", required=True)
    parser.add_argument("--small-full-raw", type=Path)
    parser.add_argument("--small-full-audit", type=Path)
    parser.add_argument("--small-full-audit-sha256")
    parser.add_argument("--batch-set", choices=("small", "large"), default="small")
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batches = (1, 2, 4) if args.batch_set == "small" else (16, 32, 64)
    raw = args.output.with_suffix(".pt")
    meta_path = args.output.with_suffix(".meta.json")
    if any(path.exists() for path in (args.output, raw, meta_path)):
        parser.error("output already exists")
    author_meta = qualified(
        args.author_raw,
        args.author_audit,
        args.author_audit_sha256,
        675,
        "PINNED_COMPLETE_MIXED_XLSTM_AUTHOR_CUDA_VS_VANILLA_FP32_NO_TIMING",
    )
    if author_meta.get("batch_set", "small") != args.batch_set:
        raise ValueError("author CUDA gate did not qualify this batch set")
    cell_meta = qualified(
        args.cell_raw,
        args.cell_audit,
        args.cell_audit_sha256,
        20,
        "PINNED_XLSTM_FP32_FLASHRNN2_CELL_VS_AUTHOR_CUDA_NO_TIMING",
    )
    candidate_source_sha256 = digest(Path(slstm_cell.__code__.co_filename))
    kernel_source_sha256 = digest(Path(recurrence.__code__.co_filename))
    layout_source_sha256 = digest(Path(recurrent_layout.__code__.co_filename))
    if (
        cell_meta["candidate_source_sha256"] != candidate_source_sha256
        or cell_meta["kernel_source_sha256"] != kernel_source_sha256
        or cell_meta["layout_source_sha256"] != layout_source_sha256
    ):
        raise ValueError("FlashRNN2 candidate differs from qualified cell source")
    if args.batch_set == "large":
        if (
            args.small_full_raw is None
            or args.small_full_audit is None
            or args.small_full_audit_sha256 is None
        ):
            parser.error("large batches require a qualified small full-model gate")
        small_meta = qualified(
            args.small_full_raw,
            args.small_full_audit,
            args.small_full_audit_sha256,
            675,
            SCOPE,
        )
        if (
            small_meta.get("batch_set", "small") != "small"
            or small_meta["candidate_source_sha256"] != candidate_source_sha256
            or small_meta["kernel_source_sha256"] != kernel_source_sha256
            or small_meta["layout_source_sha256"] != layout_source_sha256
        ):
            raise ValueError("small full-model qualification used different source")
    if digest(args.model / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("checkpoint differs from pinned model")
    if digest(args.source_archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("author source differs from pinned revision")
    if torch.cuda.device_count() != 1 or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("expected one manager-assigned GPU")
    uuid = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if uuid != args.expected_gpu_uuid:
        raise ValueError("GPU identity differs from admission")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = {
        "status": "RUNNING",
        "scope": SCOPE,
        "started": time.time(),
        "model_revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "author_audit_sha256": digest(args.author_audit),
        "cell_audit_sha256": digest(args.cell_audit),
        "candidate_source_sha256": candidate_source_sha256,
        "kernel_source_sha256": kernel_source_sha256,
        "layout_source_sha256": layout_source_sha256,
        "batch_set": args.batch_set,
        "batches": list(batches),
        "small_full_audit_sha256": (
            digest(args.small_full_audit) if args.batch_set == "large" else None
        ),
        "gpu_uuid": uuid,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "budget": BUDGET,
        "performance_claim": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    reference = load_cuda_model(args.model, args.source).cuda().eval()
    candidate = load_cuda_model(args.model, args.source).cuda().eval()
    install_xlstm_cell(candidate)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    prompts = [
        [1, *tokenizer.encode(prompt, add_special_tokens=False).ids[:4]]
        for prompt in PROMPTS
    ]
    if len(prompts) != 12 or any(len(ids) != 5 for ids in prompts):
        raise ValueError("pinned prompt contract changed")
    pairs = []
    tokens = []

    def record(batch, case, phase, step, name, expected, actual):
        expected_cpu, actual_cpu = expected.detach().cpu(), actual.detach().cpu()
        count = (
            compare(actual_cpu, expected_cpu)["failed_elements"]
            if expected_cpu.shape == actual_cpu.shape
            else max(expected_cpu.numel(), actual_cpu.numel())
        )
        pairs.append(
            {
                "batch": batch,
                "case": case,
                "phase": phase,
                "step": step,
                "name": name,
                "reference": expected_cpu,
                "candidate": actual_cpu,
                "producer_failed": count,
            }
        )

    def record_step(batch, case, phase, step, expected, actual):
        record(batch, case, phase, step, "step_logits", expected[0], actual[0])
        expected_states = state_tensors(expected[1])
        actual_states = state_tensors(actual[1])
        for name in expected_states:
            record(
                batch,
                case,
                phase,
                step,
                name,
                expected_states[name],
                actual_states[name],
            )

    with torch.inference_mode():
        for batch in batches:
            for case in range(3):
                ids = torch.tensor(
                    [
                        prompts[(4 * case + index) % len(prompts)]
                        for index in range(batch)
                    ],
                    device="cuda",
                )
                reference_state = candidate_state = None
                for step in range(5):
                    expected = recurrent_step(
                        reference, ids[:, step : step + 1], reference_state
                    )
                    actual = recurrent_step(
                        candidate, ids[:, step : step + 1], candidate_state
                    )
                    reference_state, candidate_state = expected[1], actual[1]
                    record_step(batch, case, "prefill", step, expected, actual)
                for step in range(4):
                    expected_full = full_forward(reference, ids)
                    actual_full = full_forward(candidate, ids)
                    record(
                        batch,
                        case,
                        "forecast",
                        step,
                        "full_logits",
                        expected_full,
                        actual_full,
                    )
                    record(
                        batch,
                        case,
                        "forecast",
                        step,
                        "reference_cached_vs_full",
                        expected_full,
                        expected[0],
                    )
                    record(
                        batch,
                        case,
                        "forecast",
                        step,
                        "candidate_cached_vs_full",
                        actual_full,
                        actual[0],
                    )
                    expected_token = expected_full.argmax(-1)
                    actual_token = actual_full.argmax(-1)
                    tokens.append(
                        {
                            "batch": batch,
                            "case": case,
                            "generation": step,
                            "reference": expected_token.cpu(),
                            "candidate": actual_token.cpu(),
                        }
                    )
                    ids = torch.cat((ids, expected_token[:, None]), dim=1)
                    expected = recurrent_step(
                        reference, expected_token[:, None], reference_state
                    )
                    actual = recurrent_step(
                        candidate, expected_token[:, None], candidate_state
                    )
                    reference_state, candidate_state = expected[1], actual[1]
                    record_step(batch, case, "decode", step, expected, actual)
    if len(pairs) != 675 or len(tokens) != 36:
        raise ValueError("incomplete full-model comparison")
    torch.save({"pairs": pairs, "tokens": tokens}, raw)
    rows = []
    for batch in batches:
        for case in range(3):
            selected = [p for p in pairs if (p["batch"], p["case"]) == (batch, case)]
            selected_tokens = [
                t for t in tokens if (t["batch"], t["case"]) == (batch, case)
            ]
            rows.append(
                {
                    "batch": batch,
                    "case": case,
                    "tensor_pairs": len(selected),
                    "failed_elements": sum(p["producer_failed"] for p in selected),
                    "token_rows": len(selected_tokens),
                    "token_mismatches": sum(
                        int(not torch.equal(t["reference"], t["candidate"]))
                        for t in selected_tokens
                    ),
                }
            )
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    passed = all(row["failed_elements"] == row["token_mismatches"] == 0 for row in rows)
    meta.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        tensor_pairs=len(pairs),
        token_rows=len(tokens),
        raw_sha256=digest(raw),
        rows_sha256=digest(args.output),
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
