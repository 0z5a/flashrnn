"""Paired queued-request E2E for qualified mixed xLSTM backends."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from itertools import pairwise
from pathlib import Path

import torch
from lovecraft_xlstm_cuda_gate import (
    CONFIG_SHA256,
    CPU_META_SHA256,
    SOURCE_ARCHIVE_SHA256,
    TOKENIZER_SHA256,
    load_cuda_model,
    state_tensors,
)
from lovecraft_xlstm_reference_gate import (
    REVISION,
    WEIGHT_SHA256,
    compare,
    load_model,
    recurrent_step,
)

STATE_NAMES = (
    "mlstm_0",
    "mlstm_1",
    "mlstm_2",
    "mlstm_conv",
    "slstm_conv",
    "slstm_hcnm",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    data = torch.tensor(values, dtype=torch.float64)
    return {
        name: torch.quantile(data, q).item()
        for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
    }


def gate_cases(raw_path: Path, cpu_meta: dict, batch: int) -> list[dict]:
    saved = torch.load(raw_path, map_location="cpu", weights_only=True)
    cases = []
    for case in range(3):
        pairs = {
            (pair["phase"], pair["step"], pair["name"]): pair
            for pair in saved["pairs"]
            if pair["batch"] == batch and pair["case"] == case
        }
        tokens = {
            row["generation"]: row
            for row in saved["tokens"]
            if row["batch"] == batch and row["case"] == case
        }
        assert len(pairs) == 75 and len(tokens) == 4
        cases.append(
            {
                "input_ids": torch.tensor(
                    [
                        cpu_meta["input_ids"][(4 * case + index) % 12]
                        for index in range(batch)
                    ]
                ),
                "logits": [
                    pairs["forecast", step, "full_logits"]["reference"]
                    for step in range(4)
                ],
                "tokens": torch.stack([tokens[step]["reference"] for step in range(4)]),
                "state": {
                    name: pairs["decode", 3, name]["reference"] for name in STATE_NAMES
                },
                "timed_state": {
                    name: pairs["decode", 2, name]["reference"] for name in STATE_NAMES
                },
            }
        )
    return cases


def cohort(model, cases: list[dict], concurrency: int, verify: bool) -> dict:
    batch = cases[0]["input_ids"].shape[0]
    steps = 4
    groups = concurrency // batch
    selected = [cases[index % len(cases)] for index in range(groups)]
    states = [None] * groups
    logits = [None] * groups
    delivered = [[] for _ in selected]
    stamps = [[] for _ in selected]
    prefill_starts = []
    model_batches = {"prefill": [], "decode": []}
    captured_logits = [[] for _ in selected]
    cache_checks = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter_ns()
    for group, oracle in enumerate(selected):
        prefill_starts.append((time.perf_counter_ns() - started) / 1e6)
        ids = oracle["input_ids"].cuda()
        for position in range(ids.shape[1]):
            logits[group], states[group] = recurrent_step(
                model, ids[:, position : position + 1], states[group]
            )
            model_batches["prefill"].append(batch)
    for step in range(steps):
        for group, oracle in enumerate(selected):
            captured_logits[group].append(logits[group].clone())
            token = logits[group].argmax(-1)
            delivered[group].append(token.cpu())
            stamps[group].append((time.perf_counter_ns() - started) / 1e6)
            if step + 1 < steps:
                logits[group], states[group] = recurrent_step(
                    model, token[:, None], states[group]
                )
                model_batches["decode"].append(batch)
            elif verify:
                _, states[group] = recurrent_step(model, token[:, None], states[group])
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    logit_checks = [
        {
            "group": group,
            "step": step,
            "errors": [compare(value.cpu(), selected[group]["logits"][step])],
        }
        for group, values in enumerate(captured_logits)
        for step, value in enumerate(values)
    ]
    requests = []
    token_match = True
    for group, oracle in enumerate(selected):
        token_ids = torch.stack(delivered[group])
        token_match &= torch.equal(token_ids, oracle["tokens"])
        for row in range(batch):
            requests.append(
                {
                    "request_id": group * batch + row,
                    "case": group % len(cases),
                    "row": row,
                    "arrival_ms": 0,
                    "prefill_start_ms": prefill_starts[group],
                    "token_times_ms": stamps[group],
                    "token_ids": token_ids[:, row].tolist(),
                }
            )
        actual = state_tensors(states[group])
        expected = oracle["state"] if verify else oracle["timed_state"]
        cache_checks.append(
            {
                "group": group,
                "errors": [
                    {"tensor": name, **compare(actual[name].cpu(), expected[name])}
                    for name in STATE_NAMES
                ],
            }
        )
    passed = (
        token_match
        and all(error["pass"] for row in logit_checks for error in row["errors"])
        and all(error["pass"] for row in cache_checks for error in row["errors"])
    )
    return {
        "status": "PASS" if passed else "CORRECTNESS_FAILED",
        "batch": batch,
        "concurrency": concurrency,
        "admission_batches": [batch] * groups,
        "model_batches": model_batches,
        "peak_active_requests": concurrency,
        "request_count": len(requests),
        "output_tokens": concurrency * steps,
        "elapsed_ms": elapsed_ms,
        "tokens_per_second": concurrency * steps * 1000 / elapsed_ms,
        "requests_per_second": concurrency * 1000 / elapsed_ms,
        "ttft_ms": quantiles([row["token_times_ms"][0] for row in requests]),
        "queue_wait_ms": quantiles([row["prefill_start_ms"] for row in requests]),
        "request_latency_ms": quantiles(
            [row["token_times_ms"][-1] for row in requests]
        ),
        "tpot_ms": quantiles(
            [
                (row["token_times_ms"][-1] - row["token_times_ms"][0]) / (steps - 1)
                for row in requests
            ]
        ),
        "itl_ms": quantiles(
            [
                later - earlier
                for row in requests
                for earlier, later in pairwise(row["token_times_ms"])
            ]
        ),
        "token_match": token_match,
        "cache_checks": cache_checks,
        "cache_check_step": steps - 1 if verify else steps - 2,
        "logit_checks": logit_checks,
        "requests": requests,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--cpu-reference-meta", type=Path, required=True)
    parser.add_argument("--gate-raw", type=Path, required=True)
    parser.add_argument("--gate-rows", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument(
        "--candidate-backend",
        choices=("author_cuda", "flashrnn2"),
        default="author_cuda",
    )
    parser.add_argument("--author-gate-audit", type=Path)
    parser.add_argument("--cell-gate-audit", type=Path)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--batch", type=int, choices=(1, 2, 4, 16, 32, 64), required=True
    )
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.blocks >= 20 and not args.output.exists()
    assert len(set(args.concurrency)) == len(args.concurrency)
    assert all(c >= args.batch and c % args.batch == 0 for c in args.concurrency)
    assert digest(args.model / "model.safetensors") == WEIGHT_SHA256
    assert digest(args.model / "config.json") == CONFIG_SHA256
    assert digest(args.model / "tokenizer.json") == TOKENIZER_SHA256
    assert digest(args.source_archive) == SOURCE_ARCHIVE_SHA256
    assert digest(args.cpu_reference_meta) == CPU_META_SHA256
    audit = json.loads(args.gate_audit.read_text())
    gate_meta = json.loads(args.gate_raw.with_suffix(".meta.json").read_text())
    assert audit["status"] == gate_meta["status"] == "PASS"
    assert audit["raw_sha256"] == gate_meta["raw_sha256"] == digest(args.gate_raw)
    assert audit["rows_sha256"] == gate_meta["rows_sha256"] == digest(args.gate_rows)
    assert gate_meta["model_revision"] == REVISION
    assert gate_meta["weight_sha256"] == WEIGHT_SHA256
    assert gate_meta["source_archive_sha256"] == SOURCE_ARCHIVE_SHA256
    assert gate_meta["tensor_pairs"] == audit["tensor_pairs"] == 675
    assert gate_meta["token_rows"] == audit["token_rows"] == 36
    if args.batch not in gate_meta.get("batches", [1, 2, 4]):
        raise ValueError("the numerical gate did not qualify this batch")
    if args.candidate_backend == "flashrnn2":
        if args.author_gate_audit is None or args.cell_gate_audit is None:
            parser.error("FlashRNN2 needs author CUDA and cell audit artifacts")
        assert (
            gate_meta["scope"]
            == "PINNED_XLSTM_FLASHRNN2_FP32_FULL_MODEL_VS_AUTHOR_CUDA_NO_TIMING"
        )
        assert gate_meta["author_audit_sha256"] == digest(args.author_gate_audit)
        assert gate_meta["cell_audit_sha256"] == digest(args.cell_gate_audit)
        assert json.loads(args.author_gate_audit.read_text())["status"] == "PASS"
        assert json.loads(args.cell_gate_audit.read_text())["status"] == "PASS"
    else:
        assert (
            gate_meta["scope"]
            == "PINNED_COMPLETE_MIXED_XLSTM_AUTHOR_CUDA_VS_VANILLA_FP32_NO_TIMING"
        )
        assert gate_meta["cpu_reference_meta_sha256"] == digest(args.cpu_reference_meta)
    if torch.cuda.device_count() != 1 or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("expected one assigned GPU")
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
    assert uuid == args.expected_gpu_uuid
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cpu_meta = json.loads(args.cpu_reference_meta.read_text())
    assert cpu_meta["status"] == "PASS"
    cases = gate_cases(args.gate_raw, cpu_meta, args.batch)
    if args.candidate_backend == "flashrnn2":
        from flashrnn.flashrnn2.tirex_layout import recurrent_layout
        from flashrnn.flashrnn2.triton_persistent import recurrence
        from flashrnn.flashrnn2.xlstm_adapter import install_xlstm_cell, slstm_cell

        assert gate_meta["candidate_source_sha256"] == digest(
            Path(slstm_cell.__code__.co_filename)
        )
        assert gate_meta["kernel_source_sha256"] == digest(
            Path(recurrence.__code__.co_filename)
        )
        assert gate_meta["layout_source_sha256"] == digest(
            Path(recurrent_layout.__code__.co_filename)
        )

        candidate = load_cuda_model(args.model, args.source).cuda().eval()
        install_xlstm_cell(candidate)
        arms = {
            "author_cuda": load_cuda_model(args.model, args.source).cuda().eval(),
            "flashrnn2_fp32": candidate,
        }
    else:
        arms = {
            "author_vanilla": load_model(args.model, args.source).cuda().eval(),
            "author_cuda": load_cuda_model(args.model, args.source).cuda().eval(),
        }
    baseline_name, candidate_name = arms
    meta_path = args.output.with_suffix(".meta.json")
    metadata = {
        "status": "RUNNING",
        "device": "cuda",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_uuid": uuid,
        "model_revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "gate_audit_sha256": digest(args.gate_audit),
        "gate_raw_sha256": digest(args.gate_raw),
        "cpu_reference_meta_sha256": digest(args.cpu_reference_meta),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "runner_sha256": digest(Path(__file__)),
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "arm_order": list(arms),
        "oracles": {
            "batch": args.batch,
            "generated_tokens": 4,
            "prefill_model_calls": 5,
            "cache_checks": len(STATE_NAMES),
            "cache_check_names": list(STATE_NAMES),
            "measurement_oracle_checks": True,
            "budget": {"atol": 1e-4, "rtol": 1e-4},
        },
        "concurrency": args.concurrency,
        "verification_only": False,
        "paired_blocks": args.blocks,
        "performance_claim": (
            None
            if args.candidate_backend == "flashrnn2"
            else "author native CUDA baseline only; no FlashRNN2 candidate"
        ),
        "timed": "Queued tokens to host-visible generated tokens, complete embedding, mixed recurrent blocks, full-vocabulary readout and device logit snapshots",
        "excluded": "Weight loading, tokenization, numerical warmup and JSON output",
        "arrival_policy": "All requests at time zero; static B-sized groups and round-robin decode",
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.inference_mode(), args.output.open("x") as handle:
        for concurrency in args.concurrency:
            for label, model in arms.items():
                row = cohort(model, cases, concurrency, True)
                row.update(arm=label, phase="qualification_and_warmup")
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if row["status"] != "PASS":
                    metadata.update(
                        status="FAIL_NUMERICAL",
                        failed_concurrency=concurrency,
                        rows_sha256=digest(args.output),
                    )
                    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
                    raise SystemExit(1)
            for block in range(args.blocks):
                order = list(arms) if block % 2 == 0 else list(reversed(arms))
                for label in order:
                    row = cohort(arms[label], cases, concurrency, False)
                    row.update(arm=label, phase="measurement", block=block, order=order)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    if row["status"] != "PASS":
                        metadata.update(
                            status="FAIL_NUMERICAL",
                            failed_concurrency=concurrency,
                            failed_block=block,
                            rows_sha256=digest(args.output),
                        )
                        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
                        raise SystemExit(1)
            print(
                json.dumps({"batch": args.batch, "concurrency": concurrency}),
                flush=True,
            )
    metadata.update(
        status="PASS", finished=time.time(), rows_sha256=digest(args.output)
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
