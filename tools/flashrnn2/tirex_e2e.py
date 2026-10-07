"""Paired full-checkpoint TiRex forecast E2E after GPU numerical qualification."""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import triton
from tirex_slstm_reference_gate import (
    SOURCE_REVISION,
    WEIGHT_BYTES,
    WEIGHT_SHA256,
    add_training_import_shim,
    git_blob_sha,
    sha256,
)

from flashrnn.flashrnn2.tirex_adapter import install_tirex_cells


def contexts(concurrency: int, batch: int) -> list[torch.Tensor]:
    t = torch.arange(128, dtype=torch.float32)
    series = torch.stack(
        [
            (1 + 0.03 * index) * torch.sin(t / (7 + index % 5) + index / 3)
            + 0.004 * (index + 1) * t
            + 0.2 * torch.cos(t / (17 + index % 4))
            for index in range(concurrency)
        ]
    )
    return list(series.split(batch))


def burst(model, fixtures):
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = []
    finished_ms = []
    for context in fixtures:
        outputs.append(
            model.forecast(
                context,
                prediction_length=64,
                output_type="torch",
                batch_size=context.shape[0],
                full_rollout=False,
                dynamic_padding=False,
            )
        )
        torch.cuda.synchronize()
        finished_ms.append(1000 * (time.perf_counter() - start))
    elapsed_ms = finished_ms[-1]
    return outputs, {
        "elapsed_ms": elapsed_ms,
        "series_per_second": sum(x.shape[0] for x in fixtures) * 1000 / elapsed_ms,
        "last_response_ms": finished_ms[-1],
        "completion_ms": finished_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch not in (1, 2, 4) or args.concurrency < args.batch:
        parser.error("requires B=1/2/4 and concurrency >= batch")
    if args.concurrency % args.batch or args.blocks < 20:
        parser.error("concurrency must be divisible by batch; at least 20 blocks")
    meta_path = args.output.with_suffix(".meta.json")
    if args.output.exists() or meta_path.exists():
        parser.error("output already exists")

    gate = json.loads(args.gate.read_text())
    if gate["status"] != "PASS":
        parser.error("TiRex GPU numerical gate has not passed")
    if gate["budget"] != 1e-4 or len(gate["comparisons"]) != 381:
        parser.error("complete TiRex gate at the pinned 1e-4 budget required")
    if not all(row["pass"] for row in gate["comparisons"]):
        parser.error("TiRex gate contains failed tensor comparisons")
    if (
        gate["checkpoint_sha256"] != WEIGHT_SHA256
        or gate["source_revision"] != SOURCE_REVISION
    ):
        parser.error("qualification used a different checkpoint or source")
    if sha256(args.gate.with_suffix(".pt")) != gate["raw_sha256"]:
        parser.error("qualification raw tensor archive changed")
    weight = args.model / "model.ckpt"
    if weight.stat().st_size != WEIGHT_BYTES or sha256(weight) != WEIGHT_SHA256:
        parser.error("checkpoint bytes or SHA do not match")
    pins = json.loads((args.source / "source-pins.json").read_text())
    assert pins["revision"] == SOURCE_REVISION
    for entry in pins["files"]:
        source = args.source / entry["path"]
        assert source.stat().st_size == entry["bytes"]
        assert git_blob_sha(source) == entry["sha"]

    shim = add_training_import_shim()
    sys.path.insert(0, str(args.source / "src"))
    from tirex.models.tirex import TiRexZero

    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    model = TiRexZero(backend="torch", **checkpoint["hyper_parameters"])
    model.load_state_dict(
        {
            name.replace("block_stack.", ""): value
            for name, value in checkpoint["state_dict"].items()
        },
        strict=True,
    )
    model.to("cuda").eval()
    candidate = copy.deepcopy(model)
    install_tirex_cells(candidate)
    del checkpoint
    fixtures = contexts(args.concurrency, args.batch)
    fixture_sha = hashlib.sha256(torch.cat(fixtures).numpy().tobytes()).hexdigest()
    meta = {
        "status": "RUNNING",
        "model": "TiRex 35M, 12 sLSTM blocks",
        "checkpoint_sha256": WEIGHT_SHA256,
        "source_revision": SOURCE_REVISION,
        "gate_sha256": sha256(args.gate),
        "gate_raw_sha256": gate["raw_sha256"],
        "gate_budget": gate["budget"],
        "batch": args.batch,
        "concurrency": args.concurrency,
        "blocks": args.blocks,
        "fixture_sha256": fixture_sha,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "training_only_sklearn_import_shim": shim,
        "timed": "Complete public forecast: preprocessing, two 32-step patches, all 12 blocks, quantiles and CPU output",
        "excluded": "Checkpoint loading, JIT, numerical qualification, fixture generation, warmup and JSON I/O",
        "arrival_policy": "All C series ready at time zero; one worker executes B-sized groups in order",
        "statistics": "20 within-process paired AB/BA blocks; no independent-start interval",
        "memory_scope": "Both models remain resident; no per-arm memory claim",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    with torch.inference_mode():
        goldens, _ = burst(model, fixtures)
        candidate_outputs, _ = burst(candidate, fixtures)
        for expected, actual in zip(goldens, candidate_outputs):
            for baseline_tensor, candidate_tensor in zip(expected, actual):
                torch.testing.assert_close(
                    candidate_tensor,
                    baseline_tensor,
                    atol=gate["budget"],
                    rtol=gate["budget"],
                )
        for _ in range(2):
            burst(model, fixtures)
            burst(candidate, fixtures)
        with args.output.open("x") as handle:
            for block in range(args.blocks):
                order = (
                    ("baseline", "candidate")
                    if block % 2 == 0
                    else ("candidate", "baseline")
                )
                timings = {}
                for arm in order:
                    outputs, timings[arm] = burst(
                        model if arm == "baseline" else candidate, fixtures
                    )
                    for expected, actual in zip(goldens, outputs):
                        for baseline_tensor, output_tensor in zip(expected, actual):
                            torch.testing.assert_close(
                                output_tensor,
                                baseline_tensor,
                                atol=gate["budget"],
                                rtol=gate["budget"],
                            )
                row = {
                    "block": block,
                    "order": order,
                    "baseline": timings["baseline"],
                    "candidate": timings["candidate"],
                    "status": "PASS",
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
    meta["status"] = "PASS"
    meta["raw_sha256"] = sha256(args.output)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
