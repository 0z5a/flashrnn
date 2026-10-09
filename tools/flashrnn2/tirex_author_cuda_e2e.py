"""Time complete TiRex Torch and original CUDA forecasts after parity gates."""

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

import torch
from tirex_author_cuda_forecast_gate import contexts, forecast
from tirex_slstm_reference_gate import (
    SOURCE_REVISION,
    WEIGHT_BYTES,
    WEIGHT_SHA256,
    add_training_import_shim,
    git_blob_sha,
    sha256,
)

XLSTM_SOURCE_SHA256 = "dc919005b21352a0b65f075c2dc81012303ee23b78d9bc091c2b76e48a871025"


def burst(model, fixtures):
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    outputs = []
    completion_ms = []
    for context in fixtures:
        outputs.append(forecast(model, context.clone()))
        torch.cuda.synchronize()
        completion_ms.append((time.perf_counter_ns() - started) / 1e6)
    elapsed_ms = completion_ms[-1]
    return outputs, {
        "elapsed_ms": elapsed_ms,
        "series_per_second": sum(context.shape[0] for context in fixtures)
        * 1000
        / elapsed_ms,
        "completion_ms": completion_ms,
    }


def check_outputs(goldens, outputs):
    assert len(goldens) == len(outputs)
    for expected, actual in zip(goldens, outputs):
        assert len(expected) == len(actual) == 2
        for target, value in zip(expected, actual):
            torch.testing.assert_close(value, target, atol=1e-4, rtol=1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--xlstm-source", type=Path, required=True)
    parser.add_argument("--xlstm-source-archive", type=Path, required=True)
    parser.add_argument("--forecast-gate", type=Path, required=True)
    parser.add_argument("--forecast-audit", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.blocks < 20:
        parser.error("at least 20 paired blocks required")
    meta_path = args.output.with_suffix(".meta.json")
    if args.output.exists() or meta_path.exists():
        parser.error("output already exists")
    gate = json.loads(args.forecast_gate.read_text())
    audit = json.loads(args.forecast_audit.read_text())
    if (
        gate["status"] != "PASS"
        or audit["status"] != "PASS"
        or gate["batch"] != args.batch
        or gate["concurrency"] != args.concurrency
        or gate["budget"] != 1e-4
        or gate["baseline"] != "original TiRex xLSTM CUDA cell"
        or audit["gate_sha256"] != sha256(args.forecast_gate)
        or gate["raw_sha256"] != sha256(args.forecast_gate.with_suffix(".pt"))
        or audit["raw_sha256"] != gate["raw_sha256"]
        or audit["required_pairs"] != audit["observed_pairs"]
        or audit["required_pairs"] != 2 * gate["groups"]
        or not audit["fixture_pass"]
        or audit["missing"]
        or audit["unexpected"]
        or audit["reported_independent_disagreements"]
        or audit["failed_pairs"]
    ):
        parser.error("exact full-forecast independent numerical qualification required")
    weight = args.model / "model.ckpt"
    assert weight.stat().st_size == WEIGHT_BYTES and sha256(weight) == WEIGHT_SHA256
    pins = json.loads((args.source / "source-pins.json").read_text())
    assert pins["revision"] == SOURCE_REVISION
    for entry in pins["files"]:
        path = args.source / entry["path"]
        assert (
            path.stat().st_size == entry["bytes"] and git_blob_sha(path) == entry["sha"]
        )
    assert sha256(args.xlstm_source_archive) == XLSTM_SOURCE_SHA256
    root = "xlstm-ab22eadbd245f293dd8dec38ed29963d73758a12/"
    with tarfile.open(args.xlstm_source_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                assert member.name.startswith(root)
                path = args.xlstm_source / member.name.removeprefix(root)
                assert path.read_bytes() == archive.extractfile(member).read()
    assert torch.cuda.device_count() == 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    shim = add_training_import_shim()
    sys.path.insert(0, str(args.xlstm_source))
    sys.path.insert(0, str(args.source / "src"))
    from tirex.models.tirex import TiRexZero

    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    state = {
        name.removeprefix("block_stack."): value
        for name, value in checkpoint["state_dict"].items()
    }
    reference = TiRexZero(backend="torch", **checkpoint["hyper_parameters"])
    reference.load_state_dict(state, strict=True)
    candidate = TiRexZero(backend="cuda", **checkpoint["hyper_parameters"])
    candidate.load_state_dict(state, strict=True)
    reference.to("cuda").eval()
    candidate.to("cuda").eval()
    del checkpoint, state
    fixtures = contexts(args.concurrency, args.batch)
    fixture_sha = hashlib.sha256(torch.cat(fixtures).numpy().tobytes()).hexdigest()
    assert fixture_sha == gate["fixture_sha256"]
    saved = torch.load(
        args.forecast_gate.with_suffix(".pt"), map_location="cpu", weights_only=True
    )
    goldens = [entry["reference"] for entry in saved["groups"]]
    assert len(goldens) == len(fixtures)
    meta = {
        "status": "RUNNING",
        "model": "TiRex 35M complete 12-block forecast",
        "baseline": "official Torch",
        "candidate": "original author CUDA",
        "batch": args.batch,
        "concurrency": args.concurrency,
        "blocks": args.blocks,
        "budget": 1e-4,
        "checkpoint_sha256": WEIGHT_SHA256,
        "source_revision": SOURCE_REVISION,
        "xlstm_source_sha256": XLSTM_SOURCE_SHA256,
        "forecast_gate_sha256": sha256(args.forecast_gate),
        "forecast_audit_sha256": sha256(args.forecast_audit),
        "fixture_sha256": fixture_sha,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "training_only_sklearn_import_shim": shim,
        "timed": "Complete forecast preprocessing, two patches, twelve recurrent blocks, quantiles and CPU outputs",
        "excluded": "Checkpoint loading, source verification, JIT, numerical qualification, warmup and JSON I/O",
        "arrival_policy": "All series ready at time zero; one worker executes batch-sized groups in order",
        "statistics": "At least 20 within-process paired AB/BA blocks",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    with torch.inference_mode():
        for model in (reference, candidate):
            outputs, _ = burst(model, fixtures)
            check_outputs(goldens, outputs)
        for _ in range(2):
            burst(reference, fixtures)
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
                        reference if arm == "baseline" else candidate, fixtures
                    )
                    check_outputs(goldens, outputs)
                handle.write(
                    json.dumps(
                        {
                            "block": block,
                            "order": order,
                            "baseline": timings["baseline"],
                            "candidate": timings["candidate"],
                            "status": "PASS",
                        }
                    )
                    + "\n"
                )
                handle.flush()
    meta["status"] = "PASS"
    meta["raw_sha256"] = sha256(args.output)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
