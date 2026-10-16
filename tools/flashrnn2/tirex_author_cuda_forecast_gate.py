"""Gate complete TiRex forecasts before timing the author's CUDA backend."""

import argparse
import hashlib
import json
import sys
import tarfile
import traceback
from pathlib import Path

import torch
from tirex_slstm_reference_gate import (
    SOURCE_REVISION,
    WEIGHT_BYTES,
    WEIGHT_SHA256,
    add_training_import_shim,
    compare,
    git_blob_sha,
    sha256,
)

XLSTM_SOURCE_SHA256 = "dc919005b21352a0b65f075c2dc81012303ee23b78d9bc091c2b76e48a871025"
BUDGET = 1e-4


def contexts(concurrency, batch):
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


def forecast(model, context):
    return tuple(
        output.detach().cpu().clone()
        for output in model.forecast(
            context,
            prediction_length=64,
            output_type="torch",
            batch_size=context.shape[0],
            full_rollout=False,
            dynamic_padding=False,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--xlstm-source", type=Path, required=True)
    parser.add_argument("--xlstm-source-archive", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--batch", type=int, choices=(1, 2, 4, 16, 32, 64), required=True
    )
    parser.add_argument(
        "--concurrency", type=int, choices=(1, 8, 32, 64, 128), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < args.batch or args.concurrency % args.batch:
        parser.error("concurrency must be a multiple of batch")
    raw_path = args.output.with_suffix(".pt")
    if args.output.exists() or raw_path.exists():
        parser.error("output already exists")
    gate = json.loads(args.gate.read_text())
    audit = json.loads(args.audit.read_text())
    batch_set = gate.get("batch_set", "small")
    qualified = (1, 2, 4) if batch_set == "small" else (16, 32, 64)
    if (
        args.batch not in qualified
        or gate["status"] != "PASS"
        or audit["status"] != "PASS"
        or gate["baseline"] != "original TiRex xLSTM CUDA cell"
        or gate["budget"] != BUDGET
        or gate["checkpoint_sha256"] != WEIGHT_SHA256
        or gate["source_revision"] != SOURCE_REVISION
        or gate["xlstm_source_sha256"] != XLSTM_SOURCE_SHA256
        or len(gate["comparisons"]) != 381
        or not all(row["pass"] for row in gate["comparisons"])
        or sha256(args.gate.with_suffix(".pt")) != gate["raw_sha256"]
        or audit["gate_sha256"] != sha256(args.gate)
        or audit["raw_sha256"] != gate["raw_sha256"]
        or audit["required_pairs"] != 381
        or audit["observed_pairs"] != 381
    ):
        parser.error(
            "complete independent original-CUDA numerical qualification required"
        )
    weight = args.model / "model.ckpt"
    assert weight.stat().st_size == WEIGHT_BYTES and sha256(weight) == WEIGHT_SHA256
    pins = json.loads((args.source / "source-pins.json").read_text())
    assert pins["revision"] == SOURCE_REVISION
    for entry in pins["files"]:
        path = args.source / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert git_blob_sha(path) == entry["sha"]
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

    fixtures = contexts(args.concurrency, args.batch)
    fixture_sha = hashlib.sha256(torch.cat(fixtures).numpy().tobytes()).hexdigest()
    raw = {
        "groups": [
            {
                "group": group,
                "context": context.clone(),
                "reference": None,
                "candidate": None,
            }
            for group, context in enumerate(fixtures)
        ]
    }
    rows = []
    status = "INCOMPLETE"
    failure = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
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
        with torch.inference_mode():
            for arm, model in (("reference", reference), ("candidate", candidate)):
                for entry in raw["groups"]:
                    entry[arm] = forecast(model, entry["context"].clone())
            for group, entry in enumerate(raw["groups"]):
                for name, expected, actual in zip(
                    ("quantiles", "median"), entry["reference"], entry["candidate"]
                ):
                    if expected.shape != actual.shape:
                        row = {
                            "pass": False,
                            "reference_shape": list(expected.shape),
                            "candidate_shape": list(actual.shape),
                        }
                    else:
                        row = compare(actual.float(), expected.float(), BUDGET)
                    rows.append({"group": group, "tensor": name} | row)
        assert len(rows) == 2 * len(fixtures)
        status = "PASS" if all(row["pass"] for row in rows) else "FAIL_NUMERICAL"
    except Exception:
        status = "ERROR"
        failure = traceback.format_exc()
        raise
    finally:
        torch.save(raw, raw_path)
        args.output.write_text(
            json.dumps(
                {
                    "status": status,
                    "failure_traceback": failure,
                    "baseline": "original TiRex xLSTM CUDA cell",
                    "batch": args.batch,
                    "concurrency": args.concurrency,
                    "groups": len(fixtures),
                    "budget": BUDGET,
                    "checkpoint_sha256": WEIGHT_SHA256,
                    "source_revision": SOURCE_REVISION,
                    "xlstm_source_sha256": XLSTM_SOURCE_SHA256,
                    "qualification_gate_sha256": sha256(args.gate),
                    "qualification_audit_sha256": sha256(args.audit),
                    "fixture_sha256": fixture_sha,
                    "raw_sha256": sha256(raw_path),
                    "training_only_sklearn_import_shim": shim,
                    "comparisons": rows,
                },
                indent=2,
            )
            + "\n"
        )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
