"""Qualify the pinned xLSTM sLSTM cell against its author CUDA backend."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from lovecraft_xlstm_cuda_gate import SOURCE_ARCHIVE_SHA256, load_cuda_model
from lovecraft_xlstm_reference_gate import REVISION, WEIGHT_SHA256, compare

from flashrnn.flashrnn2.tirex_layout import recurrent_layout
from flashrnn.flashrnn2.triton_persistent import recurrence
from flashrnn.flashrnn2.xlstm_adapter import slstm_cell

CASES = (
    (1, 1, "zero"),
    (2, 16, "positive"),
    (4, 64, "positive"),
    (4, 16, "mixed_zero"),
)
STATE_NAMES = ("hidden", "cell", "normalizer", "stabilizer")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inputs(batch: int, steps: int, state_kind: str, device: str):
    generator = torch.Generator(device=device).manual_seed(20261009 + batch + steps)
    gates = torch.randn(batch, steps, 256, device=device, generator=generator) * 0.25
    if state_kind == "zero":
        return gates, None
    state = torch.randn(4, batch, 64, device=device, generator=generator) * 0.1
    state[2] = 1.0 + state[2].abs()
    if state_kind == "mixed_zero":
        state[2, 0] = 0.0
    return gates, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.output.with_suffix(".pt")
    meta_path = args.output.with_suffix(".meta.json")
    assert not any(path.exists() for path in (args.output, raw, meta_path))
    assert digest(args.model / "model.safetensors") == WEIGHT_SHA256
    assert digest(args.source_archive) == SOURCE_ARCHIVE_SHA256
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
    if uuid != args.expected_gpu_uuid:
        raise ValueError("GPU identity differs from admission")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = {
        "status": "RUNNING",
        "scope": "PINNED_XLSTM_FP32_FLASHRNN2_CELL_VS_AUTHOR_CUDA_NO_TIMING",
        "started": time.time(),
        "model_revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "candidate_source_sha256": digest(Path(slstm_cell.__code__.co_filename)),
        "kernel_source_sha256": digest(Path(recurrence.__code__.co_filename)),
        "layout_source_sha256": digest(Path(recurrent_layout.__code__.co_filename)),
        "gpu_uuid": uuid,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "budget": {"atol": 1e-4, "rtol": 1e-4},
        "performance_claim": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    model = load_cuda_model(args.model, args.source).cuda().eval()
    cell = model.xlstm_block_stack.blocks[1].xlstm.slstm_cell
    records = []
    rows = []
    with torch.inference_mode():
        for batch, steps, state_kind in CASES:
            gates, state = inputs(batch, steps, state_kind, "cuda")
            reference = cell(gates, state)
            candidate = slstm_cell(gates, state, cell._recurrent_kernel_, cell._bias_)
            comparisons = (("output", reference[0], candidate[0]),) + tuple(
                (name, reference[1][index], candidate[1][index])
                for index, name in enumerate(STATE_NAMES)
            )
            failed = 0
            for name, expected, actual in comparisons:
                expected_cpu, actual_cpu = (
                    expected.detach().cpu(),
                    actual.detach().cpu(),
                )
                count = compare(actual_cpu, expected_cpu)["failed_elements"]
                failed += count
                records.append(
                    {
                        "batch": batch,
                        "steps": steps,
                        "state_kind": state_kind,
                        "name": name,
                        "reference": expected_cpu,
                        "candidate": actual_cpu,
                        "producer_failed": count,
                    }
                )
            rows.append(
                {
                    "batch": batch,
                    "steps": steps,
                    "state_kind": state_kind,
                    "tensor_pairs": len(comparisons),
                    "failed_elements": failed,
                }
            )
    assert len(records) == 20 and len(rows) == len(CASES)
    torch.save(records, raw)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    meta.update(
        status="PASS"
        if all(row["failed_elements"] == 0 for row in rows)
        else "NUMERICAL_FAILED",
        finished=time.time(),
        tensor_pairs=len(records),
        raw_sha256=digest(raw),
        rows_sha256=digest(args.output),
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    raise SystemExit(0 if meta["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
