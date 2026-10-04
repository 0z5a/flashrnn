"""Locate the frozen native sLSTM zero-state failure without changing its kernel."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from flashrnn.flashrnn2.reference import recurrence as reference
from flashrnn.flashrnn2.upstream_triton import SOURCE_HASHES, recurrence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    # Consume exactly the first four r1 input draws; only the failed case runs.
    for cell, mode in (
        ("lstm", "nonzero"),
        ("lstm", "zero"),
        ("slstm", "nonzero"),
        ("slstm", "zero"),
    ):
        states = 2 if cell == "lstm" else 4
        wx = (torch.randn(3, 17, 4, 2, 64) * 0.1).bfloat16()
        r = (torch.randn(4, 2, 64, 64) * (0.1 / 64**0.5)).bfloat16()
        bias = (torch.randn(4, 2, 64) * 0.05).bfloat16()
        initial = (torch.randn(states, 3, 1, 2, 64) * 0.05).bfloat16()
        if cell == "slstm":
            initial[2].abs_().add_(1)
        if mode == "zero":
            initial.zero_()
    inputs = (wx, r, bias, initial)
    with torch.no_grad():
        target = reference(
            *(x.float() for x in inputs),
            "slstm",
            mma_dtype=torch.bfloat16,
            slstm_init="elementwise",
        )[0]
        actual = recurrence(*(x.cuda() for x in inputs), "slstm")[0].cpu()
    rounded = target.bfloat16()
    delta = (actual.float() - rounded.float()).abs()
    locations = torch.nonzero(delta > 0.003)
    detail = []
    for location in locations[:64]:
        index = tuple(location.tolist())
        detail.append(
            {
                "S_B_T_H_D": list(index),
                "actual_bf16": actual[index].item(),
                "reference_fp32": target[index].item(),
                "reference_bf16": rounded[index].item(),
                "absolute_error": delta[index].item(),
            }
        )
    snapshot = args.output.with_suffix(".pt")
    torch.save(
        {"inputs": inputs, "actual_bf16": actual, "reference_fp32": target}, snapshot
    )
    with snapshot.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    record = {
        "scope": "ONE_FROZEN_NATIVE_TRITON_ZERO_STATE_FAILURE_LOCALIZATION",
        "torch": torch.__version__,
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "kernel_sha256": SOURCE_HASHES["slstm"],
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapshot_file": snapshot.name,
        "snapshot_sha256": digest,
        "shape_B_T_H_D": [3, 17, 2, 64],
        "threshold_unchanged": 0.003,
        "max_abs_by_state": {
            name: delta[i].max().item() for i, name in enumerate(("h", "c", "n", "m"))
        },
        "over_budget_count": len(locations),
        "first_64_over_budget": detail,
        "status": "REPRODUCED_FAILURE" if len(locations) else "FAILURE_NOT_REPRODUCED",
        "fp32_kernel_snapshot": "NOT_EXECUTED; DTYPE also controls recurrent MMA operand cast",
        "speed_claim": False,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    raise SystemExit(1 if len(locations) else 0)


if __name__ == "__main__":
    main()
