"""State-aware, long-sequence FP8 weight-error exploration on the existing CPU.

R is packed once, then dequantized to BF16 outside the recurrence. This is
not an FP8 MMA kernel, GPU speed test or pretrained-model quality result.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from flashrnn.flashrnn2.reference import SIZES, recurrence


def pack(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    limit = torch.finfo(torch.float8_e4m3fn).max
    scale = weights.float().abs().amax(dim=(-2, -1), keepdim=True) / limit
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    encoded = (weights.float() / scale).to(torch.float8_e4m3fn)
    return encoded, scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261011)
    metadata = {
        "scope": "CPU_LONG_SEQUENCE_FP8_R_WEIGHT_ERROR_ONLY",
        "started": time.time(),
        "torch": torch.__version__,
        "seed": 20261011,
        "state": "FP32 local/history states; recurrent hidden rounded to BF16",
        "weights": "BF16 baseline; E4M3FN per-gate/head symmetric amax scale; unpack once to BF16",
        "input_projection_bias_initial": "BF16 values evaluated in FP32 reference",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(
            Path(recurrence.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "gpu_executed": False,
        "fp8_mma_executed": False,
        "accuracy_acceptance": "UNQUALIFIED; error characterization only",
        "gradients": "NOT_RUN",
        "speed_claim": False,
        "status": "RUNNING",
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    # A completely zero gate must remain exactly zero without a zero scale.
    zero = torch.zeros(4, 1, 64, 64, dtype=torch.bfloat16)
    encoded, scales = pack(zero)
    assert torch.equal(encoded.float() * scales, zero.float())
    rows = []
    with torch.no_grad(), args.output.open("x") as handle:
        for cell in SIZES:
            gates, bias_gates, states = SIZES[cell]
            names = {
                "lstm": ("h", "c"),
                "slstm": ("h", "c", "n", "m"),
                "gru": ("h",),
                "elman": ("h",),
            }[cell]
            for length in (4096, 16384):
                for gain in (0.1, 1.0):
                    wx = (torch.randn(1, length, gates, 1, 64) * 0.1).bfloat16()
                    r = (torch.randn(gates, 1, 64, 64) * gain / 8).bfloat16()
                    bias = (torch.randn(bias_gates, 1, 64) * 0.05).bfloat16()
                    # Exercise persistent memory, not only a rapidly forgetting state.
                    if cell in ("lstm", "slstm"):
                        bias[1].add_(2)
                    elif cell == "gru":
                        bias[2].add_(2)
                    initial = torch.zeros(states, 1, 1, 1, 64)
                    encoded, scale = pack(r)
                    unpacked = (encoded.float() * scale).bfloat16()
                    target = recurrence(
                        wx.float(),
                        r.float(),
                        bias.float(),
                        initial,
                        cell,
                        mma_dtype=torch.bfloat16,
                    )[0]
                    actual = recurrence(
                        wx.float(),
                        unpacked.float(),
                        bias.float(),
                        initial,
                        cell,
                        mma_dtype=torch.bfloat16,
                    )[0]
                    delta = actual - target
                    snapshots = {}
                    for end in (16, 128, 1024, 4096, 16384):
                        if end > length:
                            continue
                        snapshots[str(end)] = {
                            name: {
                                "max_abs": delta[index, :, :end].abs().max().item(),
                                "relative_l2": (
                                    delta[index, :, :end].norm()
                                    / target[index, :, :end].norm().clamp_min(1e-30)
                                ).item(),
                                "last_max_abs": delta[index, :, end - 1]
                                .abs()
                                .max()
                                .item(),
                            }
                            for index, name in enumerate(names)
                        }
                    packed_bytes = (
                        encoded.numel() * encoded.element_size()
                        + scale.numel() * scale.element_size()
                    )
                    baseline_bytes = r.numel() * r.element_size()
                    finite = (
                        torch.isfinite(actual).all().item()
                        and torch.isfinite(target).all().item()
                    )
                    row = {
                        "cell": cell,
                        "B_T_H_D": [1, length, 1, 64],
                        "recurrent_gain": gain,
                        "r_max_abs_error": (unpacked.float() - r.float())
                        .abs()
                        .max()
                        .item(),
                        "bf16_weight_bytes": baseline_bytes,
                        "fp8_weight_and_scale_bytes": packed_bytes,
                        "weight_storage_ratio": baseline_bytes / packed_bytes,
                        "state_errors_by_prefix": snapshots,
                        "status": "MEASURED_FINITE_ACCURACY_UNQUALIFIED"
                        if finite
                        else "NONFINITE",
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "cell": cell,
                                "T": length,
                                "gain": gain,
                                "max_abs_by_state": {
                                    name: snapshots[str(length)][name]["max_abs"]
                                    for name in names
                                },
                                "status": row["status"],
                            }
                        ),
                        flush=True,
                    )
    failures = sum(row["status"] == "NONFINITE" for row in rows)
    metadata.update(
        finished=time.time(),
        rows=len(rows),
        nonfinite=failures,
        status="ERROR_CHARACTERIZATION_COMPLETE"
        if not failures
        else "NONFINITE_RETAINED",
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
