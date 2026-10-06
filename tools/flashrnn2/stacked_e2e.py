"""Paired complete stacked-RNN sequence inference on a fixed synthetic model.

One queued cohort has C requests, served by one worker in B-sized groups. Timing
includes ID transfer, embedding, every recurrent layer, readout and host delivery.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from flashrnn.flashrnn2.torch_layer import TorchLayer
from flashrnn.flashrnn2.triton_persistent import recurrence as persistent
from flashrnn.flashrnn2.upstream_triton import recurrence as upstream


class StackedModel:
    def __init__(self, cell: str, baseline: str, layers: int, width: int) -> None:
        generator = torch.Generator().manual_seed(20261006)
        self.cell, self.baseline, self.width = cell, baseline, width
        self.embedding = (
            (torch.randn(512, width, generator=generator) * 0.1).bfloat16().cuda()
        )
        self.readout = (
            (torch.randn(width, 512, generator=generator) * 0.1).bfloat16().cuda()
        )
        self.weights = []
        self.modules = []
        for _ in range(layers):
            w = (
                (
                    torch.randn(4, 1, width, width, generator=generator)
                    * (0.2 / width**0.5)
                )
                .bfloat16()
                .cuda()
            )
            r = (
                (
                    torch.randn(4, 1, width, width, generator=generator)
                    * (0.2 / width**0.5)
                )
                .bfloat16()
                .cuda()
            )
            b = (torch.randn(4, 1, width, generator=generator) * 0.02).bfloat16().cuda()
            self.weights.append((w, r, b))
            self.modules.append(
                TorchLayer(w, r, b, "lstm") if baseline == "cudnn" else None
            )

    def forward(self, ids: torch.Tensor, arm: str, trace: bool = False):
        batch = ids.shape[0]
        x = self.embedding[ids.to("cuda")]
        states = 4 if self.cell == "slstm" else 2
        layers = []
        for (w, r, bias), module in zip(self.weights, self.modules):
            initial = torch.zeros(
                (states, batch, 1, 1, self.width), device="cuda", dtype=torch.bfloat16
            )
            if self.cell == "slstm":
                initial[2].fill_(1)
            if arm == "baseline" and module is not None:
                hidden, final = module(x, initial)
            else:
                wx = torch.einsum("bti,ghdi->btghd", x, w)
                kernel = upstream if arm == "baseline" else persistent
                history, final = kernel(wx, r, bias, initial, self.cell)
                hidden = history[0]
            x = hidden[:, :, 0]
            if trace:
                layers.append((hidden, final))
        logits = x[:, -1] @ self.readout
        return logits, layers


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(
    model: StackedModel, fixtures: list[torch.Tensor]
) -> tuple[list[torch.Tensor], float, list[dict]]:
    goldens = []
    evidence = []
    worst = 0.0
    for ids in fixtures:
        baseline, base_layers = model.forward(ids, "baseline", True)
        candidate, candidate_layers = model.forward(ids, "candidate", True)
        for base_layer, candidate_layer in zip(base_layers, candidate_layers):
            for expected, actual in zip(base_layer, candidate_layer):
                error = (actual.float() - expected.float()).abs().max().item()
                worst = max(worst, error)
                torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.03)
                assert error <= 0.03, error
        error = (candidate.float() - baseline.float()).abs().max().item()
        worst = max(worst, error)
        torch.testing.assert_close(candidate, baseline, atol=0.015, rtol=0.03)
        assert error <= 0.03, error
        goldens.append(baseline.cpu())
        evidence.append(
            {
                "ids": ids,
                "baseline": {
                    "logits": baseline.cpu(),
                    "layers": [tuple(t.cpu() for t in layer) for layer in base_layers],
                },
                "candidate": {
                    "logits": candidate.cpu(),
                    "layers": [
                        tuple(t.cpu() for t in layer) for layer in candidate_layers
                    ],
                },
            }
        )
    return goldens, worst, evidence


def burst(model: StackedModel, fixtures: list[torch.Tensor], groups: int, arm: str):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter_ns()
    outputs = []
    finished_ms = []
    for group in range(groups):
        logits, _ = model.forward(fixtures[group % len(fixtures)], arm)
        outputs.append(logits.cpu())
        finished_ms.append((time.perf_counter_ns() - started) / 1e6)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return outputs, {
        "elapsed_ms": elapsed_ms,
        "responses_per_second": groups * fixtures[0].shape[0] * 1000 / elapsed_ms,
        "last_response_ms": finished_ms[-1],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=("lstm", "slstm"), required=True)
    parser.add_argument(
        "--baseline", choices=("cudnn", "flashrnn1_triton"), required=True
    )
    parser.add_argument("--batch", type=int, choices=(1, 4, 16, 32, 64), required=True)
    parser.add_argument(
        "--concurrency", type=int, choices=(1, 8, 32, 64, 128), required=True
    )
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cell == "slstm" and args.baseline == "cudnn":
        parser.error("cuDNN has no sLSTM cell")
    if args.batch < 1 or args.concurrency < args.batch or args.concurrency % args.batch:
        parser.error("concurrency must be a positive multiple of batch")
    if args.steps < 1 or args.layers < 2 or args.width != 64 or args.blocks < 20:
        parser.error("requires T>=1, at least two layers, D64 and 20 paired blocks")
    qualification_path = args.output.with_suffix(".qualification.pt")
    meta_path = args.output.with_suffix(".meta.json")
    if any(path.exists() for path in (args.output, meta_path, qualification_path)):
        parser.error("output or qualification file already exists")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    groups = args.concurrency // args.batch
    generator = torch.Generator().manual_seed(20261006 + args.batch)
    fixtures = [
        torch.randint(0, 512, (args.batch, args.steps), generator=generator)
        for _ in range(groups)
    ]
    props = torch.cuda.get_device_properties(0)
    if str(props.uuid).removeprefix("GPU-") != args.expected_gpu_uuid.removeprefix(
        "GPU-"
    ):
        raise ValueError("GPU UUID differs from the admitted device")
    model = StackedModel(args.cell, args.baseline, args.layers, args.width)
    source_files = [
        Path(__file__),
        Path(TorchLayer.forward.__code__.co_filename),
        Path(persistent.__code__.co_filename),
    ]
    if args.baseline == "flashrnn1_triton":
        source_files.extend(
            (
                Path(upstream.__code__.co_filename),
                Path(__file__).parents[2]
                / "flashrnn/flashrnn/triton_fused"
                / f"{args.cell}_fw.py",
            )
        )
    metadata = {
        "status": "RUNNING",
        "started": time.time(),
        "pid": os.getpid(),
        "scope": "SYNTHETIC_COMPLETE_STACKED_SEQUENCE_MODEL_E2E",
        "model": {
            "seed": 20261006,
            "layers": args.layers,
            "width": args.width,
            "vocab": 512,
            "cell": args.cell,
        },
        "workload": {
            "batch": args.batch,
            "concurrency": args.concurrency,
            "sequence_length": args.steps,
            "groups": groups,
            "fixture_seed": 20261006 + args.batch,
        },
        "baseline": args.baseline,
        "candidate": "flashrnn2_triton_persistent",
        "dtype": "bfloat16",
        "device_uuid": str(props.uuid),
        "device_name": props.name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": {path.name: source_hash(path) for path in source_files},
        "fixture_sha256": [
            hashlib.sha256(ids.numpy().tobytes()).hexdigest() for ids in fixtures
        ],
        "timed": "Batched IDs H2D, embedding, every recurrent layer, final readout, logits D2H and queued groups",
        "excluded": "Model construction, JIT, warmup, qualification, fixture generation and JSON I/O",
        "arrival_policy": "All C requests at time zero; one worker runs B-sized groups in order",
        "statistics": "Within-process paired AB/BA blocks; no independent-start interval",
        "memory_scope": "Both arms' model weights remain resident; peak is process total",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.inference_mode():
        goldens, worst, evidence = qualify(model, fixtures)
        torch.save({"cases": evidence, "max_abs": worst}, qualification_path)
        metadata.update(
            qualification_max_abs=worst,
            qualification_sha256=source_hash(qualification_path),
            qualification_bytes=qualification_path.stat().st_size,
        )
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if args.baseline == "cudnn":
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                model.forward(fixtures[0], "baseline")
                torch.cuda.synchronize()
            operators = sorted(
                event.key
                for event in profile.key_averages()
                if "cudnn" in event.key.lower()
            )
            if not operators:
                raise RuntimeError("Torch baseline did not dispatch cuDNN")
            metadata["cudnn_operators"] = operators
        for _ in range(3):
            burst(model, fixtures, groups, "baseline")
            burst(model, fixtures, groups, "candidate")
        with args.output.open("x") as handle:
            for block in range(args.blocks):
                order = (
                    ("baseline", "candidate")
                    if block % 2 == 0
                    else ("candidate", "baseline")
                )
                timings = {}
                for arm in order:
                    outputs, timings[arm] = burst(model, fixtures, groups, arm)
                    for group, output in enumerate(outputs):
                        torch.testing.assert_close(
                            output, goldens[group % len(goldens)], atol=0.015, rtol=0.03
                        )
                row = {
                    "block": block,
                    "order": order,
                    "baseline": timings["baseline"],
                    "candidate": timings["candidate"],
                    "speedup": timings["baseline"]["elapsed_ms"]
                    / timings["candidate"]["elapsed_ms"],
                    "status": "PASS",
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(
                    json.dumps({"block": block, "speedup": row["speedup"]}), flush=True
                )
    metadata.update(status="PASS", finished=time.time())
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
