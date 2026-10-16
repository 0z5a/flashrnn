"""Paired complete stacked-RNN sequence inference on a fixed synthetic model.

One queued cohort has C requests, served by one worker in B-sized groups. Timing
includes ID transfer, embedding, every recurrent layer, readout and host delivery.
"""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch

from flashrnn.flashrnn2.reference import SIZES
from flashrnn.flashrnn2.torch_layer import TorchLayer
from flashrnn.flashrnn2.triton_basic import recurrence as basic
from flashrnn.flashrnn2.triton_persistent import recurrence as persistent
from flashrnn.flashrnn2.upstream_triton import recurrence as upstream


class StackedModel:
    def __init__(
        self,
        cell: str,
        baseline: str,
        layers: int,
        width: int,
        dtype: torch.dtype = torch.bfloat16,
        haste_module=None,
    ) -> None:
        generator = torch.Generator().manual_seed(20261006)
        self.cell, self.baseline, self.width, self.dtype = cell, baseline, width, dtype
        self.embedding = (torch.randn(512, width, generator=generator) * 0.1).to(
            device="cuda", dtype=dtype
        )
        self.readout = (torch.randn(width, 512, generator=generator) * 0.1).to(
            device="cuda", dtype=dtype
        )
        self.weights = []
        self.modules = []
        self.haste_module = haste_module
        self.haste_weights = []
        gates, bias_gates, _ = SIZES[cell]
        for _ in range(layers):
            w = (
                torch.randn(gates, 1, width, width, generator=generator)
                * (0.2 / width**0.5)
            ).to(device="cuda", dtype=dtype)
            r = (
                torch.randn(gates, 1, width, width, generator=generator)
                * (0.2 / width**0.5)
            ).to(device="cuda", dtype=dtype)
            b = (torch.randn(bias_gates, 1, width, generator=generator) * 0.02).to(
                device="cuda", dtype=dtype
            )
            self.weights.append((w, r, b))
            self.modules.append(
                TorchLayer(w, r, b, cell).eval() if baseline == "cudnn" else None
            )
            self.haste_weights.append(
                (
                    w[[1, 0, 2], 0].reshape(3 * width, width).T.contiguous(),
                    r[[2, 1, 0], 0].reshape(3 * width, width).T.contiguous(),
                    torch.cat((b[2, 0], b[1, 0], b[3, 0])).contiguous(),
                    torch.cat((b[2, 0] * 0, b[1, 0] * 0, b[0, 0])).contiguous(),
                )
                if baseline == "haste"
                else None
            )

    def forward(self, ids: torch.Tensor, arm: str, trace: bool = False):
        batch = ids.shape[0]
        x = self.embedding[ids.to("cuda")]
        states = SIZES[self.cell][2]
        layers = []
        for (w, r, bias), module, haste_weights in zip(
            self.weights, self.modules, self.haste_weights
        ):
            initial = torch.zeros(
                (states, batch, 1, 1, self.width), device="cuda", dtype=self.dtype
            )
            if self.cell == "slstm":
                initial[2].fill_(1)
            if arm == "baseline" and module is not None:
                hidden, final = module(x, initial)
            elif arm == "baseline" and haste_weights is not None:
                h, _ = self.haste_module.gru_forward(
                    False,
                    0.0,
                    x.transpose(0, 1).contiguous(),
                    initial[0, :, 0, 0].contiguous(),
                    *haste_weights,
                    x.new_empty(0),
                )
                hidden = h[1:].transpose(0, 1)[:, :, None]
                final = h[-1][None, :, None, None]
            else:
                wx = torch.einsum("bti,ghdi->btghd", x, w)
                if arm == "baseline":
                    kernel = upstream
                elif self.cell in ("lstm", "slstm"):
                    kernel = persistent
                else:
                    kernel = basic
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
    parser.add_argument(
        "--cell", choices=("lstm", "slstm", "gru", "elman"), required=True
    )
    parser.add_argument(
        "--baseline", choices=("cudnn", "flashrnn1_triton", "haste"), required=True
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
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--require-packed-cudnn", action="store_true")
    parser.add_argument("--haste-module-root", type=Path)
    parser.add_argument("--expected-haste-so-sha")
    parser.add_argument("--haste-audit", type=Path)
    parser.add_argument("--haste-audit-raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cell == "slstm" and args.baseline == "cudnn":
        parser.error("cuDNN has no sLSTM cell")
    if (
        args.cell in ("gru", "elman")
        and args.baseline != "cudnn"
        and (args.cell, args.baseline) != ("gru", "haste")
    ):
        parser.error("GRU and Elman require a matched cuDNN or Haste baseline")
    if args.baseline == "haste" and (
        args.cell != "gru"
        or args.dtype != "fp16"
        or args.haste_module_root is None
        or args.expected_haste_so_sha is None
        or args.haste_audit is None
        or args.haste_audit_raw is None
    ):
        parser.error("Haste requires GRU FP16 and a qualified native extension")
    if args.batch < 1 or args.concurrency < args.batch or args.concurrency % args.batch:
        parser.error("concurrency must be a positive multiple of batch")
    if args.steps < 1 or args.layers < 2 or args.width != 64 or args.blocks < 20:
        parser.error("requires T>=1, at least two layers, D64 and 20 paired blocks")
    if args.require_packed_cudnn and args.baseline != "cudnn":
        parser.error("packed-weight gate applies to the cuDNN baseline")
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
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    if "," in visible or torch.cuda.device_count() != 1:
        raise ValueError("expected exactly one assigned GPU")
    device_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    expected_uuids = args.expected_gpu_uuid.split(",")
    if device_uuid not in expected_uuids:
        raise ValueError("GPU UUID differs from the admitted device")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    haste_module = None
    haste_qualification = None
    if args.baseline == "haste":
        audit = json.loads(args.haste_audit.read_text())
        if (
            audit["status"] != "PASS"
            or audit["tensor_pairs"] != 20
            or len(audit["results"]) != 20
            or any(row["failed"] != 0 for row in audit["results"])
            or source_hash(args.haste_audit_raw) != audit["raw_sha256"]
            or audit["extension_sha256"] != args.expected_haste_so_sha
        ):
            raise ValueError("Haste native numerical qualification is incomplete")
        haste_qualification = {
            "audit_sha256": source_hash(args.haste_audit),
            "raw_sha256": audit["raw_sha256"],
            "extension_sha256": args.expected_haste_so_sha,
        }
        sys.path.insert(0, str(args.haste_module_root))
        haste_module = importlib.import_module("haste_lstm_gru_native")
        if source_hash(Path(haste_module.__file__)) != args.expected_haste_so_sha:
            raise ValueError("Haste native extension differs from qualified build")
    model = StackedModel(
        args.cell, args.baseline, args.layers, args.width, dtype, haste_module
    )
    if args.baseline == "cudnn":
        assert all(
            module is not None
            and not module.training
            and all(not layer.training for layer in module.layers)
            for module in model.modules
        )
    cudnn_weights = [
        layer._flat_weights
        for module in model.modules
        if module is not None
        for layer in module.layers
    ]
    cudnn_storage_counts = [
        len({weight.untyped_storage().data_ptr() for weight in weights})
        for weights in cudnn_weights
    ]
    cudnn_weights_acceptable = [
        all(torch.backends.cudnn.is_acceptable(weight) for weight in weights)
        for weights in cudnn_weights
    ]
    if args.require_packed_cudnn and (
        len(cudnn_storage_counts) != args.layers
        or not all(count == 1 for count in cudnn_storage_counts)
        or not all(cudnn_weights_acceptable)
    ):
        raise RuntimeError("cuDNN weights are not packed on this runtime")
    candidate = persistent if args.cell in ("lstm", "slstm") else basic
    source_files = [
        Path(__file__),
        Path(TorchLayer.forward.__code__.co_filename),
        Path(candidate.__code__.co_filename),
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
    if haste_module is not None:
        source_files.append(Path(haste_module.__file__))
    metadata = {
        "evidence_version": 2,
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
        "candidate": (
            "flashrnn2_triton_persistent"
            if args.cell in ("lstm", "slstm")
            else "flashrnn2_triton_basic"
        ),
        "haste_qualification": haste_qualification,
        "cudnn_training_flags": [
            {
                "module": module.training,
                "heads": [layer.training for layer in module.layers],
            }
            for module in model.modules
            if module is not None
        ],
        "dtype": str(dtype),
        "cudnn_weight_storage_counts": cudnn_storage_counts,
        "cudnn_weights_acceptable": cudnn_weights_acceptable,
        "packed_cudnn_required": args.require_packed_cudnn,
        "device_uuid": device_uuid,
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
        if args.require_packed_cudnn:
            warnings.filterwarnings(
                "error",
                message="RNN module weights are not part of single contiguous chunk of memory",
            )
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
    metadata["paired_blocks"] = args.blocks
    metadata["raw_sha256"] = source_hash(args.output)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
