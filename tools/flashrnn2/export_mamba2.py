"""Static full-checkpoint Mamba2 export for an existing Torch-only runtime."""

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from mamba2_checkpoint import load_mamba2
from torch import nn
from transformers import Mamba2ForCausalLM
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

Output = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Mamba2Execution(nn.Module):
    def __init__(self, model: Mamba2ForCausalLM) -> None:
        super().__init__()
        self.model = model

    def _run(
        self, ids: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor, prefill: bool
    ) -> Output:
        cache = Mamba2Cache(
            self.model.config, ids.shape[0], self.model.dtype, ids.device
        )
        cache.conv_states = conv
        cache.ssm_states = ssm
        position = (
            torch.arange(self.model.config.conv_kernel, device=ids.device)
            if prefill
            else torch.full((1,), self.model.config.conv_kernel, device=ids.device)
        )
        output = self.model(
            ids, cache_params=cache, cache_position=position, use_cache=True
        )
        return output.logits, cache.conv_states, cache.ssm_states

    def prefill(
        self, ids: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor
    ) -> Output:
        return self._run(ids, conv, ssm, True)

    def decode(
        self, ids: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor
    ) -> Output:
        return self._run(ids, conv, ssm, False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=5)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    torch.manual_seed(20261012)
    started = time.time()
    manifest = json.loads((args.model / "verified-manifest.json").read_text())
    for entry in manifest["files"]:
        path = args.model / entry["file"]
        assert path.stat().st_size == entry["size"]
        if path.suffix == ".bin":
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        else:
            data = path.read_bytes()
            checksum = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert checksum == entry["checksum"]
    model = load_mamba2(args.model).eval()
    runner = Mamba2Execution(model).eval()

    def inputs(length: int) -> Output:
        cache = Mamba2Cache(model.config, args.batch, torch.float32, "cpu")
        return (
            torch.randint(model.config.vocab_size, (args.batch, length)),
            cache.conv_states,
            cache.ssm_states,
        )

    records = []
    with torch.inference_mode():
        traced = torch.jit.trace_module(
            runner,
            {"prefill": inputs(args.prompt_length), "decode": inputs(1)},
            check_trace=False,
        )
        torch.manual_seed(20261013)
        for sample in range(3):
            prefix = inputs(args.prompt_length)
            eager = runner.prefill(*(x.clone() for x in prefix))
            actual = traced.prefill(*(x.clone() for x in prefix))
            prefill_errors = [(a - b).abs().max().item() for a, b in zip(actual, eager)]
            for a, b in zip(actual, eager):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            ids = eager[0][:, -1].argmax(-1, keepdim=True)
            eager_step = runner.decode(ids, eager[1].clone(), eager[2].clone())
            actual_step = traced.decode(ids, actual[1].clone(), actual[2].clone())
            decode_errors = [
                (a - b).abs().max().item() for a, b in zip(actual_step, eager_step)
            ]
            for a, b in zip(actual_step, eager_step):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            records.append(
                {
                    "sample": sample,
                    "prefill_max_errors": prefill_errors,
                    "decode_max_errors": decode_errors,
                    "status": "PASS",
                }
            )
            print(json.dumps(records[-1]), flush=True)
        traced.save(str(args.output))
    with args.output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "status": "CPU_PARITY_PASS_CUDA_UNTESTED",
        "model_family": "mamba2",
        "started": started,
        "finished": time.time(),
        "torch": torch.__version__,
        "batch": args.batch,
        "prompt_length": args.prompt_length,
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_layers": len(model.backbone.layers),
        "checkpoint_manifest": manifest,
        "artifact_sha256": digest,
        "artifact_bytes": args.output.stat().st_size,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(inspect.getfile(load_mamba2)),
                Path(inspect.getfile(Mamba2ForCausalLM)),
            )
        },
        "dtype": "float32",
        "path": "Transformers built-in torch_forward; not mamba_ssm CUDA",
        "input_state_policy": "Request-owned mutable convolution and SSM cache tensors",
        "device_literal_portability": "NOT_QUALIFIED",
        "records": records,
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
