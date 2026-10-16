"""Export full Mamba prefill/decode methods without a Transformers runtime dependency.

Tracing fixes batch and prompt length. This is a separate TorchScript baseline;
CPU parity and later CUDA parity are required before performance measurement.
"""

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from torch import nn
from transformers import MambaForCausalLM
from transformers.models.mamba.modeling_mamba import MambaCache

Output = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class MambaExecution(nn.Module):
    def __init__(self, model: MambaForCausalLM) -> None:
        super().__init__()
        self.model = model

    def _run(
        self, ids: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor, prefill: bool
    ) -> Output:
        cache = MambaCache(
            self.model.config,
            max_batch_size=ids.shape[0],
            dtype=self.model.dtype,
            device=ids.device,
        )
        cache.conv_states = list(conv.unbind(0))
        cache.ssm_states = list(ssm.unbind(0))
        if prefill:
            position = torch.arange(self.model.config.conv_kernel, device=ids.device)
        else:
            position = torch.full(
                (1,), self.model.config.conv_kernel, device=ids.device
            )
        output = self.model(
            ids,
            cache_params=cache,
            cache_position=position,
            use_cache=True,
        )
        return (
            output.logits,
            torch.stack(cache.conv_states),
            torch.stack(cache.ssm_states),
        )

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
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-length", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    started = time.time()
    model, loading = MambaForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    assert all(not values for values in loading.values()), loading
    model.eval()
    runner = MambaExecution(model).eval()
    config = model.config

    def inputs(batch: int, length: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.randint(config.vocab_size, (batch, length)),
            torch.zeros(
                config.num_hidden_layers,
                batch,
                config.intermediate_size,
                config.conv_kernel,
            ),
            torch.zeros(
                config.num_hidden_layers,
                batch,
                config.intermediate_size,
                config.state_size,
            ),
        )

    samples = {
        "prefill": inputs(args.batch, args.prompt_length),
        "decode": inputs(args.batch, 1),
    }
    records = []
    with torch.inference_mode():
        traced = torch.jit.trace_module(runner, samples, check_trace=False)
        # Cache indexing specializes batch dimensions during tracing.
        # Independently sampled inputs validate each declared static shape.
        for sample in range(3):
            prefix = inputs(args.batch, args.prompt_length)
            eager = runner.prefill(*(x.clone() for x in prefix))
            actual = traced.prefill(*(x.clone() for x in prefix))
            errors = [(a - b).abs().max().item() for a, b in zip(actual, eager)]
            for a, b in zip(actual, eager):
                torch.testing.assert_close(a, b, atol=0.00001, rtol=0.00001)
            next_ids = eager[0][:, -1].argmax(-1, keepdim=True)
            eager_step = runner.decode(next_ids, eager[1].clone(), eager[2].clone())
            trace_step = traced.decode(next_ids, actual[1].clone(), actual[2].clone())
            step_errors = [
                (a - b).abs().max().item() for a, b in zip(trace_step, eager_step)
            ]
            for a, b in zip(trace_step, eager_step):
                torch.testing.assert_close(a, b, atol=0.00001, rtol=0.00001)
            records.append(
                {
                    "batch": args.batch,
                    "sample": sample,
                    "prefill_max_errors": errors,
                    "decode_max_errors": step_errors,
                    "status": "PASS",
                }
            )
            print(json.dumps(records[-1]), flush=True)
        traced.save(str(args.output))
    with args.output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "status": "CPU_PARITY_PASS_CUDA_UNTESTED",
        "started": started,
        "finished": time.time(),
        "torch": torch.__version__,
        "prompt_length": args.prompt_length,
        "batch": args.batch,
        "checkpoint_manifest": json.loads(
            (args.model / "verified-manifest.json").read_text()
        ),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(inspect.getfile(MambaForCausalLM)).read_bytes()
        ).hexdigest(),
        "artifact_sha256": digest,
        "artifact_bytes": args.output.stat().st_size,
        "model_layers": len(model.backbone.layers),
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_state_policy": "request-owned mutable convolution/SSM cache",
        "device_literal_portability": "NOT_QUALIFIED",
        "records": records,
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
