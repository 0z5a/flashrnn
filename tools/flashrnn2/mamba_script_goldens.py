"""Generate independent native-model oracles for a complete TorchScript model."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from export_mamba import MambaExecution
from transformers import MambaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261006)
    model, loading = MambaForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    assert all(not values for values in loading.values()), loading
    runner = MambaExecution(model.eval()).eval()
    rows = []
    with torch.inference_mode():
        for index in range(3):
            ids = torch.randint(0, model.config.vocab_size, (1, 5))
            if index == 0:
                ids = torch.tensor([[510, 5347, 273, 6181, 310]])
            conv = torch.zeros(24, 1, 1536, 4)
            ssm = torch.zeros(24, 1, 1536, 16)
            prefill = runner.prefill(ids, conv, ssm)
            next_ids = prefill[0][:, -1].argmax(-1, keepdim=True)
            decode = runner.decode(next_ids, prefill[1].clone(), prefill[2].clone())
            rows.append(
                {
                    "input_ids": ids,
                    "prefill": prefill,
                    "next_ids": next_ids,
                    "decode": decode,
                }
            )
    torch.save(rows, args.output)
    with args.output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    meta = {
        "source": "native Transformers MambaForCausalLM",
        "scope": "FULL_MODEL_PREFILL_AND_ONE_DECODE_ORACLES",
        "artifact_sha256": digest,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(
            Path(MambaExecution._run.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "seed": 20261006,
        "batch": 1,
        "prompt_length": 5,
        "cases": len(rows),
        "logits_contract": {"atol": 0.001, "rtol": 0.001},
        "cache_contract": {"atol": 0.00001, "rtol": 0.00001},
        "tolerance_source": "Logits use the Transformers v4.54.0 pretrained fixture tolerance. Cache tolerance matches the CPU trace gate and is frozen before CUDA execution. This does not revise the stricter cached/full-prefix experiment.",
        "checkpoint": json.loads((args.model / "verified-manifest.json").read_text()),
        "cuda_executed": False,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "NATIVE_CPU_ORACLES_COMPLETE",
                "cases": len(rows),
                "artifact_sha256": digest,
            }
        )
    )


if __name__ == "__main__":
    main()
