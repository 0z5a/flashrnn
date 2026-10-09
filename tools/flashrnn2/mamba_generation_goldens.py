"""Native complete-model oracles for fixed WikiText prompts and greedy decode."""

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from export_mamba import MambaExecution
from export_mamba2 import Mamba2Execution
from mamba2_checkpoint import load_mamba2
from pyarrow import parquet
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_SHA256 = "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_checkpoint(directory: Path, file_names: set[str] | None = None) -> dict:
    manifest = json.loads((directory / "verified-manifest.json").read_text())
    if file_names is not None:
        manifest["files"] = [
            entry for entry in manifest["files"] if entry["file"] in file_names
        ]
        assert {entry["file"] for entry in manifest["files"]} == file_names
    for entry in manifest["files"]:
        path = directory / entry["file"]
        assert path.stat().st_size == entry["size"]
        if len(entry["checksum"]) == 64:
            checksum = digest(path)
        else:
            data = path.read_bytes()
            checksum = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert checksum == entry["checksum"]
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--family", choices=("mamba", "mamba2"), default="mamba")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--generated-tokens", type=int, default=32)
    parser.add_argument("--cases", type=int, default=3)
    args = parser.parse_args()
    assert min(*args.batches, args.prompt_length, args.generated_tokens, args.cases) > 0
    assert digest(args.dataset) == DATASET_SHA256
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint = verified_checkpoint(args.model)
    tokenizer_path = args.tokenizer or args.model
    tokenizer_checkpoint = (
        checkpoint
        if tokenizer_path == args.model
        else verified_checkpoint(
            tokenizer_path, {"tokenizer.json", "tokenizer_config.json"}
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompts = []
    if args.inputs:
        input_manifest = json.loads(args.inputs.read_text())
        assert input_manifest["dataset_sha256"] == DATASET_SHA256
        assert input_manifest["tokenizer_sha256"] == digest(
            tokenizer_path / "tokenizer.json"
        )
        assert input_manifest["prompt_length"] == args.prompt_length
        assert input_manifest["generated_tokens"] == args.generated_tokens
        group = input_manifest["batches"][str(max(args.batches))]
        assert len(group["dataset_rows"]) == len(group["input_ids"]) == args.cases
        prompts = [
            {"dataset_row": row, "ids": ids}
            for rows, case in zip(
                group["dataset_rows"], group["input_ids"], strict=True
            )
            for row, ids in zip(rows, case, strict=True)
        ]
    else:
        for row, text in enumerate(
            parquet.read_table(args.dataset, columns=["text"])["text"].to_pylist()
        ):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) >= args.prompt_length:
                prompts.append({"dataset_row": row, "ids": ids[: args.prompt_length]})
            if len(prompts) == max(args.batches) * args.cases:
                break
    assert len(prompts) == max(args.batches) * args.cases
    if args.family == "mamba2":
        model = load_mamba2(args.model).eval()
        runner = Mamba2Execution(model).eval()
    else:
        model, loading = MambaForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float32,
            output_loading_info=True,
        )
        assert all(not values for values in loading.values()), loading
        runner = MambaExecution(model.eval()).eval()
    config = model.config
    with torch.inference_mode():
        for batch in args.batches:
            started = time.time()
            cases = []
            manifest = []
            for index in range(args.cases):
                selection = prompts[
                    index * max(args.batches) : index * max(args.batches) + batch
                ]
                ids = torch.tensor([p["ids"] for p in selection])
                if args.family == "mamba2":
                    cache = Mamba2Cache(config, batch, torch.float32, "cpu")
                    conv, ssm = cache.conv_states, cache.ssm_states
                else:
                    conv = torch.zeros(
                        config.num_hidden_layers,
                        batch,
                        config.intermediate_size,
                        config.conv_kernel,
                    )
                    ssm = torch.zeros(
                        config.num_hidden_layers,
                        batch,
                        config.intermediate_size,
                        config.state_size,
                    )
                output = runner.prefill(ids, conv, ssm)
                # Decode mutates its incoming caches; preserve the prefill oracle.
                prefill_cache = (output[1].clone(), output[2].clone())
                logits = []
                tokens = []
                for step in range(args.generated_tokens):
                    logits.append(output[0][:, -1].clone())
                    tokens.append(logits[-1].argmax(-1))
                    if step + 1 < args.generated_tokens:
                        output = runner.decode(
                            tokens[-1][:, None], output[1], output[2]
                        )
                cases.append(
                    {
                        "input_ids": ids,
                        "logits": torch.stack(logits),
                        "generated_ids": torch.stack(tokens),
                        "prefill_cache": prefill_cache,
                        "final_cache": (output[1].clone(), output[2].clone()),
                    }
                )
                manifest.append(
                    {
                        "case": index,
                        "dataset_rows": [p["dataset_row"] for p in selection],
                        "input_ids_sha256": hashlib.sha256(
                            ids.numpy().tobytes()
                        ).hexdigest(),
                        "generated_ids_sha256": hashlib.sha256(
                            torch.stack(tokens).numpy().tobytes()
                        ).hexdigest(),
                    }
                )
                print(
                    json.dumps(
                        {"batch": batch, "case": index, "status": "ORACLE_COMPLETE"}
                    ),
                    flush=True,
                )
            target = (
                args.output
                / f"generation-b{batch}-p{args.prompt_length}-g{args.generated_tokens}.pt"
            )
            assert not target.exists(), target
            torch.save(cases, target)
            metadata = {
                "status": "NATIVE_CPU_ORACLES_COMPLETE",
                "started": started,
                "finished": time.time(),
                "batch": batch,
                "prompt_length": args.prompt_length,
                "generated_tokens": args.generated_tokens,
                "cases": manifest,
                "generation_contract": "Fixed token count; no EOS stop. Prefill emits token 1; G-1 decode calls emit tokens 2..G. Final caches contain prompt plus first G-1 generated tokens.",
                "logits_scope": "Complete vocabulary at each emitted token; earlier prefill positions excluded",
                "cache_scope": "All layers, prefill and final snapshots; intermediate caches not saved",
                "source": f"native Transformers {type(model).__name__} CPU FP32",
                "model_family": args.family,
                "torch": torch.__version__,
                "parameters": sum(p.numel() for p in model.parameters()),
                "layers": len(model.backbone.layers),
                "artifact_sha256": digest(target),
                "artifact_bytes": target.stat().st_size,
                "harness_sha256": digest(Path(__file__)),
                "wrapper_sha256": digest(Path(inspect.getfile(type(runner)))),
                "implementation_sha256": digest(Path(inspect.getfile(type(model)))),
                "tokenizer_sha256": digest(tokenizer_path / "tokenizer.json"),
                "tokenizer_checkpoint": tokenizer_checkpoint,
                "dataset": {
                    "id": "Salesforce/wikitext",
                    "revision": DATASET_REVISION,
                    "file": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
                    "sha256": DATASET_SHA256,
                    "selection": (
                        "Pinned input manifest; B uses the first B rows of each max(B) group"
                        if args.inputs
                        else "First cases*max(batches) rows having at least P tokens; no special tokens; first P tokens; B uses the first B rows of each max(B) group"
                    ),
                },
                "input_manifest_sha256": digest(args.inputs) if args.inputs else None,
                "checkpoint": checkpoint,
                "logits_contract": {"atol": 0.001, "rtol": 0.001},
                "cache_contract": {"atol": 0.00001, "rtol": 0.00001},
                "tolerance_source": "Same budgets as the pre-existing one-step script gate; frozen before long-generation execution, without revising cached/full-prefix failures",
                "gpu_executed": False,
            }
            target.with_suffix(".meta.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
