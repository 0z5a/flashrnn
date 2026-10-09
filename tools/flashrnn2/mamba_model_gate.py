"""Full-checkpoint cached/uncached Mamba qualification in an existing runtime."""

import argparse
import hashlib
import inspect
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, MambaForCausalLM

PROMPTS = (
    "The capital of France is",
    "Recurrent neural networks process a sequence by",
    "Write a short explanation of the water cycle:",
    "In mathematics, the Fibonacci sequence begins with",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--family", choices=("mamba", "mamba2"), default="mamba")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--official-fixture-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    started = time.time()
    manifest = json.loads((args.model / "verified-manifest.json").read_text())
    for entry in manifest["files"]:
        path = args.model / entry["file"]
        assert path.stat().st_size == entry["size"], path
        if path.suffix in {".safetensors", ".bin"}:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        else:
            payload = path.read_bytes()
            digest = hashlib.sha1(
                f"blob {len(payload)}\0".encode() + payload
            ).hexdigest()
        assert digest == entry["checksum"], path
    adapter_hash = None
    if args.family == "mamba2":
        from mamba2_checkpoint import load_mamba2

        assert not args.official_fixture_only, "the fixed upstream fixture is Mamba1"
        model = load_mamba2(args.model)
        loading = {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }
        adapter_hash = hashlib.sha256(
            Path(inspect.getfile(load_mamba2)).read_bytes()
        ).hexdigest()
    else:
        model, loading = MambaForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float32,
            output_loading_info=True,
        )
    assert not loading["missing_keys"], loading
    assert not loading["unexpected_keys"], loading
    assert not loading["mismatched_keys"], loading
    assert not loading["error_msgs"], loading
    model.eval()
    assert len(model.backbone.layers) == 24 and model.config.hidden_size == 768
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    source = Path(inspect.getfile(type(model)))
    metadata = {
        "scope": f"FULL_{args.family.upper()}130M_CPU_CACHE_CORRECTNESS",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "model_class": type(model).__name__,
        "model_layers": len(model.backbone.layers),
        "parameters": sum(p.numel() for p in model.parameters()),
        "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_manifest": manifest,
        "checkpoint_adapter_sha256": adapter_hash,
        "tokenizer_sha256": {
            name: hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest()
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        "dtype": "float32",
        "device": "cpu",
        "path": "Transformers built-in "
        + ("torch_forward" if args.family == "mamba2" else "slow_forward"),
        "atol": 0.0001,
        "rtol": 0.00001,
        "started": started,
        "status": "RUNNING",
        "loading_info": loading,
        "speedup": None,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.official_fixture_only:
        fixture_path = (
            Path(__file__).parent / "fixtures/mamba130m-transformers-v4540.json"
        )
        fixture = json.loads(fixture_path.read_text())
        tokens = tokenizer(fixture["prompt"], return_tensors="pt").input_ids
        with torch.inference_mode():
            generated = model.generate(
                tokens,
                do_sample=False,
                use_cache=True,
                max_new_tokens=fixture["max_new_tokens"],
            )
            text = tokenizer.decode(generated[0])
            assert text == fixture["expected_text"], text
            logits = model(tokens).logits[0, 0, :40]
            target = torch.tensor(fixture["expected_logits_first_position"])
            torch.testing.assert_close(
                logits, target, atol=fixture["atol"], rtol=fixture["rtol"]
            )
        metadata.update(
            scope="FULL_MAMBA130M_UPSTREAM_FIXTURE_ONLY",
            status="PASS",
            finished=time.time(),
            fixture=fixture,
            actual_text=text,
            max_logit_error=(logits - target).abs().max().item(),
        )
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
        return
    failed_batches = 0
    with args.output.open("x") as handle, torch.inference_mode():
        for batch in args.batches:
            # Equal-length rows isolate cache semantics from padding conventions.
            tokens = tokenizer(PROMPTS[0], return_tensors="pt").input_ids
            tokens = tokens.repeat(batch, 1)
            for row in range(1, batch):
                alternate = tokenizer.encode(PROMPTS[row % len(PROMPTS)])
                tokens[row, -1] = alternate[-1]
            prefix = tokens.clone()
            output = model(prefix, use_cache=True)
            cache = output.cache_params
            cached_logits = output.logits[:, -1]
            max_error = 0.0
            generated = []
            steps = []
            for step in range(args.tokens):
                reference = model(prefix, use_cache=False).logits[:, -1]
                error = (cached_logits - reference).abs().max().item()
                max_error = max(max_error, error)
                close = torch.isclose(
                    cached_logits,
                    reference,
                    atol=metadata["atol"],
                    rtol=metadata["rtol"],
                )
                next_ids = cached_logits.argmax(-1, keepdim=True)
                token_match = torch.equal(next_ids, reference.argmax(-1, keepdim=True))
                steps.append(
                    {
                        "step": step,
                        "max_logit_error": error,
                        "mismatched_logits": (~close).sum().item(),
                        "greedy_token_match": token_match,
                    }
                )
                if not close.all().item():
                    torch.save(
                        {"cached": cached_logits, "uncached": reference},
                        args.output.with_name(
                            f"{args.output.stem}-b{batch}-t{step}.pt"
                        ),
                    )
                generated.append(next_ids[:, 0].tolist())
                prefix = torch.cat((prefix, next_ids), dim=1)
                if step + 1 < args.tokens:
                    output = model(
                        next_ids,
                        use_cache=True,
                        cache_params=cache,
                        cache_position=torch.tensor([model.config.conv_kernel + step]),
                    )
                    cached_logits = output.logits[:, -1]
            passed = all(
                row["mismatched_logits"] == 0 and row["greedy_token_match"]
                for row in steps
            )
            failed_batches += int(not passed)
            row = {
                "batch": batch,
                "prompt_tokens": tokens.shape[1],
                "generated_tokens": args.tokens,
                "input_ids": tokens.tolist(),
                "new_ids_by_step": generated,
                "texts": tokenizer.batch_decode(prefix, skip_special_tokens=True),
                "max_logit_error": max_error,
                "steps": steps,
                "greedy_token_match": all(row["greedy_token_match"] for row in steps),
                "status": "PASS" if passed else "NUMERICAL_FAILED",
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps(row), flush=True)
    metadata.update(
        status="NUMERICAL_FAILED" if failed_batches else "PASS",
        failed_batches=failed_batches,
        finished=time.time(),
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(1 if failed_batches else 0)


if __name__ == "__main__":
    main()
