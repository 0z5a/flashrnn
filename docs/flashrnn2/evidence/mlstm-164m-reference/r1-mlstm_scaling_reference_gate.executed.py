"""Full official 164M mLSTM checkpoint: CPU cached versus full-prefix generation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.xlstm import configuration_xlstm, modeling_xlstm
from transformers.models.xlstm.configuration_xlstm import xLSTMConfig
from transformers.models.xlstm.modeling_xlstm import xLSTMForCausalLM

PROMPTS = (
    "The history of recurrent neural networks began with attempts to model memory.",
    "A small research team compared the predictions of several language models.",
    "The train crossed the bridge before arriving at the central station.",
    "Scientists collected samples from the river and measured the water temperature.",
    "A computer program can process a sequence one element at a time.",
    "The library opened its doors early on a quiet Monday morning.",
    "An engineer checked the measurements before designing the new circuit.",
    "The next chapter explains how the experiment was conducted and evaluated.",
    "A mountain path led the hikers through a forest toward the lake.",
    "The teacher asked the students to explain their answers in complete sentences.",
    "The performance of a system depends on both computation and communication.",
    "A musician practiced the final movement before the evening concert began.",
)
STATE_NAMES = ("cell", "normalizer", "stabilizer")
WEIGHT_SHA256 = "1dc957a158235bd26f0e1554345e3f695232ea3107dd28c4f81f71d53c1f0afc"
REVISION = "2ce74c14add515517ae6a32d3ae80cc766f62c03"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare(actual: torch.Tensor, expected: torch.Tensor, budget: float) -> dict:
    assert actual.shape == expected.shape
    difference = (actual - expected).abs()
    allowed = budget + budget * expected.abs()
    failed = (
        ~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)
    )
    return {
        "pass": not failed.any().item(),
        "max_abs": difference.max().item(),
        "worst_normalized": (difference / allowed).max().item(),
        "failed_elements": failed.sum().item(),
    }


def rename_weight(name: str) -> str:
    for old, new in (
        (".mlstm_layer.q.", ".mlstm_layer.query."),
        (".mlstm_layer.k.", ".mlstm_layer.key."),
        (".mlstm_layer.v.", ".mlstm_layer.value."),
    ):
        name = name.replace(old, new)
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-budget", type=float, default=1e-4)
    parser.add_argument(
        "--batches", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2, 4]
    )
    args = parser.parse_args()
    assert args.state_budget > 0 and len(set(args.batches)) == len(args.batches)
    assert not args.output.exists()
    assert not list(args.output.parent.glob(args.output.stem + "-b*-c*.pt"))
    torch.set_num_threads(1)
    assert transformers.__version__ == "4.54.1"

    manifest = json.loads((args.model / "verified-manifest.json").read_text())
    assert manifest["revision"] == REVISION and len(manifest["files"]) == 4
    weight = args.model / "model_0.safetensors"
    assert weight.stat().st_size == 328241392 and sha256(weight) == WEIGHT_SHA256
    tokenizer_receipt = json.loads(
        (args.output.parent / "xlstm-mlstm-164m-tokenizer-r1.json").read_text()
    )
    assert tokenizer_receipt["revision"] == "9dc507bd0939cf372a4a4f667335651d8e49dddb"
    for entry in tokenizer_receipt["files"]:
        path = args.model / entry["file"]
        assert path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assert tokenizer.bos_token_id == 0
    inputs = [
        [0, *tokenizer(text, add_special_tokens=False).input_ids[:4]]
        for text in PROMPTS
    ]
    assert all(len(ids) == 5 and max(ids) < 50304 for ids in inputs)

    config_dict = json.loads((args.model / "config.json").read_text())
    assert (
        config_dict["embedding_dim"],
        config_dict["num_heads"],
        config_dict["num_blocks"],
    ) == (768, 6, 12)
    assert config_dict["vocab_size"] == 50304 and config_dict["weight_mode"] == "single"
    config_dict["hidden_size"] = config_dict.pop("embedding_dim")
    config_dict["mode"] = "inference"
    with torch.device("meta"):
        model = xLSTMForCausalLM(xLSTMConfig(**config_dict))
    raw = load_file(weight, device="cpu")
    state = {rename_weight(name): tensor.float() for name, tensor in raw.items()}
    assert len(raw) == len(state) == len(model.state_dict()) == 183
    model.load_state_dict(state, strict=True, assign=True)
    del raw, state
    assert sum(parameter.numel() for parameter in model.parameters()) == 164110224
    model.eval()

    metadata = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": time.time(),
        "scope": "FULL_XLSTM_MLSTM_164M_CPU_CACHED_VS_FULL_PREFIX_P5_G4",
        "device": "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "weight_dtype": "torch.float32",
        "model_config_mode": "inference",
        "logits_budget": {"atol": 1e-3, "rtol": 1e-3},
        "state_budget": {"atol": args.state_budget, "rtol": args.state_budget},
        "prompts": PROMPTS,
        "input_ids": inputs,
        "tokenizer": tokenizer_receipt,
        "checkpoint": manifest,
        "checkpoint_tensors": 183,
        "parameters": 164110224,
        "config_sha256": sha256(args.model / "config.json"),
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__),
                Path(modeling_xlstm.__file__),
                Path(configuration_xlstm.__file__),
            )
        },
        "performance_claim": False,
        "native_accelerated_mlstm_qualified": False,
        "snapshots": [],
        "batches": args.batches,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with torch.inference_mode(), args.output.open("x") as handle:
        for batch in args.batches:
            for case in range(3):
                ids = torch.tensor(inputs[4 * case : 4 * case + batch])
                full_ids = ids.clone()
                output = model(input_ids=ids, use_cache=True)
                logits, cache = output.logits, output.cache_params
                cached_logits, full_logits, tokens = [], [], []
                for step in range(4):
                    expected = model(input_ids=full_ids, use_cache=True)
                    full_cache = expected.cache_params
                    left, right = logits[:, -1], expected.logits[:, -1]
                    layers = []
                    for layer in range(12):
                        a, b = cache.rnn_state[layer], full_cache.rnn_state[layer]
                        layers.append(
                            {
                                name: compare(x, y, args.state_budget)
                                for name, x, y in zip(STATE_NAMES, a, b, strict=True)
                            }
                        )
                    checks = {"logits": compare(left, right, 1e-3)}
                    for name in STATE_NAMES:
                        values = [item[name] for item in layers]
                        checks[name] = {
                            "pass": all(item["pass"] for item in values),
                            "max_abs": max(item["max_abs"] for item in values),
                            "worst_normalized": max(
                                item["worst_normalized"] for item in values
                            ),
                            "failed_elements": sum(
                                item["failed_elements"] for item in values
                            ),
                        }
                    next_ids = left.argmax(-1, keepdim=True)
                    equal = torch.equal(next_ids, right.argmax(-1, keepdim=True))
                    row = {
                        "batch": batch,
                        "case": case,
                        "step": step,
                        "checks": checks,
                        "layers": layers,
                        "token_equal": equal,
                        "pass": equal and all(item["pass"] for item in checks.values()),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                key: value
                                for key, value in row.items()
                                if key != "layers"
                            }
                        ),
                        flush=True,
                    )
                    cached_logits.append(left.clone())
                    full_logits.append(right.clone())
                    tokens.append(next_ids[:, 0].clone())
                    if step < 3:
                        full_ids = torch.cat((full_ids, next_ids), dim=1)
                        output = model(
                            input_ids=next_ids, cache_params=cache, use_cache=True
                        )
                        logits, cache = output.logits, output.cache_params
                snapshot = {
                    "batch": batch,
                    "case": case,
                    "input_ids": ids,
                    "cached_logits": torch.stack(cached_logits),
                    "full_prefix_logits": torch.stack(full_logits),
                    "generated_ids": torch.stack(tokens),
                }
                if case == 0:
                    snapshot.update(
                        cached_final_states=cache.rnn_state,
                        full_prefix_final_states=full_cache.rnn_state,
                    )
                path = args.output.with_name(f"{args.output.stem}-b{batch}-c{case}.pt")
                partial = path.with_suffix(".pt.partial")
                assert not partial.exists()
                torch.save(snapshot, partial)
                partial.replace(path)
                metadata["snapshots"].append(
                    {
                        "file": path.name,
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                        "batch": batch,
                        "case": case,
                    }
                )
                meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
                del snapshot, cache, full_cache
    passed = all(row["pass"] for row in rows)
    metadata.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        cases=3 * len(args.batches),
        batch_steps=len(rows),
        token_choices=sum(row["batch"] for row in rows),
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
