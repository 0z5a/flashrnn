"""Full-checkpoint RWKV7 oracles from the pinned official scalar RNN forward."""

import argparse
import ast
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch
from pyarrow import parquet
from rwkv7_native import NativeRWKV7

DATASET_SHA = "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--converter-source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=5)
    parser.add_argument("--generated-tokens", type=int, default=4)
    parser.add_argument("--cases", type=int, default=3)
    args = parser.parse_args()
    assert min(args.prompt_length, args.generated_tokens, args.cases) > 0
    assert not args.output.exists()
    assert digest(args.dataset) == DATASET_SHA
    torch.set_num_threads(1)
    started = time.time()
    model = NativeRWKV7(args.model, args.native_source, args.converter_source)
    vocab = args.model / "rwkv_vocab_v20230424.txt"
    # The pinned tokenizer uses eval; establish that every vocabulary entry is a literal.
    for line in vocab.read_text().splitlines():
        item = ast.literal_eval(line[line.index(" ") : line.rindex(" ")])
        assert isinstance(item, (str, bytes))
    tokenizer_source = args.model / "hf_rwkv_tokenizer.py"
    spec = importlib.util.spec_from_file_location(
        "rwkv7_pinned_tokenizer", tokenizer_source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.RwkvTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    for row, text in enumerate(
        parquet.read_table(args.dataset, columns=["text"])["text"].to_pylist()
    ):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= args.prompt_length:
            prompts.append((row, ids[: args.prompt_length]))
        if len(prompts) == args.cases:
            break
    assert len(prompts) == args.cases
    records, cases = [], []
    with torch.inference_mode():
        for index, (row, ids) in enumerate(prompts):
            state = model.zero_state()
            for token in ids:
                logits, state = model.step(token, state)
            prefill = [value.clone() for value in state]
            emitted, outputs = [], []
            for step in range(args.generated_tokens):
                assert torch.isfinite(logits).all()
                emitted.append(int(logits.argmax()))
                outputs.append(logits.clone())
                if step + 1 < args.generated_tokens:
                    logits, state = model.step(emitted[-1], state)
            assert all(torch.isfinite(value).all() for value in state)
            cases.append(
                {
                    "input_ids": torch.tensor(ids),
                    "generated_ids": torch.tensor(emitted),
                    "logits": torch.stack(outputs),
                    "prefill_cache": prefill,
                    "final_cache": [value.clone() for value in state],
                }
            )
            records.append(
                {
                    "case": index,
                    "dataset_row": row,
                    "input_ids": ids,
                    "generated_ids": emitted,
                    "decoded": tokenizer.decode(emitted),
                    "state_tensors": len(state),
                    "status": "FINITE_NATIVE_EXECUTION",
                }
            )
            print(json.dumps(records[-1]), flush=True)
    torch.save(cases, args.output)
    metadata = {
        "status": "NATIVE_CPU_ORACLES_COMPLETE_ACCELERATOR_PARITY_UNTESTED",
        "started": started,
        "finished": time.time(),
        "torch": torch.__version__,
        "checkpoint_tensors": model.checkpoint_tensors,
        "checkpoint_elements": model.checkpoint_elements,
        "checkpoint_manifest": model.manifest,
        "model_layers": model.n_layer,
        "checkpoint_dtype": "bfloat16",
        "execution_dtype": "float32",
        "actual_native_step_batch": 1,
        "prompt_length": args.prompt_length,
        "generated_tokens": args.generated_tokens,
        "generation_contract": "Fixed count, no EOS stop; prefill emits the first token, then G-1 recurrent calls. All 72 layer-state tensors retained at prefill/final.",
        "dataset": {
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "sha256": DATASET_SHA,
            "selection": "First cases rows with at least P native tokenizer IDs; no added special tokens",
        },
        "source_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path(NativeRWKV7.step.__code__.co_filename),
                args.native_source,
                args.converter_source,
                tokenizer_source,
            )
        },
        "artifact_sha256": digest(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "mapping": model.mapping,
        "records": records,
        "performance_claim": False,
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
