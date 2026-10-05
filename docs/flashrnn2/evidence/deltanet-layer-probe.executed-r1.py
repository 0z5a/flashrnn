"""Localize a retained DeltaNet failure without changing model arithmetic."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "source/tools/flashrnn2"))
from deltanet_torch_reference import DeltaNetReference
from gla_reference_gate import PROMPTS, compare

torch.set_num_threads(1)
prefix = root / "evidence/deltanet-layer-probe-r1"
assert not prefix.with_suffix(".json").exists()
started = time.time()
source_paths = [Path(__file__)] + [root / "source/tools/flashrnn2" / f for f in (
    "gla_torch_reference.py", "gla_reference_gate.py", "deltanet_torch_reference.py"
)]
source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
model = DeltaNetReference(root / "models/deltanet-1.3b", root / "evidence/models/deltanet-sources")
tokenizer = Tokenizer.from_file(str(root / "models/deltanet-1.3b/tokenizer.json"))
ids = torch.tensor([tokenizer.encode(PROMPTS[8]).ids[:5]])
full_ids = ids.clone()
captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def capture(name: str):
    def hook(module: torch.nn.Module, args: tuple, kwargs: dict, output: torch.Tensor | tuple) -> None:
        x = args[0] if args else kwargs["x"]
        value = output[0] if isinstance(output, tuple) else output
        assert isinstance(x, torch.Tensor) and isinstance(value, torch.Tensor)
        captured[name] = (x[:, -1].clone(), value[:, -1].clone())
    return hook


handles = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) or type(module).__name__ in (
        "TorchRMSNorm", "ShortConvolution", "SwiGLULinear"
    ):
        handles.append(module.register_forward_hook(capture(name), with_kwargs=True))
rows, witnesses = [], {}
original = [json.loads(line) for line in (root / "evidence/deltanet-reference-r1.jsonl").read_text().splitlines()]
original = [row for row in original if row["batch"] == 1 and row["case"] == 2]
with torch.inference_mode():
    logits, cache = model(ids)
    for step in range(4):
        cached = captured.copy()
        captured.clear()
        reference, reference_cache = model(full_ids)
        full = captured.copy()
        captured.clear()
        assert cached.keys() == full.keys()
        modules = []
        for name, (left_input, left_output) in cached.items():
            right_input, right_output = full[name]
            modules.append({"name": name, "input": compare(left_input, right_input, 1e-5), "output": compare(left_output, right_output, 1e-5)})
        checks = {"logits": compare(logits[:, -1], reference[:, -1], 1e-3)}
        state_rows, conv_rows = [], []
        for layer, (a, b) in enumerate(zip(cache.states, reference_cache.states, strict=True)):
            state_rows.append({"layer": layer, **compare(a["recurrent_state"], b["recurrent_state"], 1e-5)})
            for channel, (x, y) in enumerate(zip(a["conv_state"], b["conv_state"], strict=True)):
                conv_rows.append({"layer": layer, "channel": channel, **compare(x, y, 1e-5)})
        for name, values in (("state", state_rows), ("convolution", conv_rows)):
            checks[name] = {"pass": all(x["pass"] for x in values), "max_abs": max(x["max_abs"] for x in values), "worst_normalized": max(x["worst_normalized"] for x in values), "failed_elements": sum(x["failed_elements"] for x in values)}
        assert checks == original[step]["checks"], (step, checks, original[step]["checks"])
        if step == 1:
            first_difference = next(item for item in modules if item["output"]["max_abs"] > 0)
            name = first_difference["name"]
            witnesses["first_different_module"] = {"name": name, "cached": cached[name], "full_prefix": full[name]}
            layer = next(x["layer"] for x in conv_rows if not x["pass"])
            witnesses["first_failed_convolution_layer"] = {"layer": layer, "cached": cache.states[layer]["conv_state"], "full_prefix": reference_cache.states[layer]["conv_state"]}
            # Later decode mutates convolution caches in place.
            for arm in ("cached", "full_prefix"):
                witnesses["first_failed_convolution_layer"][arm] = tuple(t.clone() for t in witnesses["first_failed_convolution_layer"][arm])
        next_ids = logits[:, -1].argmax(-1, keepdim=True)
        assert torch.equal(next_ids, reference[:, -1].argmax(-1, keepdim=True))
        row = {"step": step, "checks": checks, "modules": modules, "recurrent_layers": state_rows, "convolution_layers": conv_rows}
        rows.append(row)
        print(json.dumps({"step": step, "modules": len(modules), "checks": checks}), flush=True)
        if step < 3:
            full_ids = torch.cat((full_ids, next_ids), dim=1)
            logits, cache = model(next_ids, cache)
for handle in handles:
    handle.remove()
torch.save(witnesses, prefix.with_suffix(".pt"))
with prefix.with_suffix(".pt").open("rb") as stream:
    witness_hash = hashlib.file_digest(stream, "sha256").hexdigest()
record = {"scope": "FULL_DELTANET_B1_CASE2_P5_G4_LAYER_LOCALIZATION_ONLY", "pid": os.getpid(), "started": started, "finished": time.time(), "source_sha256": source_hashes, "original_checks_exact": True, "model_arithmetic_modified": False, "witness_sha256": witness_hash, "rows": rows}
prefix.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"status": "DIAGNOSTIC_COLLECTED", "first_difference": witnesses["first_different_module"]["name"], "first_failed_convolution_layer": witnesses["first_failed_convolution_layer"]["layer"]}), flush=True)
