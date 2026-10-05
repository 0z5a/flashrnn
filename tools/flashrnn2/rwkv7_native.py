"""Run pinned RWKV-LM recurrent math with an audited inverse FLA key mapping.

No FLA package, compiler, sampling loop or checkpoint code is imported.
Only the official forward and two arithmetic functions are selected by AST.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

NATIVE_SHA = "a61f35716b2ef81fa1c97bfd7f67bccd78d3a8968d0570748d4631fecf885500"
CONVERTER_SHA = "b3bc860599f7612f395be4fd1d6b119e1498d4c30b2cb7c4420e80025229e05a"
State = list[torch.Tensor]


def official_name(name: str) -> tuple[str, bool]:
    direct = {
        "model.embeddings.weight": "emb.weight",
        "model.norm.weight": "ln_out.weight",
        "model.norm.bias": "ln_out.bias",
        "lm_head.weight": "head.weight",
    }
    if name in direct:
        return direct[name], False
    prefix, layers, index, component, *tail = name.split(".")
    assert (prefix, layers) == ("model", "layers")
    component = {
        "pre_norm": "ln0",
        "attn_norm": "ln1",
        "ffn_norm": "ln2",
        "attn": "att",
        "ffn": "ffn",
    }[component]
    transposed = False
    if component == "att" and tail[0].endswith("_lora"):
        kind, lora, number, field = tail
        assert lora == "lora"
        digit = {("0", "weight"): "1", ("2", "weight"): "2", ("2", "bias"): "0"}[
            (number, field)
        ]
        tail = [kind[0] + digit]
        transposed = field == "weight"
    elif component == "att":
        projection = {
            "r_proj": "receptance",
            "k_proj": "key",
            "v_proj": "value",
            "o_proj": "output",
            "g_norm": "ln_x",
        }
        tail[0] = projection.get(tail[0], tail[0])
    return ".".join(("blocks", index, component, *tail)), transposed


def native_functions(source: Path) -> dict:
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == NATIVE_SHA
    tree = ast.parse(data)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"time_mixing__", "channel_mixing__"}
    ]
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RWKV_RNN"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    forward.name = "native_forward"
    forward.decorator_list = []
    selected.append(forward)
    assert len(selected) == 3
    namespace = {"torch": torch, "F": F, "List": list}
    exec(  # noqa: S102 - hash-pinned arithmetic/mapping AST only
        compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    namespace["time_mixing"] = namespace["time_mixing__"]
    namespace["channel_mixing"] = namespace["channel_mixing__"]
    return namespace


def conversion_function(source: Path, layers: int) -> Callable[[str], tuple[str, bool]]:
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == CONVERTER_SHA
    tree = ast.parse(data)
    convert = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "convert"
    )
    translate = next(
        n
        for n in convert.body
        if isinstance(n, ast.FunctionDef) and n.name == "translate_into_fla"
    )
    namespace = {
        "re": re,
        "config": SimpleNamespace(num_hidden_layers=layers),
        "unused_names": ["blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"],
    }
    exec(  # noqa: S102 - hash-pinned arithmetic/mapping AST only
        compile(ast.Module(body=[translate], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["translate_into_fla"]


class NativeRWKV7:
    def __init__(self, directory: Path, source: Path, converter: Path) -> None:
        self.manifest = json.loads((directory / "verified-manifest.json").read_text())
        for entry in self.manifest["files"]:
            path = directory / entry["file"]
            assert path.stat().st_size == entry["size"]
            with path.open("rb") as handle:
                digest = (
                    hashlib.sha256()
                    if len(entry["checksum"]) == 64
                    else hashlib.sha1(f"blob {entry['size']}\0".encode())
                )
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            assert digest.hexdigest() == entry["checksum"], entry["file"]
        config = json.loads((directory / "config.json").read_text())
        assert config["model_type"] == "rwkv7" and config["attn"] is None
        assert (
            config["norm_first"] and config["norm_bias"] and config["norm_eps"] == 1e-5
        )
        self.n_layer, self.n_embd = config["num_hidden_layers"], config["hidden_size"]
        self.head_size = config["head_dim"]
        self.n_head = self.n_embd // self.head_size
        assert (
            self.head_size == 64 and config["value_dim"] == [self.n_embd] * self.n_layer
        )
        to_fla = conversion_function(converter, self.n_layer)
        weights = load_file(directory / "model.safetensors", device="cpu")
        self.checkpoint_elements = sum(w.numel() for w in weights.values())
        self.checkpoint_tensors = len(weights)
        self.z = {}
        self.mapping = []
        for name, weight in weights.items():
            native, transposed = official_name(name)
            assert to_fla(native) == (name, transposed), name
            assert native not in self.z
            tensor = weight.t() if transposed else weight
            self.z[native] = tensor.squeeze().float().contiguous()
            self.mapping.append(
                {
                    "source": name,
                    "native": native,
                    "transpose": transposed,
                    "shape": list(weight.shape),
                }
            )
        assert self.checkpoint_tensors == 6 + self.n_layer * 30 + (self.n_layer - 1) * 3
        assert self.z["blocks.0.att.r_k"].shape == (self.n_head, self.head_size)
        for index in range(self.n_layer):
            self.z[f"blocks.{index}.att.r_k"] = self.z[
                f"blocks.{index}.att.r_k"
            ].flatten()
        # These are the same inference-only preparations as the official constructor.
        self.z["emb.weight"] = F.layer_norm(
            self.z["emb.weight"],
            (self.n_embd,),
            self.z["blocks.0.ln0.weight"],
            self.z["blocks.0.ln0.bias"],
        )
        for number in range(3):
            self.z[f"blocks.0.att.v{number}"] = self.z[f"blocks.0.att.a{number}"]
        self.native_forward = native_functions(source)["native_forward"]

    def zero_state(self) -> State:
        return [
            tensor
            for _ in range(self.n_layer)
            for tensor in (
                torch.zeros(self.n_embd),
                torch.zeros(self.n_head, self.head_size, self.head_size),
                torch.zeros(self.n_embd),
            )
        ]

    def step(self, token: int, state: State) -> tuple[torch.Tensor, State]:
        return self.native_forward(self, token, state)
