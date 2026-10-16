"""Map all FLA RWKV6 weights to the pinned BlinkDL CPU forward bodies."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

SOURCE_SHA = "9c6790f2d9d5b9a9a5718c5bdd85bfbdebaa6e2b35185af268bfde1d74835ae2"


def blinkdl_reference(weights: dict[str, torch.Tensor], source: Path) -> nn.Module:
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA
    cls = next(
        n
        for n in ast.parse(data).body
        if isinstance(n, ast.ClassDef) and n.name == "RWKV_RNN"
    )
    methods = {"layer_norm", "channel_mixing", "time_mixing", "forward"}
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    assert {n.name for n in cls.body} == methods
    namespace = {
        "torch": torch,
        "F": F,
        "MyModule": nn.Module,
        "MyFunction": lambda fn: fn,
    }
    exec(  # noqa: S102
        compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace
    )
    model = namespace["RWKV_RNN"]()
    model.args = SimpleNamespace(n_layer=24, n_embd=2048, vocab_size=65536)
    model.n_head, model.head_size = 32, 64
    remaining = dict(weights)

    def norm(prefix: str) -> SimpleNamespace:
        return SimpleNamespace(
            weight=remaining.pop(prefix + ".weight"),
            bias=remaining.pop(prefix + ".bias"),
        )

    def linear(name: str) -> SimpleNamespace:
        return SimpleNamespace(weight=remaining.pop(name))

    model.w = SimpleNamespace(
        emb=linear("model.embeddings.weight"),
        head=linear("lm_head.weight"),
        ln_out=norm("model.norm"),
        blocks={},
    )
    reordered = []
    for layer in range(24):
        prefix = f"model.layers.{layer}"
        block = SimpleNamespace(
            ln1=norm(prefix + ".attn_norm"), ln2=norm(prefix + ".ffn_norm")
        )
        if layer == 0:
            block.ln0 = norm(prefix + ".pre_norm")
        attn = prefix + ".attn."
        first = remaining.pop(attn + "x_proj.0.linear.weight")
        second = remaining.pop(attn + "x_proj.2.weight")
        order, inverse = [1, 2, 3, 0, 4], [3, 0, 1, 2, 4]
        w1 = first.reshape(5, 32, 2048)[order].reshape(160, 2048).t().contiguous()
        w2 = second.t().reshape(5, 32, 2048)[order].contiguous()
        assert torch.equal(
            w1.t().reshape(5, 32, 2048)[inverse].reshape_as(first), first
        )
        assert torch.equal(w2[inverse].reshape(160, 2048).t(), second)
        r, w, k, v, g = remaining.pop(attn + "x_bias").unbind(0)
        block.att = SimpleNamespace(
            time_maa_x=remaining.pop(attn + "x_proj.0.mu"),
            time_maa_w=w,
            time_maa_k=k,
            time_maa_v=v,
            time_maa_r=r,
            time_maa_g=g,
            time_maa_w1=w1,
            time_maa_w2=w2,
            time_decay_w1=remaining.pop(attn + "w_proj.linear.lora.0.weight").t(),
            time_decay_w2=remaining.pop(attn + "w_proj.linear.lora.2.weight").t(),
            time_decay=remaining.pop(attn + "w_proj.linear.lora.2.bias"),
            time_faaaa=remaining.pop(attn + "bonus").unsqueeze(-1),
            receptance=linear(attn + "r_proj.linear.weight"),
            key=linear(attn + "k_proj.linear.weight"),
            value=linear(attn + "v_proj.linear.weight"),
            gate=linear(attn + "g_proj.linear.weight"),
            output=linear(attn + "o_proj.weight"),
            ln_x=norm(attn + "g_norm"),
        )
        ffn = prefix + ".ffn."
        block.ffn = SimpleNamespace(
            time_maa_k=remaining.pop(ffn + "key.mu"),
            time_maa_r=remaining.pop(ffn + "receptance.mu"),
            key=linear(ffn + "key.linear.weight"),
            value=linear(ffn + "value.weight"),
            receptance=linear(ffn + "receptance.linear.weight"),
        )
        model.w.blocks[layer] = block
        reordered.append({"layer": layer, "both_lora_roundtrips_bitwise": True})
    assert not remaining and len(weights) == 582
    model.provenance = {
        "scope": "BLINKDL_UNCHANGED_CPU_FORWARD_BODIES_WITH_REVERSE_MAPPED_FLA_CHECKPOINT",
        "source_sha256": SOURCE_SHA,
        "mapped_checkpoint_tensors": len(weights),
        "mapped_parameters": sum(t.numel() for t in weights.values()),
        "reordered_projection_checks": reordered,
        "layer_norm_eps": 1e-5,
        "group_norm_eps": 64e-5,
        "constructor_replaced": True,
        "torchscript_decorators_disabled": True,
        "native_original_checkpoint_or_tokenizer_qualified": False,
        "performance_claim": False,
    }
    model.eval()
    return model
