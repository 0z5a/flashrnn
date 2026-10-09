"""Map the official Mamba2-130M checkpoint into the built-in HF model.

Configuration follows state-spaces/mamba at e9594ce1c732d97440f0332fdc43170a2294dbfa:
Mamba2 defaults are expand=2, d_state=128, headdim=64 and ngroups=1.
Only the embedding key changes spelling; every tensor loads strictly.
"""

import json
from pathlib import Path

import torch
from transformers import Mamba2Config, Mamba2ForCausalLM


def load_mamba2(path: Path) -> Mamba2ForCausalLM:
    original = json.loads((path / "config.json").read_text())
    assert original["ssm_cfg"] == {"layer": "Mamba2"}
    assert original["d_intermediate"] == 0 and not original["attn_layer_idx"]
    state = torch.load(
        path / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    assert torch.equal(state["backbone.embedding.weight"], state["lm_head.weight"])
    config = Mamba2Config(
        hidden_size=original["d_model"],
        num_hidden_layers=original["n_layer"],
        num_heads=2 * original["d_model"] // 64,
        head_dim=64,
        n_groups=1,
        state_size=128,
        conv_kernel=4,
        expand=2,
        chunk_size=256,
        vocab_size=state["backbone.embedding.weight"].shape[0],
        tie_word_embeddings=original["tie_embeddings"],
        layer_norm_epsilon=1e-5,
        rms_norm=original["rms_norm"],
        residual_in_fp32=original["residual_in_fp32"],
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    assert config.vocab_size == (
        (original["vocab_size"] + original["pad_vocab_size_multiple"] - 1)
        // original["pad_vocab_size_multiple"]
        * original["pad_vocab_size_multiple"]
    )
    renamed = {
        "backbone.embeddings.weight"
        if key == "backbone.embedding.weight"
        else key: value
        for key, value in state.items()
    }
    model = Mamba2ForCausalLM(config)
    model.load_state_dict(renamed, strict=True)
    return model
