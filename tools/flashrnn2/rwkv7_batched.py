"""Batched Torch RWKV7 recurrence, qualified against the pinned scalar native path."""

import torch
from rwkv7_native import NativeRWKV7
from torch import nn
from torch.nn import functional as F

Output = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class RWKV7Layer(nn.Module):
    def __init__(self, native: NativeRWKV7, index: int) -> None:
        super().__init__()
        self.index = index
        self.heads, self.head_size = native.n_head, native.head_size
        self.hidden = native.n_embd
        block, attn, ffn = (
            f"blocks.{index}{suffix}" for suffix in (".", ".att.", ".ffn.")
        )
        for name in (
            "x_r",
            "x_w",
            "x_k",
            "x_v",
            "x_a",
            "x_g",
            "w0",
            "w1",
            "w2",
            "a0",
            "a1",
            "a2",
            "g1",
            "g2",
            "k_k",
            "k_a",
            "r_k",
        ):
            self.register_buffer(name, native.z[attn + name])
        for name in ("v0", "v1", "v2"):
            self.register_buffer(
                name, native.z[attn + name] if index else torch.empty(0)
            )
        for short, key in {
            "rw": attn + "receptance.weight",
            "kw": attn + "key.weight",
            "vw": attn + "value.weight",
            "ow": attn + "output.weight",
            "group_w": attn + "ln_x.weight",
            "group_b": attn + "ln_x.bias",
            "attn_norm_w": block + "ln1.weight",
            "attn_norm_b": block + "ln1.bias",
            "ffn_norm_w": block + "ln2.weight",
            "ffn_norm_b": block + "ln2.bias",
            "ffn_x_k": ffn + "x_k",
            "ffn_kw": ffn + "key.weight",
            "ffn_vw": ffn + "value.weight",
        }.items():
            self.register_buffer(short, native.z[key])

    def forward(
        self,
        hidden: torch.Tensor,
        previous_attn: torch.Tensor,
        previous_ffn: torch.Tensor,
        state: torch.Tensor,
        first_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = hidden.shape[0]
        x = F.layer_norm(hidden, (self.hidden,), self.attn_norm_w, self.attn_norm_b)
        delta = previous_attn - x
        xr, xw, xk = x + delta * self.x_r, x + delta * self.x_w, x + delta * self.x_k
        xv, xa, xg = x + delta * self.x_v, x + delta * self.x_a, x + delta * self.x_g
        r = F.linear(xr, self.rw)
        w = torch.tanh(xw @ self.w1) @ self.w2
        k, v = F.linear(xk, self.kw), F.linear(xv, self.vw)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = F.normalize(
            (k * self.k_k).view(batch, self.heads, self.head_size), dim=-1, p=2.0
        )
        k = k * (1 + (a - 1) * self.k_a)
        if self.index == 0:
            first_value = v
        else:
            v = v + (first_value - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )
        w = torch.exp(-0.606531 * torch.sigmoid(self.w0 + w.float()))
        vh = v.view(batch, self.heads, self.head_size)
        kh = k.view(batch, self.heads, self.head_size)
        rh = r.view(batch, self.heads, self.head_size)
        ah = a.view(batch, self.heads, self.head_size)
        vk = vh.unsqueeze(-1) @ kh.unsqueeze(-2)
        ab = (-kk).unsqueeze(-1) @ (kk * ah).unsqueeze(-2)
        state = (
            state * w.view(batch, self.heads, 1, self.head_size)
            + state @ ab.float()
            + vk.float()
        )
        output = (state.to(x.dtype) @ rh.unsqueeze(-1)).view(batch, self.hidden)
        output = F.group_norm(output, self.heads, self.group_w, self.group_b, eps=64e-5)
        output = output + (
            (rh * kh * self.r_k.view(self.heads, self.head_size)).sum(-1, keepdim=True)
            * vh
        ).view(batch, self.hidden)
        hidden = hidden + F.linear(output * g, self.ow)
        ffn_x = F.layer_norm(hidden, (self.hidden,), self.ffn_norm_w, self.ffn_norm_b)
        ffn_k = ffn_x + (previous_ffn - ffn_x) * self.ffn_x_k
        hidden = hidden + F.linear(
            torch.relu(F.linear(ffn_k, self.ffn_kw)) ** 2, self.ffn_vw
        )
        return hidden, x, ffn_x, state, first_value


class RWKV7Batched(nn.Module):
    def __init__(self, native: NativeRWKV7) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            RWKV7Layer(native, i) for i in range(native.n_layer)
        )
        self.hidden = native.n_embd
        self.register_buffer("embedding", native.z["emb.weight"])
        self.register_buffer("norm_w", native.z["ln_out.weight"])
        self.register_buffer("norm_b", native.z["ln_out.bias"])
        self.register_buffer("head", native.z["head.weight"])

    def forward(
        self, ids: torch.Tensor, shifts: torch.Tensor, states: torch.Tensor
    ) -> Output:
        shifts, states = shifts.clone(), states.clone()
        logits = []
        for token in ids.unbind(1):
            hidden = self.embedding[token]
            first_value = torch.empty_like(hidden)
            for i, layer in enumerate(self.layers):
                hidden, attn, ffn, state, first_value = layer(
                    hidden, shifts[0, i], shifts[1, i], states[i], first_value
                )
                shifts[0, i].copy_(attn)
                shifts[1, i].copy_(ffn)
                states[i].copy_(state)
            hidden = F.layer_norm(hidden, (self.hidden,), self.norm_w, self.norm_b)
            logits.append(F.linear(hidden, self.head))
        return torch.stack(logits, 1), shifts, states

    @torch.jit.export
    def prefill(
        self, ids: torch.Tensor, shifts: torch.Tensor, states: torch.Tensor
    ) -> Output:
        return self.forward(ids, shifts, states)

    @torch.jit.export
    def decode(
        self, ids: torch.Tensor, shifts: torch.Tensor, states: torch.Tensor
    ) -> Output:
        return self.forward(ids, shifts, states)
