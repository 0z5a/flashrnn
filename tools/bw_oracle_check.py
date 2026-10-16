"""CPU checks of the FlashRNN2 gradient oracle (flashrnn2.grad_reference), float64.

  forward   oracle forward == reference.recurrence, bitwise
  v1        oracle gradients == FlashRNN v1 vanilla backward (vanilla_fwbw.bw.backward_sequence) on the same inputs:
            d initial, dWx, dR, db (relative Frobenius and max-abs / max|ref|)
  gradcheck torch.autograd.gradcheck of the oracle against finite differences of the forward:
            LSTM all inputs and outputs; sLSTM hidden outputs h (c and n each carry exp(-m)), with a zero-normalizer start or n0 >= 1 (clamp inactive), every input except the
            initial m (and n0 for the zero start, where the normalizer test switches branch). Fixtures with
            0 < n0 < 1 (clamp can be active) are reported, not gated.
Writes bw-oracle.json; status PASS only if every gated check passes.
"""
import argparse
import json
from pathlib import Path

import torch

from flashrnn.flashrnn.vanilla_fwbw.bw import backward_sequence, lstm_pointwise_bw, slstm_pointwise_bw
from flashrnn.flashrnn.vanilla_fwbw.fw import forward_sequence, lstm_pointwise_fw, slstm_pointwise_fw
from flashrnn.flashrnn2 import grad_reference, reference

D64 = torch.float64


def inputs(cell, init, batch, steps, heads, width, seed):
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    wx = torch.randn(batch, steps, 4, heads, width, generator=g, dtype=D64) * 0.7
    r = torch.randn(4, heads, width, width, generator=g, dtype=D64) / width ** 0.5
    bias = torch.randn(4, heads, width, generator=g, dtype=D64) * 0.2
    initial = torch.randn(states, batch, 1, heads, width, generator=g, dtype=D64) * 0.3
    if cell == "slstm":
        if init == "zero":
            initial[1:] = 0
        elif init == "warm":
            initial[2] = initial[2].abs() + 1
        else:  # clamp: 0 < n0 < 1
            initial[2] = torch.rand(initial[2].shape, generator=g, dtype=D64) * 0.6 + 0.2
    return wx, r, bias, initial


def rel(a, b):
    scale = b.abs().max().item()
    return dict(rel_fro=((a - b).norm() / b.norm().clamp_min(1e-300)).item(),
                max_abs_scaled=((a - b).abs().max() / max(scale, 1e-300)).item())


def check_forward(cell, init, seed):
    wx, r, bias, initial = inputs(cell, init, 3, 7, 2, 8, seed)
    a = grad_reference.recurrence(wx, r, bias, initial, cell)
    b = reference.recurrence(wx, r, bias, initial, cell)
    return dict(bitwise=all(torch.equal(x, y) for x, y in zip(a, b)))


def check_v1(cell, init, seed):
    wx, r, bias, initial = inputs(cell, init, 3, 9, 2, 16, seed)
    g = torch.Generator().manual_seed(seed + 1)
    states = initial.shape[0]
    d_hist = torch.randn(states, 3, 9, 2, 16, generator=g, dtype=D64)
    d_last = torch.randn(states, 3, 1, 2, 16, generator=g, dtype=D64)
    if cell == "slstm":  # the stabilizer receives no gradient under either implementation
        d_hist[3] = 0
        d_last[3] = 0
    leaves = [x.clone().requires_grad_(True) for x in (wx, r, bias, initial)]
    hist, last = grad_reference.recurrence(*leaves, cell)
    ((hist * d_hist).sum() + (last * d_last).sum()).backward()
    ours = [x.grad for x in leaves]
    fw = lstm_pointwise_fw if cell == "lstm" else slstm_pointwise_fw
    bw = lstm_pointwise_bw if cell == "lstm" else slstm_pointwise_bw
    (all_states, _), all_gates = forward_sequence(states_initial=initial[:, :, 0], Wx=wx, R=r, b=bias,
                                                   forward_pointwise=fw, output_gates_and_states_initial=True)
    d_init, d_wx, d_r, d_b = backward_sequence(
        delta_states_all_outside=d_hist.permute(2, 0, 1, 3, 4).contiguous(),
        delta_states_last_outside=d_last[:, :, 0].contiguous(), R=r, states_all=all_states, gates_all=all_gates,
        backward_pointwise=bw)
    v1_forward = (all_states[1:].permute(1, 2, 0, 3, 4) - hist).abs().max().item()
    out = dict(forward_max_abs=v1_forward, d_wx=rel(ours[0], d_wx), d_r=rel(ours[1], d_r), d_bias=rel(ours[2], d_b),
               d_initial=rel(ours[3][:, :, 0], d_init))
    out["pass_"] = v1_forward < 1e-12 and all(v["rel_fro"] < 1e-10 for k, v in out.items() if k.startswith("d_"))
    return out


def check_gradcheck(cell, init, seed):
    wx, r, bias, initial = inputs(cell, init, 2, 4, 1, 4, seed)
    fixed = initial.clone()
    if cell == "lstm":
        free = [0, 1]
    elif init == "zero":
        free = [0, 1]  # n0 = 0 selects the zero-normalizer branch, so n0 stays fixed
    else:
        free = [0, 1, 2]

    def fn(wx_, r_, bias_, *components):
        init_ = fixed.clone()
        for k, comp in zip(free, components):
            init_ = torch.cat([init_[:k], comp.unsqueeze(0), init_[k + 1:]])
        hist, last = grad_reference.recurrence(wx_, r_, bias_, init_, cell)
        if cell == "slstm":
            # Only h is invariant to the stabilizers (c and n each carry exp(-m)); the convention claims the true
            # gradient of losses on h.
            return hist[0], last[0]
        return hist, last

    args = [x.clone().requires_grad_(True) for x in (wx, r, bias)] + \
           [initial[k].clone().requires_grad_(True) for k in free]
    try:
        ok = torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True)
        return dict(pass_=bool(ok), free_initial=free)
    except Exception as error:  # noqa: BLE001
        return dict(pass_=False, free_initial=free, error=str(error)[:400])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_default_dtype(D64)
    rec, gates = dict(cases=[]), {}
    fixtures = [("lstm", "warm")] + [("slstm", init) for init in ("zero", "warm", "clamp")]
    for cell, init in fixtures:
        for seed in (1, 2, 3):
            case = dict(cell=cell, init=init, seed=seed, forward=check_forward(cell, init, seed),
                        v1=check_v1(cell, init, seed), gradcheck=check_gradcheck(cell, init, seed))
            gated_gradcheck = not (cell == "slstm" and init == "clamp")
            case["gated"] = dict(forward=case["forward"]["bitwise"], v1=case["v1"]["pass_"],
                                 gradcheck=case["gradcheck"]["pass_"] if gated_gradcheck else None)
            rec["cases"].append(case)
            gates[f"{cell}-{init}-{seed}"] = all(v for v in case["gated"].values() if v is not None)
    rec["gates"] = gates
    rec["status"] = "PASS" if all(gates.values()) else "FAIL"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "bw-oracle.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(dict(status=rec["status"], gates=gates,
                          clamp_gradcheck=[c["gradcheck"]["pass_"] for c in rec["cases"] if c["init"] == "clamp"])))


if __name__ == "__main__":
    main()
