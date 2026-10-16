"""CPU check of the multi-GPU forward kernel's arithmetic with the Triton interpreter (TRITON_INTERPRET=1).

One "GPU" (P = 1) and one CTA per group (S = D), so the in-kernel wait is satisfied by the CTA's own arrival and the
sequential interpreter can run it. Covers the chunked-K MMA (KC < D), both R modes, LSTM and sLSTM, ragged batch
tiles, against the FP64 reference recurrence; and the two R modes against each other bitwise.
"""
import argparse
import json
import os
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402

from flashrnn.flashrnn2 import multi_gpu, reference  # noqa: E402
from flashrnn.flashrnn2.state_contract import begin_sequence  # noqa: E402
from flashrnn.flashrnn2.triton_rtiled import SPIN_LIMIT  # noqa: E402


def run(wx, r, b, init, cell, kc, rmode):
    batch, steps, _, heads, width = wx.shape
    states = init.shape[0]
    ctx = begin_sequence(init, cell, wx.dtype)
    groups = -(-batch // 16) * heads
    out = torch.zeros((batch, steps, heads, width), dtype=torch.float32)
    final = torch.zeros((states, batch, 1, heads, width), dtype=torch.float32)
    xbuf = torch.zeros((groups, 2, 16, width), dtype=wx.dtype)
    sync = torch.zeros((groups + 8,), dtype=torch.int32)
    err = torch.zeros((1,), dtype=torch.int32)
    peers = torch.tensor([xbuf.data_ptr()], dtype=torch.int64)
    syncs = torch.tensor([sync.data_ptr()], dtype=torch.int64)
    s = 1 << (width - 1).bit_length()  # one CTA covers all D columns (slice widths are powers of two)
    kcc = min(kc, s)
    nch = -(-width // kcc)
    multi_gpu._rtiled_mg[(1, groups)](
        wx, r, b, init, out, final, peers, syncs, err, ctx.zero_normalizer,
        batch, steps, groups, groups, 0, 0, 1, 1,
        heads, width, cell == "slstm", 16, s, kcc, nch, 1, SPIN_LIMIT,
        1, 0, 1, 1, width, 0 if rmode == "smem" else 1, 0)
    assert int(err[0]) == 0, "barrier error flag set"
    return out, final


def case(cell, init_mode, batch, steps, heads, width, kc, seed):
    g = torch.Generator().manual_seed(seed)
    states = 4 if cell == "slstm" else 2
    wx = (torch.randn(batch, steps, 4, heads, width, generator=g) * 0.5).half()
    r = (torch.randn(4, heads, width, width, generator=g) * (0.5 / width ** 0.5)).half()
    b = (torch.randn(4, heads, width, generator=g) * 0.1).half()
    init = torch.randn(states, batch, 1, heads, width, generator=g) * 0.2
    if cell == "slstm":
        if init_mode == "zero":
            init[1:] = 0
        else:
            init[2] = init[2].abs() + 1
    init = init.half()
    hist, _ = reference.recurrence(wx.double(), r.double(), b.double(), init.double(), cell)
    ref = hist[0].float()
    out = {}
    s = 1 << (width - 1).bit_length()
    chunks = -(-width // min(kc, s))
    c, _ = run(wx, r, b, init, cell, kc, "stream")
    if chunks <= 2:
        a, _ = run(wx, r, b, init, cell, kc, "smem")
        out["smem_vs_ref"] = ((a - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
        out["stream_bitwise_smem"] = bool(torch.equal(a, c))
    else:  # more than two K chunks: stream mode only
        out["smem_vs_ref"] = ((c - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
        out["stream_bitwise_smem"] = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for cell, init in (("lstm", "warm"), ("slstm", "zero"), ("slstm", "warm")):
        for batch, steps, heads, width, kc in ((3, 5, 1, 32, 32), (3, 5, 1, 64, 32), (19, 4, 2, 48, 32),
                                               (5, 3, 1, 128, 64), (5, 3, 1, 128, 16)):
            m = case(cell, init, batch, steps, heads, width, kc, 7 + width + batch)
            ok = m["smem_vs_ref"] < 2e-2 and m["stream_bitwise_smem"] is not False
            rows.append(dict(cell=cell, init=init, batch=batch, steps=steps, heads=heads, width=width, k_chunk=kc,
                             metrics=m, pass_=ok))
            print(json.dumps(rows[-1]), flush=True)
    status = "PASS" if all(r["pass_"] for r in rows) else "FAIL"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "scale-interpret.json").write_text(json.dumps(dict(rows=rows, status=status), indent=1))
    print("STATUS", status)


if __name__ == "__main__":
    main()
