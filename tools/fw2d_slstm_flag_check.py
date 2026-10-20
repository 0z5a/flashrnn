"""Check of the sLSTM 2-D forward's device-side first-step flag (max-width.md section 76): sLSTM BF16, D = 2048,
B = 64, with the initial normalizer zero (flag true) and nonzero (flag false). The 2-D forward's output must match the
TileLang forward's within BF16 tolerance in both cases, and the two cases must differ (the flag is honoured)."""
import json
import sys

import torch

import bw_qualify
from flashrnn.flashrnn2 import dispatch, hybrid_fw2d
from flashrnn.flashrnn2.execution_policy import KernelConfig
from flashrnn.flashrnn2.state_contract import begin_sequence

case = dict(cell="slstm", dtype="bfloat16", width=2048, batch=64, steps=64, heads=1, init="warm")
leaves, _, _ = bw_qualify.inputs(case, 77)
wx, r, b, init = leaves
tl = KernelConfig(family="tilelang", slice_width=16, sync="grid", warps=8, reg_k=896)
res = {}
for name in ("zero", "nonzero"):
    i0 = init.clone()
    i0[2] = 0 if name == "zero" else i0[2].abs() + 1
    ctx = begin_sequence(i0, "slstm", wx.dtype)
    with torch.no_grad():
        ref = dispatch.recurrence(wx, r, b, i0, "slstm", output_mode="full", snapshot_dtype=torch.float32,
                                  config=tl)[0]
        out = hybrid_fw2d.recurrence(wx, r, b, i0, "slstm", sequence_context=ctx, output_mode="full",
                                     snapshot_dtype=torch.float32)[0]
    rel = ((out[0] - ref[0]).norm() / ref[0].norm()).item()
    res[name] = dict(flag=bool(ctx.zero_normalizer.item()), rel_fro_h_vs_tilelang=rel, out=out[0].float())
res_out = {k: {kk: vv for kk, vv in v.items() if kk != "out"} for k, v in res.items()}
res_out["cases_differ"] = not torch.equal(res["zero"]["out"], res["nonzero"]["out"])
res_out["pass_"] = res_out["cases_differ"] and all(v["rel_fro_h_vs_tilelang"] < 5e-2 for k, v in res_out.items()
                                                    if isinstance(v, dict))
print(json.dumps(res_out))
open(sys.argv[1], "w").write(json.dumps(res_out, indent=1))
