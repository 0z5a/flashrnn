"""cuDNN baseline and end-to-end training (docs/flashrnn2/max-width.md section 70).

Usage: e2e_bench.py --case JSON --out FILE --start N [--blocks 20] [--warmup-blocks 1]
Case: {"mode": "layer" | "lm", "cell", "dtype", "width", "batch", "steps", "arms": ["frnn2", "v1", "v1cuda", "cudnn"],
       "config": optional pinned KernelConfig fields for frnn2's forward (default: training.training_forward),
       "calls": steps per block entry, "layers": 4 (lm), "vocab": 4096 (lm), "check_steps": 20 (lm)}
Arms (same function, same initial weights):
  frnn2   FlashRNN2 library default (training.recurrence, configuration from training.training_forward)
  v1      pinned FlashRNN1 cuda_fused (FRNN_V1_BASE / FRNN_V1_BUILD_ROOT)
  v1cuda  pinned FlashRNN1 cuda (alternating)
  cudnn   torch.nn.LSTM (cuDNN) with weight_ih = W, weight_hh = R, bias_ih = b, bias_hh = 0; LSTM only
A recurrent layer is x -> W x (input projection) -> recurrence -> hidden output.
  layer  one layer; one call = forward + backward of <cot, h> into x, W, R, bias
  lm     embedding -> layers x (RMSNorm -> recurrent layer -> residual) -> RMSNorm -> readout, cross-entropy on
         learnable synthetic tokens (each sequence repeats a random 32-token segment); one call = one AdamW step
         (lr 3e-4, gradient norm clipped to 1); every arm owns its model and optimizer
Parameters and optimizer state are FP32 (mixed precision); every forward runs in the input dtype.
Before timing: layer mode compares every arm's output and gradients with frnn2's (relative Frobenius); lm mode trains
every arm for check_steps steps on the same batches and records its losses. Timing: ABBA/BAAB blocks of frnn2 against
every other arm (paired.json, the layout of heads_bench paired).
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from flashrnn.flashrnn2 import training

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=1, default=str))
    tmp.replace(path)


class Recurrent:
    """One recurrent layer: FP32 master parameters, run in the input dtype (mixed precision: every arm casts its
    weights to the input dtype in each forward, as torch.autocast does for cuDNN)."""

    def __init__(self, case, gen):
        d, dt = case["width"], DT[case["dtype"]]
        self.cell, self.d, self.dt = case["cell"], d, dt
        self.W = (torch.randn(4 * d, d, generator=gen) * (1.0 / d ** 0.5)).cuda().requires_grad_(True)
        self.R = (torch.randn(4, 1, d, d, generator=gen) * (0.5 / d ** 0.5)).cuda().requires_grad_(True)
        self.b = (torch.randn(4, 1, d, generator=gen) * 0.1).cuda().requires_grad_(True)
        self.states = 4 if self.cell == "slstm" else 2

    def params(self):
        return [self.W, self.R, self.b]

    def clone(self):
        c = Recurrent.__new__(Recurrent)
        c.cell, c.d, c.dt, c.states = self.cell, self.d, self.dt, self.states
        c.W, c.R, c.b = (p.detach().clone().requires_grad_(True) for p in self.params())
        return c

    def init(self, batch):
        z = torch.zeros(self.states, batch, 1, 1, self.d, device="cuda", dtype=self.dt)
        if self.cell == "slstm":
            z[2] = 1
        return z

    def forward(self, x, arm, ctx):
        b, t, d = x.shape
        W, R, bias = (p.to(self.dt) for p in self.params())
        wx = (x.reshape(b * t, d) @ W.t()).reshape(b, t, 4, 1, d)
        if arm == "frnn2":
            y, _ = training.recurrence(wx, R, bias, self.init(b), self.cell, output_mode="hidden", config=ctx["cfg"])
            return y[:, :, 0]
        states, _ = ctx["v1"]["flashrnn"](wx, R, bias, self.init(b), config=ctx["v1cfg"][arm])
        return states[0][:, :, 0]


class CudnnRecurrent:
    """torch.nn.LSTM (cuDNN) with FP32 parameters copied from a Recurrent layer (bias_hh = 0), flattened once, run
    under torch.autocast in the input dtype."""

    def __init__(self, src):
        d = src.d
        self.d, self.dt = d, src.dt
        self.lstm = torch.nn.LSTM(d, d, batch_first=True).cuda()
        with torch.no_grad():
            self.lstm.weight_ih_l0.copy_(src.W)
            self.lstm.weight_hh_l0.copy_(src.R.reshape(4 * d, d))
            self.lstm.bias_ih_l0.copy_(src.b.reshape(4 * d))
            self.lstm.bias_hh_l0.zero_()
        self.lstm.flatten_parameters()

    def params(self):
        return [self.lstm.weight_ih_l0, self.lstm.weight_hh_l0, self.lstm.bias_ih_l0, self.lstm.bias_hh_l0]

    def forward(self, x, arm, ctx):
        with torch.autocast("cuda", dtype=self.dt):
            out, _ = self.lstm(x)
        return out.to(self.dt)


def for_arm(layer, arm):
    return CudnnRecurrent(layer) if arm == "cudnn" else layer.clone()


def rms(x):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)


class LM:
    def __init__(self, case, gen):
        d, v = case["width"], case.get("vocab", 4096)
        self.dt = DT[case["dtype"]]
        self.emb = (torch.randn(v, d, generator=gen) * 0.1).cuda().requires_grad_(True)
        self.layers = [Recurrent(case, gen) for _ in range(case.get("layers", 4))]
        self.out = (torch.randn(d, v, generator=gen) * (1.0 / d ** 0.5)).cuda().requires_grad_(True)

    def params(self):
        return [self.emb, self.out] + [p for l in self.layers for p in l.params()]

    def clone(self, arm):
        c = LM.__new__(LM)
        c.dt = self.dt
        c.emb, c.out = (p.detach().clone().requires_grad_(True) for p in (self.emb, self.out))
        c.layers = [for_arm(l, arm) for l in self.layers]
        return c

    def loss(self, ids, arm, ctx):
        x = self.emb[ids].to(self.dt)
        for layer in self.layers:
            x = x + layer.forward(rms(x), arm, ctx)
        logits = (rms(x) @ self.out.to(self.dt)).float()
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def contexts(case, arms):
    ctx = {}
    dt = DT[case["dtype"]]
    if case.get("config"):  # a pinned forward configuration (KernelConfig fields)
        from flashrnn.flashrnn2.execution_policy import KernelConfig
        cfg, source = KernelConfig(**case["config"]), "pinned"
    else:
        cfg, source = training.training_forward(case["cell"], dt, case["batch"], case["steps"], 1, case["width"])
    ctx["cfg"], ctx["cfg_source"] = cfg, source
    if any(a in arms for a in ("v1", "v1cuda")):
        import rtiled_m4
        v1 = rtiled_m4.v1_module()
        ctx["v1"] = v1
        ctx["v1cfg"] = {a: v1["config"](function=case["cell"], backend="cuda_fused" if a == "v1" else "cuda",
                                        dtype=case["dtype"], batch_size=case["batch"], hidden_dim=case["width"],
                                        num_heads=1) for a in ("v1", "v1cuda") if a in arms}
    if "cudnn" in arms and case["cell"] != "lstm":
        raise ValueError("cuDNN has no sLSTM")
    ctx["cudnn_version"] = torch.backends.cudnn.version()
    return ctx


def relfro(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def build(case, arms, seed):
    gen = torch.Generator().manual_seed(seed)
    if case["mode"] == "layer":
        base = Recurrent(case, gen)
        x = (torch.randn(case["batch"], case["steps"], case["width"], generator=gen) * 0.5).to("cuda", base.dt)
        cot = torch.randn(case["batch"], case["steps"], case["width"], generator=gen).cuda()
        models = {a: for_arm(base, a) for a in arms}
        return dict(models=models, x=x, cot=cot)
    base = LM(case, gen)
    seg = torch.randint(0, case.get("vocab", 4096), (64, case["batch"], 32), generator=gen)
    data = seg.repeat(1, 1, -(-case["steps"] // 32))[:, :, :case["steps"]].contiguous().cuda()
    models = {a: base.clone(a) for a in arms}
    opts = {a: torch.optim.AdamW(m.params(), lr=3e-4, foreach=True) for a, m in models.items()}
    return dict(models=models, opts=opts, data=data, cursor={a: 0 for a in arms})


def make_step(case, state, arm, ctx):
    m = state["models"][arm]
    if case["mode"] == "layer":
        x = state["x"].detach().requires_grad_(True)

        def step():
            y = m.forward(x, arm, ctx)
            grads = torch.autograd.grad((y.float() * state["cot"]).sum(), [x] + m.params())
            return y, grads
        return step
    opt, data = state["opts"][arm], state["data"]

    def step():
        i = state["cursor"][arm] % data.shape[0]
        state["cursor"][arm] += 1
        opt.zero_grad(set_to_none=True)
        loss = m.loss(data[i], arm, ctx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.params(), 1.0, foreach=True)
        opt.step()
        return loss.detach()
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--warmup-blocks", type=int, default=1)
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text())
    rec = dict(case=case, start=a.start, started=time.time(), gpu=torch.cuda.get_device_name(), blocks=[],
               arm_errors={}, outputs_stable={}, warmup_blocks=a.warmup_blocks)
    try:
        torch.cuda.set_device(0)
        arms = list(case["arms"])
        ctx = contexts(case, arms)
        rec["config"] = ctx["cfg"].as_dict()
        rec["config_source"] = ctx["cfg_source"]
        rec["cudnn_version"] = ctx["cudnn_version"]
        state = build(case, arms, 4242 + case["width"] + case["batch"])
        steps = {}
        for arm in arms:  # build every arm once; a failing arm is recorded and dropped
            try:
                steps[arm] = make_step(case, state, arm, ctx)
                steps[arm]()
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                rec["arm_errors"][arm] = traceback.format_exc()[-1500:]
                steps.pop(arm, None)
        if "frnn2" not in steps:
            raise RuntimeError("frnn2 arm failed")
        # Checks (start 0 only): layer -> outputs and gradients against frnn2; lm -> loss curves on fresh models.
        if a.start == 0:
            if case["mode"] == "layer":
                ref_y, ref_g = steps["frnn2"]()
                chk = {}
                for arm in steps:
                    y, g = steps[arm]()
                    chk[arm] = dict(y=relfro(y, ref_y), dx=relfro(g[0], ref_g[0]), dW=relfro(g[1], ref_g[1]),
                                    dR=relfro(g[2].reshape(ref_g[2].shape), ref_g[2]),
                                    db=relfro(g[3].reshape(ref_g[3].shape), ref_g[3]))
                rec["check"] = chk
            else:
                fresh = build(case, list(steps), 4242 + case["width"] + case["batch"])
                curves = {}
                for arm in steps:
                    st = make_step(case, fresh, arm, ctx)
                    curves[arm] = [float(st()) for _ in range(case.get("check_steps", 20))]
                ref = curves["frnn2"]
                rec["check"] = dict(losses=curves, max_rel_vs_frnn2={
                    arm: max(abs(x - y) / abs(y) for x, y in zip(c, ref)) for arm, c in curves.items()})
                del fresh
            torch.cuda.synchronize()
        calls = case.get("calls", 3)

        def timed(fn):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(calls):
                fn()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) / calls
        others = [x for x in steps if x != "frnn2"]
        for block in range(a.warmup_blocks + a.blocks):
            row = {}
            for other in others:
                order = ["frnn2", other, other, "frnn2"] if block % 2 == 0 else [other, "frnn2", "frnn2", other]
                t = {"frnn2": [], other: []}
                for name in order:
                    t[name].append(timed(steps[name]))
                row[other] = dict(one_ms=sum(t["frnn2"]) / 2, other_ms=sum(t[other]) / 2, order=order)
            if block >= a.warmup_blocks:
                rec["blocks"].append(row)
        rec["status"] = "DONE"
    except Exception:  # noqa: BLE001
        rec.update(status="ERROR", error=traceback.format_exc()[-2000:])
    rec["finished"] = time.time()
    write(a.out, rec)
    write(Path(a.out).parent / "DONE.json", dict(status=rec["status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
