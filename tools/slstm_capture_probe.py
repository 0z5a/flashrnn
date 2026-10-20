"""max-width 74.2(b): where does the sLSTM BF16 D = 2048 B = 64 training step stop being capturable? Prints the
capture status after the forward, the loss and the backward, with the library's configuration."""
import ctypes
import traceback

import torch

import bw_qualify
from flashrnn.flashrnn2 import training

rt = ctypes.CDLL("libcudart.so.12")
case = dict(cell="slstm", dtype="bfloat16", width=2048, batch=64, steps=256, heads=1, init="warm")
leaves, cot_h, _ = bw_qualify.inputs(case, 1)
cfg, source = training.training_forward("slstm", torch.bfloat16, 64, 256, 1, 2048)
print("config", cfg.as_dict(), source)


def status(tag):
    st = ctypes.c_int(-1)
    rt.cudaStreamIsCapturing(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), ctypes.byref(st))
    print(f"{tag:12s} capture_status={st.value} (1 active, 2 invalidated)", flush=True)


def step(trace):
    xs = [x.detach().requires_grad_(True) for x in leaves]
    y, _ = training.recurrence(*xs, "slstm", output_mode="hidden", config=cfg, backward_options={}); trace("forward")
    loss = (y.float() * cot_h).sum(); trace("loss")
    g = torch.autograd.grad(loss, xs); trace("backward")
    return g[0]


step(lambda t: None); torch.cuda.synchronize()
side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(2):
        step(lambda t: None)
torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        status("begin")
        step(status)
    print("CAPTURE_OK")
except Exception:
    print("CAPTURE_FAIL", traceback.format_exc()[-1800:])
