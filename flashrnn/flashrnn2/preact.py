"""Fused preactivation recompute for the large-D backward (max-width.md section 35).

pre[m, j] = Wx[m, j] + sum_k round(h_{t-1})[m, k] * R2[j, k] + bias[j], with m = (b, t), j = g * D + o and
R2 = recurrent.reshape(4D, D) -- exactly what triton_rtiled_bw.preactivations computes for one head with addmm and
an elementwise add, but in one Triton GEMM on input-dtype tensor cores with FP32 accumulation and FP32 output, the
Wx and bias additions fused into the epilogue. Both GEMM operands are input-dtype values, so every product is exact
in FP32 (as in the FP32 / TF32 GEMM it replaces); only the summation order differs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _preact_kernel(H, RT, WX, BIAS, PRE, M, N, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % (GROUP_M * num_n)) % size_m
    pid_n = (pid % (GROUP_M * num_n)) // size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(H + rm[:, None] * K + (k0 + rk)[None, :], (rm[:, None] < M) & ((k0 + rk)[None, :] < K), 0)
        b = tl.load(RT + (k0 + rk)[:, None] * N + rn[None, :], (rn[None, :] < N) & ((k0 + rk)[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    wx = tl.load(WX + rm[:, None] * N + rn[None, :], mask, 0).to(tl.float32)
    bias = tl.load(BIAS + rn, rn < N, 0).to(tl.float32)
    tl.store(PRE + rm[:, None] * N + rn[None, :], acc + wx + bias[None, :], mask)


def preactivations_fused(wx, recurrent, bias, h_prev):
    """One head: wx [B,T,4,1,D], recurrent [4,1,D,D], bias [4,1,D], h_prev [B,T,1,D] (input dtype).
    Returns the FP32 preactivations [B,T,4,1,D]."""
    batch, steps, gates, heads, width = wx.shape
    if heads != 1:
        raise ValueError("the fused preactivation is for one head")
    m, n, k = batch * steps, gates * width, width
    h2 = h_prev.reshape(m, k).contiguous()
    rt = recurrent.reshape(n, k).t().contiguous()  # [D, 4D]: coalesced B-operand rows
    pre = torch.empty((batch, steps, gates, 1, width), device=wx.device, dtype=torch.float32)
    bm, bn, bk = 128, 128, 64
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _preact_kernel[grid](h2, rt, wx.contiguous(), bias.reshape(n).contiguous(), pre, m, n, k, bm, bn, bk, 8,
                         num_warps=8, num_stages=3)
    return pre


def h_prev_of(wx, initial, history):
    """round(h_{t-1}) in the input dtype [B, T, H, D], as triton_rtiled_bw.preactivations builds it."""
    batch, steps, gates, heads, width = wx.shape
    h_prev = torch.empty((batch, steps, heads, width), device=wx.device, dtype=wx.dtype)
    h_prev[:, :1] = initial[0]
    h_prev[:, 1:] = history[0][:, :-1]
    return h_prev
