"""Fused preactivation recompute for the large-D backward (max-width.md section 35).

pre[m, j] = Wx[m, j] + sum_k round(h_{t-1})[m, k] * R2[j, k] + bias[j], with m = (b, t), j = g * D + o and
R2 = recurrent.reshape(4D, D) -- exactly what triton_rtiled_bw.preactivations computes for one head with addmm and
an elementwise add, but in one Triton GEMM on input-dtype tensor cores with FP32 accumulation and FP32 output, the
Wx and bias additions fused into the epilogue. Both GEMM operands are input-dtype values, so every product is exact
in FP32 (as in the FP32 / TF32 GEMM it replaces); only the summation order differs.

H > 1 (max-width.md section 60): the second grid axis is the head; each head runs exactly the H = 1 tile arithmetic on
its own rows of R, h and Wx (at H = 1 the addresses reduce to the one-head layout), so every head's result is bitwise
the H = 1 call's on that head.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _preact_kernel(H, RT, WX, BIAS, PRE, M, N, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, HEADS: tl.constexpr):
    pid = tl.program_id(0)
    head = tl.program_id(1)
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
        a = tl.load(H + rm[:, None] * (HEADS * K) + head * K + (k0 + rk)[None, :],
                    (rm[:, None] < M) & ((k0 + rk)[None, :] < K), 0)
        b = tl.load(RT + head * K * N + (k0 + rk)[:, None] * N + rn[None, :],
                    (rn[None, :] < N) & ((k0 + rk)[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    col = (rn // K) * (HEADS * K) + head * K + rn % K  # (gate, unit) -> [4, HEADS, D] offset
    wx = tl.load(WX + rm[:, None] * (HEADS * N) + col[None, :], mask, 0).to(tl.float32)
    bias = tl.load(BIAS + col, rn < N, 0).to(tl.float32)
    tl.store(PRE + rm[:, None] * (HEADS * N) + col[None, :], acc + wx + bias[None, :], mask)


def preactivations_fused(wx, recurrent, bias, h_prev):
    """wx [B,T,4,H,D], recurrent [4,H,D,D], bias [4,H,D], h_prev [B,T,H,D] (input dtype).
    Returns the FP32 preactivations [B,T,4,H,D]."""
    batch, steps, gates, heads, width = wx.shape
    m, n, k = batch * steps, gates * width, width
    h2 = h_prev.contiguous()
    # [H, D, 4D]: per head R2^T (R2 = recurrent[:, h].reshape(4D, D)), coalesced B-operand rows
    rt = recurrent.permute(1, 3, 0, 2).reshape(heads, k, n).contiguous()
    pre = torch.empty((batch, steps, gates, heads, width), device=wx.device, dtype=torch.float32)
    bm, bn, bk = 128, 128, 64
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), heads)
    _preact_kernel[grid](h2, rt, wx.contiguous(), bias.contiguous(), pre, m, n, k, bm, bn, bk, 8, heads,
                         num_warps=8, num_stages=3)
    return pre


def h_prev_of(wx, initial, history):
    """round(h_{t-1}) in the input dtype [B, T, H, D], as triton_rtiled_bw.preactivations builds it."""
    batch, steps, gates, heads, width = wx.shape
    h_prev = torch.empty((batch, steps, heads, width), device=wx.device, dtype=wx.dtype)
    h_prev[:, :1] = initial[0]
    h_prev[:, 1:] = history[0][:, :-1]
    return h_prev
