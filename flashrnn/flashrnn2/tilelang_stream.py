"""TileLang grid kernel with wide batch tiles for large D (BM = 32 / 64): R resident in registers + shared memory,
h^T streamed in K chunks.

At large D one batch-tile group needs k = D/16 CTAs, so only one group is resident and B = 64 runs four 16-row tiles
one after another (docs/flashrnn2/max-width.md sections 7-8). Here one CTA carries BM = 32 or 64 batch rows per step
(WGMMA N = BM). h^T [D, BM] no longer fits shared memory next to R, so it is streamed from the exchange buffer in
KC-row chunks: the first KR rows of the CTA's R slice live in NR = KR / KC register fragments (WGMMA A from registers,
one chunk of h^T staged per fragment through two alternating shared tiles), the remaining rows in a shared buffer
[NCS, 4S, KC] consumed by a pipelined loop that copies the matching h^T chunk ahead of its WGMMA. The exchange
(double-buffered XBUF, one grid barrier per step), the cell epilogue and the outputs are those of build_grid.

TileLang's builder parses the kernel's Python source, and one register fragment per chunk needs one name per chunk, so
the kernel is generated as source text for each (D, KR, BM, ...) and imported from a cache directory.
"""
import hashlib
import importlib.util
import os
from pathlib import Path

S, KC, THREADS = 16, 64, 128
SMEM_LIMIT = 232448
_KERNELS = {}


def stream_smem(width, reg_k, bm, stages=2):
    return (4 * S * (width - reg_k) * 2          # R rows in shared memory
            + stages * KC * bm * 2               # pipelined h^T chunks (shared part)
            + 2 * KC * bm * 2                    # two alternating h^T chunks (register part)
            + 4 * S * bm * 4)                    # gate stage


def min_reg_k(width, bm):
    for reg_k in range(0, 1025, KC):
        if reg_k <= width - KC and stream_smem(width, reg_k, bm) <= SMEM_LIMIT:
            return reg_k
    return None


def supported_stream(width, reg_k, bm, sms=132):
    if bm not in (32, 64) or width % KC or reg_k % KC or not (0 <= reg_k <= width - KC):
        return False
    # Register budget per thread: 16 per R chunk (64 x 64 FP16 over 128 threads), gates 4S*BM/128, epilogue state.
    regs = reg_k // KC * 16 + 4 * S * bm // THREADS + 6 * bm * S // THREADS
    return 2 <= width // S <= sms and stream_smem(width, reg_k, bm) <= SMEM_LIMIT and regs <= 200


_TEMPLATE = '''
import tilelang.language as T


def make(B, TT, H, D, KR, BM, k, resident, slots, tiles, groups, slstm, dtype, snapshot, out_mode, states):
    S, KC = {S}, {KC}
    NR, NCS = KR // KC, (D - KR) // KC
    acc = "float32"
    out_shape = {{0: (states, B, TT, H, D), 1: (B, TT, H, D), 2: (B, 1, H, D)}}[out_mode]

    @T.prim_func
    def main(
        WX: T.Tensor((B, TT, 4, H, D), dtype),
        R: T.Tensor((4, H, D, D), dtype),
        BIAS: T.Tensor((4, H, D), dtype),
        INIT: T.Tensor((states, B, 1, H, D), acc),
        ZERO: T.Tensor((1,), "int32"),
        OUT: T.Tensor(out_shape, snapshot),
        FINAL: T.Tensor((states, B, 1, H, D), acc),
        XBUF: T.Tensor((2, groups, D, BM), dtype),
        NSTEPS: T.int32,
        BASE: T.int32,
        START: T.int32,
    ):
        with T.Kernel(k, resident, threads={THREADS}) as (s, gy):
{alloc}
            rs = T.alloc_shared((NCS, 4 * S, KC), dtype)
            hb = T.alloc_shared((KC, BM), dtype)
            gates = T.alloc_fragment((4 * S, BM), acc)
            stage = T.alloc_shared((4 * S, BM), acc)
            c = T.alloc_fragment((BM, S), acc)
            n = T.alloc_fragment((BM, S), acc)
            m = T.alloc_fragment((BM, S), acc)
            hid = T.alloc_fragment((BM, S), acc)

            for slot in T.serial(slots):
                g = slot * resident + gy
                active = g < groups
                head = T.min(g, groups - 1) // tiles
                bt = T.min(g, groups - 1) % tiles
                if active:
{load_r}
                    for cc, q, kk in T.Parallel(NCS, 4 * S, KC):
                        rs[cc, q, kk] = R[q // S, head, s * S + q % S, KR + cc * KC + kk]
                    for j, b in T.Parallel(S, BM):  # own columns of h_0 into the parity read at t = 0
                        XBUF[1, g, s * S + j, b] = T.if_then_else(bt * BM + b < B,
                                                                  T.cast(INIT[0, bt * BM + b, 0, head, s * S + j], dtype),
                                                                  T.cast(0, dtype))
                    for b, j in T.Parallel(BM, S):
                        valid = bt * BM + b < B
                        c[b, j] = T.if_then_else(valid, INIT[1, bt * BM + b, 0, head, s * S + j], 0.0)
                        hid[b, j] = T.if_then_else(valid, INIT[0, bt * BM + b, 0, head, s * S + j], 0.0)
                        if slstm:
                            n[b, j] = T.if_then_else(valid, INIT[2, bt * BM + b, 0, head, s * S + j], 1.0)
                            m[b, j] = T.if_then_else(valid, INIT[3, bt * BM + b, 0, head, s * S + j], 0.0)
                T.fence_proxy_async()
                T.sync_grid()

                for t in T.serial(NSTEPS):
                    if active:
                        par = (t + 1) % 2
{gemm_r}
                        for cc in T.Pipelined(NCS, num_stages=2):
                            T.copy(XBUF[par, g, KR + cc * KC:KR + (cc + 1) * KC, 0:BM], hb)
                            T.gemm(rs[cc, :, :], hb, gates, clear_accum={clear_s})
                        T.copy(gates, stage)
                        for b, j in T.Parallel(BM, S):
                            row = bt * BM + b
                            col = s * S + j
                            valid = row < B
                            step = BASE + t
                            gi = stage[j, b] + T.if_then_else(valid, T.cast(WX[row, step, 0, head, col], acc), 0.0) \\
                                + T.cast(BIAS[0, head, col], acc)
                            gf = stage[S + j, b] + T.if_then_else(valid, T.cast(WX[row, step, 1, head, col], acc),
                                                                  0.0) + T.cast(BIAS[1, head, col], acc)
                            gz = stage[2 * S + j, b] + T.if_then_else(
                                valid, T.cast(WX[row, step, 2, head, col], acc), 0.0) + T.cast(BIAS[2, head, col], acc)
                            go = stage[3 * S + j, b] + T.if_then_else(
                                valid, T.cast(WX[row, step, 3, head, col], acc), 0.0) + T.cast(BIAS[3, head, col], acc)
                            if slstm:
                                zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                                og = 1.0 / (1.0 + T.exp(-go))
                                logf = T.min(gf, 0.0) - T.log(1.0 + T.exp(-T.abs(gf))) + m[b, j]
                                nm = T.if_then_else((START != 0) & (t == 0) & (ZERO[0] != 0), gi, T.max(gi, logf))
                                ig = T.exp(gi - nm)
                                fg = T.exp(logf - nm)
                                c[b, j] = fg * c[b, j] + ig * zv
                                n[b, j] = T.max(fg * n[b, j] + ig, 1.0)
                                hid[b, j] = og * c[b, j] / n[b, j]
                                m[b, j] = nm
                            else:
                                zv = 2.0 / (1.0 + T.exp(-2.0 * gz)) - 1.0
                                og = 1.0 / (1.0 + T.exp(-go))
                                c[b, j] = 1.0 / (1.0 + T.exp(-gf)) * c[b, j] + 1.0 / (1.0 + T.exp(-gi)) * zv
                                hid[b, j] = og * (2.0 / (1.0 + T.exp(-2.0 * c[b, j])) - 1.0)
                            if valid:
                                if out_mode == 0:
                                    OUT[0, row, step, head, col] = T.cast(hid[b, j], snapshot)
                                    OUT[1, row, step, head, col] = T.cast(c[b, j], snapshot)
                                    if slstm:
                                        OUT[2, row, step, head, col] = T.cast(n[b, j], snapshot)
                                        OUT[3, row, step, head, col] = T.cast(m[b, j], snapshot)
                                elif out_mode == 1:
                                    OUT[row, step, head, col] = T.cast(hid[b, j], snapshot)
                                else:
                                    if step == TT - 1:
                                        OUT[row, 0, head, col] = T.cast(hid[b, j], snapshot)
                            if t + 1 < NSTEPS:
                                XBUF[t % 2, g, col, b] = T.cast(hid[b, j], dtype)
                    if t + 1 < NSTEPS:
                        T.sync_grid()

                if active:
                    for b, j in T.Parallel(BM, S):
                        row = bt * BM + b
                        col = s * S + j
                        if row < B:
                            FINAL[0, row, 0, head, col] = hid[b, j]
                            FINAL[1, row, 0, head, col] = c[b, j]
                            if slstm:
                                FINAL[2, row, 0, head, col] = n[b, j]
                                FINAL[3, row, 0, head, col] = m[b, j]
                T.sync_grid()

    return main
'''


def _source(reg_k):
    nr = reg_k // KC
    ind = " " * 12
    alloc = "\n".join([f"{ind}rr{i} = T.alloc_fragment((4 * S, KC), dtype)" for i in range(nr)]
                      + [f"{ind}hr0 = T.alloc_shared((KC, BM), dtype)", f"{ind}hr1 = T.alloc_shared((KC, BM), dtype)"])
    ind = " " * 20
    load_r = "\n".join(f"{ind}for q, kk in T.Parallel(4 * S, KC):\n{ind}    rr{i}[q, kk] = R[q // S, head, s * S + q % S, {i} * KC + kk]"
                       for i in range(nr)) or f"{ind}pass"
    ind = " " * 24
    gemm_r = "\n".join(f"{ind}T.copy(XBUF[par, g, {i} * KC:{i + 1} * KC, 0:BM], hr{i % 2})\n"
                       f"{ind}T.fence_proxy_async()\n{ind}T.sync_threads()\n"
                       f"{ind}T.gemm(rr{i}, hr{i % 2}, gates, clear_accum={i == 0})" for i in range(nr)) or f"{ind}pass"
    return _TEMPLATE.format(S=S, KC=KC, THREADS=THREADS, alloc=alloc, load_r=load_r, gemm_r=gemm_r,
                            clear_s="cc == 0" if nr == 0 else "False")


def _module(reg_k):
    src = _source(reg_k)
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    cache = Path(os.environ.get("FRNN_TL_STREAM_CACHE", os.environ.get("TMPDIR", "/tmp") + "/frnn_tl_stream"))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"tl_stream_{digest}.py"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        tmp.replace(path)
    spec = importlib.util.spec_from_file_location(f"tl_stream_{digest}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(batch, total_steps, heads, width, reg_k, bm, *, sms, cell="lstm", dtype="float16", mode="full",
          snapshot="float32"):
    from .tilelang_cluster import MODES

    if not supported_stream(width, reg_k, bm, sms):
        raise ValueError("unsupported (D, reg_k, BM) for the streamed-h grid path")
    k = width // S
    tiles = (batch + bm - 1) // bm
    groups = tiles * heads
    resident = max(1, min(groups, sms // k))
    slots = (groups + resident - 1) // resident
    slstm = cell == "slstm"
    return _module(reg_k).make(batch, total_steps, heads, width, reg_k, bm, k, resident, slots, tiles, groups, slstm,
                               dtype, snapshot, MODES[mode], 4 if slstm else 2)


def kernel(batch, total_steps, heads, width, reg_k, bm, *, cell, dtype, mode, snapshot, sms):
    import tilelang

    key = (sms, batch, total_steps, heads, width, reg_k, bm, cell, dtype, mode, snapshot)
    if key not in _KERNELS:
        func = build(batch, total_steps, heads, width, reg_k, bm, sms=sms, cell=cell, dtype=dtype, mode=mode,
                     snapshot=snapshot)
        # No automatic warp specialization: it adds a producer warpgroup that never reaches the per-step grid barrier
        # (job "bm64 v2" deadlocked at its first case).
        _KERNELS[key] = tilelang.compile(func, target={"kind": "cuda", "arch": "sm_90a"},
                                         pass_configs={"tl.disable_warp_specialized": True})
    return _KERNELS[key]
