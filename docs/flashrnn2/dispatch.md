# Table-served one-call recurrence (`flashrnn2.dispatch`)

Until now the exact-shape tables took effect only through
`PreparedSequence(config=…)`, and the public one-call recurrence functions ran
whatever configuration the caller passed. `flashrnn/flashrnn2/dispatch.py`
(development commit 63444a5) adds a one-call path that serves the table.

* `select_config(cell, dtype, batch, steps, heads, width, *, policy=None)`
  returns `(KernelConfig, source)`, where `source` is `"table"` or `"default"`.
  The default policy is the packaged H100 table set (M6 + M7, 50 exact shapes).
  It is served only when this runtime's identity (GPU name, capability, Triton
  and TileLang versions) equals the tables' identity. Otherwise every shape gets
  `default_config` (single CTA for D ≤ 128, Triton R tiling above). A policy
  passed explicitly under the wrong identity raises.
* `recurrence(wx, recurrent, bias, initial, cell, *, sequence_context=None,
  output_mode="full", snapshot_dtype=None, config=None, policy=None)` runs one
  logical call, or one chunk of one, on the backend the configuration names:
  * `triton_persistent` for single-CTA mma with BM 16;
  * `triton_rtiled`;
  * `tilelang_cluster` (cluster or grid).

  The contract is the backends' own: full/hidden/last output, FP32 compute
  carry, continuation through a `SequenceContext`, and the sLSTM normalizer rule
  at the logical call's first step. A TileLang entry on a runtime without
  TileLang raises; nothing switches backend silently.
* Table keys use the logical call's total steps, so every chunk of a chunked
  call must run the configuration selected for the whole call. Call
  `select_config(..., steps=<total>)` once and pass the result as `config` to
  every chunk. A continuation chunk without a config is rejected.
  Configurations without a one-call path (SIMT, MMA BM 32) raise and point to
  `PreparedSequence`.

## Contract checks (run dispatch check)

`tools/dispatch_check.py` on one H100 node, 2 min 28 s, `ALL_PASS`; archive
verified (38 files). Evidence: `evidence/flashrnn2-rtiled/dispatch/`.

**Unit (passes on CPU as well):**
* a table hit returns the table entry with source "table";
* a miss returns `default_config`;
* an identity mismatch gives defaults;
* an explicit policy under a wrong identity raises;
* a continuation chunk without a config raises;
* a configuration without a one-call path raises.

**Contract, per shape:**
* the served configuration equals the table entry or `default_config`;
* dispatch is bitwise equal to the backend called directly with that
  configuration (full FP32, full input dtype, hidden, last, and the FP32 carry);
* dispatch is bitwise equal to `PreparedSequence(config)` (full FP32, hidden,
  last);
* public chunked calls with the logical call's configuration (chunks 1, 3, 13
  and the rest) are bitwise equal to one call;
* fixture_math `CELL_BUDGET` E1 holds for history, carry, hidden and last
  (sLSTM: zero, mixed and warm normalizer).

| Shape | Source | Served | Worst E1 (share of budget) | Result |
|---|---|---|---|---|
| LSTM FP16 B16 T128 D64 | default | single-CTA mma, 4 warps | 0.0004 | PASS |
| sLSTM FP16 B32 T128 D128 | default | single-CTA mma | 0.0007 | PASS |
| sLSTM BF16 B8 T40 H2 D128 | default | single-CTA mma | 0.0031 | PASS |
| LSTM FP16 B16 T1024 D192 | table | TileLang cluster S16 | 0.0003 | PASS |
| sLSTM FP16 B32 T128 D256 | table | TileLang cluster S16 | 0.0005 | PASS |
| LSTM FP16 B48 T64 D256 | default | Triton R tiling S64, 8 warps | 0.0002 | PASS |
| LSTM BF16 B32 T128 D256 | default | Triton R tiling S64, 8 warps | 0.0018 | PASS |
| LSTM FP16 B1 T32 D384 | table | Triton R tiling S32, 4 warps | 0.0002 | PASS |
| LSTM FP16 B16 T1024 D512 | table | TileLang cluster S32 | 0.0003 | PASS |
| LSTM FP16 B16 T1024 D768 | table | TileLang grid S16 | 0.0002 | PASS |
| sLSTM FP16 B64 T1024 D768 | table | TileLang grid S32 | 0.0004 | PASS |
| sLSTM BF16 B16 T64 D768 | default | Triton R tiling S32, 8 warps | 0.0029 | PASS |

D ≤ 128 keeps the single-CTA kernel (E2 in `rtiled-m4.md`: the single CTA was
faster in 24/24 comparisons). The packaged tables cover FP16 only, so BF16 and
unmeasured shapes are served `default_config` until a table is measured for
them.

## Not claimed

* Timing: the dispatched path runs kernels that were already timed.
* Backward, multi-GPU, other GPUs or compiler versions.
