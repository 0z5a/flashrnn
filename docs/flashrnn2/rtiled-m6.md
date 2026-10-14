# R tiling: full-contract TileLang backend and the exact-shape policy table (M6, G4)

G4 asks for one qualified fallback backend under the same contract as the Triton
path, registered in an exact-shape table. M5 measured the TileLang cluster kernel
as the fastest of the three R-tiled implementations at D = 192–512, so M6 makes
it a full-contract backend and qualifies it.

Source: `flashrnn/flashrnn2/tilelang_cluster.py` (TileLang 0.1.15, CUDA 12.8,
gated by `FRNN_TILELANG=1`). It keeps the M5 algorithm (cluster of k = D/S CTAs,
WGMMA with the R slice as operand A, `copy_cluster` of each h slice to every peer
with a remote mbarrier, double buffering, no per-step cluster barrier) and adds
the full contract:

* output modes full / hidden / last; FP32 final carry; snapshot in the input
  dtype or FP32;
* chunk continuation: the step count is a runtime loop bound, the kernel takes
  `BASE` (global step of the first local step) and `START` (whether this launch
  begins the logical call), and reads the FP32 carry of the previous chunk;
* sLSTM full-call normalizer initialization only on the first step of the logical
  call (`START != 0`, `t == 0`, zero-normalizer flag);
* the `tilelang` family in `KernelConfig` (batch tile 16, slice width 16/32/64,
  `sync="cluster"`), `PreparedSequence` (prepared chunks and CUDA Graph capture),
  and `ExecutionPolicy`, which refuses a table entry that names TileLang on a
  runtime without TileLang instead of silently switching backend.

Qualification tool: `tools/rtiled_m6_tilelang.py` (stages e1, protocol, stress,
sanity). Batch driver: `run_m6.py` in the batch control directory.

## v1: job 47304358 (name 38) — stopped by a harness bug

One ACC node `as04r3b01`, 9 min 55 s, `COMPLETED`; raw archive verified
(`m6-47304358.tar.gz`, SHA256 `b1a5b3b0dd4de91535a0153c34e874fbda082c31b49bc40d930435ec000019a0`,
1274 files). Evidence: `evidence/flashrnn2-rtiled-m6-20261014/job47304358/`.

* **TileLang qualification: harness ERROR, not a kernel result.** The first
  configuration (LSTM, BF16, D = 256, S = 16) passed E1 (all four outputs under
  `CELL_BUDGET`, worst normalized error 1.2e-3) and the 50-repeat stress stage
  (up to 1024 CTAs, bit-identical). The protocol stage raised in the tool: for
  T = 17 the ragged chunk list `[1, 3, 13]` already summed to T, the appended
  remainder was 0, and `PreparedSequence` rejected it. The case before it
  (T = 33) had run through the whole protocol path, but its checks were lost
  because the tool wrote the protocol rows only after the loop. By design the
  other 39 configurations were not started.
* **Sanitizer: not executed.** The batch's `runtime.sh` ended in
  `exec python3 -B "$@"` and ignored `FRNN_PREFIX`, so the two "sanitizer" tasks
  ran the sanity stage without `compute-sanitizer` (their logs contain no
  `ERROR SUMMARY`). Their bit-identical sanity results are not sanitizer results.
* **Phases 1–2 ran Triton-only.** The driver skipped the remaining qualifications
  but still ran selection and timing; with no TileLang configuration available,
  every policy entry became `mma_rtiled`. This is not G4 timing and its frozen
  table is not shipped as the registry table. It is a valid side result about
  Triton R-tiled tuning, below.

### Side result: Triton exact-shape tuning vs the conservative default

Split-0 selection among the M0-qualified Triton configurations, frozen table
(SHA256 `5fa0e882d21a4c6dccb06e48ae96782feaf1aab55375c201803f03161cb96b75`),
split-1 timing, 5 starts × 20 ABBA/BAAB blocks, 200/200 starts passed, 10000-draw
hierarchical bootstrap. Baseline = `default_config(D)` (eight warps, S = 64 for
D ≤ 256, else 32). Speedup = default time / selected time.

* 7 of 40 rows selected the default itself (LSTM D256 B64 kernel; sLSTM D384 at
  all six shapes; sLSTM D512 B64 kernel). They act as built-in A/A controls:
  0.9977–1.0010, every 95% interval contains 1.
* The other 33 rows are all faster with the interval above 1: 1.0025–1.3743,
  geomean 1.097 (kernel level 12 rows, geomean 1.111; classifier level 21 rows,
  geomean 1.090). The largest gains are four warps at S = 32 for sLSTM D192
  (1.30–1.37×) and LSTM D384 (1.17–1.23×).

`analysis/` was computed after the archive was written, so it is not in the
archive manifest; it is reproducible from the archived `science/` tree with
`tools/rtiled_m4_analyze.py`.

### Fixes for v2

* `ragged_chunks(steps)` builds the uneven chunk list with a positive tail
  (checked for T = 1…299; the old construction reproduces the zero tail at
  T = 17). Protocol rows are written after every case. The distance to the
  Triton default is recorded but cannot turn a qualification into ERROR.
* `runtime.sh` applies `FRNN_PREFIX`. A login-node control
  (`dryrun_prefix.py`) requires the v2 runtime to print the prefix marker and the
  v1 runtime not to; both held.
* The sanitizer runs use `--target-processes application-only
  --error-exitcode 99`. memcheck and synccheck block: each needs exit 0, at least
  one `ERROR SUMMARY` line and every such line equal to
  `========= ERROR SUMMARY: 0 errors`, plus the bitwise sanity gate. racecheck
  is recorded but does not block (its coverage of TMA/cluster copies may be
  partial).
* After phase 0 the driver stops (`PHASE0_GATE_STOP`) if the first configuration
  ERRORs, a blocking sanitizer is not clean, or no FP16 TileLang configuration
  qualified, so a failed qualification can no longer produce a Triton-only
  "G4" table.

## v2: job 47304776 (name 39)

One ACC node `as03r3b16`, 19 min 50 s, `COMPLETED`; driver status
`ALL_STARTS_COMPLETED`, 200/200 starts; no surviving owned processes; raw
archive verified (`m6-47304776.tar.gz`, SHA256
`2c6c68107a7f706115a5e598bd4c4ce938df0b4b6371a57ce88ac17adc8268fc`, 1407 files).
Prepared from commit `0863d39`, seal READY (289 sealed files); all three
login-node dry-runs passed (prefix control, Triton launcher build, 28 TileLang
variants). Evidence: `evidence/flashrnn2-rtiled-m6-20261014/job47304776/`.

### Qualification (phase 0)

Configurations (cell × dtype × (D, S)): LSTM/sLSTM × FP16/BF16 ×
{(128,32), (128,64), (192,16), (192,32), (192,64), (256,16), (256,32), (256,64),
(384,32), (512,32)}; the first (LSTM, BF16, 256, 16) ran alone.

| Gate | Result |
|---|---|
| E1, fixture_math `CELL_BUDGET`, FP64 oracle; history, FP32 carry, hidden, last; LSTM warm, sLSTM zero/mixed/warm; (B,T,H) = (3,17,2), (16,128,1) | 40/40 configurations, 160/160 cases; worst normalized error 0.0042 (BF16 sLSTM D192) |
| Protocol, bitwise: per-step launches, prepared ragged chunks, public chunked calls, hidden/last = full, 20 repeats, side stream with producer, CUDA Graph replay, Graph tracks the bound initial state; (B,T,H) = (1,33,1), (3,17,2), (33,40,2); sLSTM zero and nonzero normalizer | 1440/1440 checks |
| Stress: (64,128,4) and (128,64,8), 50 bit-identical repeats, up to 1024 CTAs | 80/80 |
| compute-sanitizer memcheck (LSTM FP16 D256 S16, small case) | exit 0, `ERROR SUMMARY: 0 errors`, sanity bitwise |
| compute-sanitizer synccheck (same case) | exit 0, `ERROR SUMMARY: 0 errors`, sanity bitwise |
| compute-sanitizer racecheck (same case, non-blocking) | exit 0, `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |

The driver receipt labels racecheck `FAIL`: its rule looked for an
`ERROR SUMMARY` line, which racecheck does not print (it prints the
`RACECHECK SUMMARY` above). The raw log is clean; the receipt is left as
written. The sanitizer runs cover one small shape; racecheck's coverage of
the TMA/cluster bulk copies may be partial. TileLang still emits its
`[ThreadSync] Hoisting sync out of an if ... Condition: tx == 0` warning for
the single-thread copy branch; none of the gates above observed a race.

The distance to the Triton R-tiled default (informational) is at most
1.0e-4 (FP16 LSTM), 3.6e-4 (FP16 sLSTM), 5.2e-4 (BF16 LSTM) and 2.6e-3
(BF16 sLSTM) in the history, never bitwise (different accumulation order). For
D = 128 the default is the single-CTA kernel, so no R-tiled distance is recorded.

### Policy table (phase 1) and timing against the default (phase 2)

Selection on split 0 among the M0-qualified Triton configurations and the 40
qualified TileLang configurations; frozen table SHA256
`3b1b8167b2e61b219397cc4d40e5546fbb8006c7054e0fb57af21cd6d51f52fd`. The policy
entry for each shape is the selected candidate. 37 entries select TileLang (S = 16
for D ≤ 256, S = 32 for D = 384/512), 3 select Triton R tiling with four warps,
S = 32: LSTM D384 and D512 and sLSTM D512, all at B = 1 (one batch tile).

Timing on split 1: every start served the candidate through
`ExecutionPolicy(policy_table, identity=policy_identity).select(...,
identity=runtime_identity())` and recorded it (`served_by_policy`). Baseline =
`default_config(D)`. Speedup = default time / policy time; 95% interval from
the hierarchical bootstrap.

| Level | Rows | Policy | Speedup range | Geomean | Interval above 1 |
|---|---|---|---|---|---|
| Kernel (B = 16/64, T = 1024) | 16 | TileLang | 1.371–2.331 | 1.679 | 16/16 |
| Classifier (B = 1/16/32, T = 32/128) | 21 | TileLang | 1.012–1.976 | 1.318 | 21/21 |
| Classifier | 3 | Triton, 4 warps | 1.078–1.180 | 1.132 | 3/3 |
| All | 40 | | 1.012–2.331 | 1.435 | 40/40 |

Per-row values: `job47304776/analysis/m4-summary.csv`. The largest gains are
sLSTM D192 (kernel 2.24–2.33×, classifier B32 1.98×); the smallest is sLSTM
D384 B1 (1.012×, interval 1.006–1.020).

These numbers compare against the conservative default, not against the best
Triton configuration. M5 (job 47303121) measured the TileLang kernel against
selected Triton configurations: kernel 1.24–1.85×, classifier 1.05–1.52×.

### Registry

The frozen table ships as
`flashrnn/flashrnn2/tables/h100-rtiled-m6-20261014.json` (format
`flashrnn2-execution-policy-table/1`: identity, 40 exact-shape entries, scope,
provenance and the per-row timing) and loads with
`ExecutionPolicy.from_json(TABLES / "h100-rtiled-m6-20261014.json")`. Its
identity is `{gpu: "NVIDIA H100", capability: [9, 0], triton: "3.6.0",
tilelang: "0.1.15"}`; `select` rejects any other identity, and a TileLang entry
on a runtime without TileLang raises instead of switching backend. Shapes
outside the table keep `default_config(D)`. `tests/test_execution_policy.py`
checks the shipped file (entries, identity binding, fallback, format).

### Not claimed

* Shapes, dtypes and head counts outside the table; BF16 is qualified but not
  timed or registered.
* D = 768 and other widths needing k > 16 CTAs per cluster (TileLang's cluster
  limit); they stay on Triton R tiling.
* Other GPUs, drivers or compiler versions; backward; multi-GPU; model quality.
* Sanitizer coverage beyond the one small shape.
