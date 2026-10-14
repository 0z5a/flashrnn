# R tiling: TileLang cooperative-grid path for D = 768 (M7)

D = 768 needs k = D/S = 24 CTAs at S = 32 (R slice 192 KiB of shared memory)
or 48 at S = 16, beyond the 16-CTA thread-block-cluster limit that the M6
TileLang path relies on. M7 adds a grid-synchronized TileLang path and qualifies
and times it at D = 768.

Source: `build_grid` in `flashrnn/flashrnn2/tilelang_cluster.py` (commit
`da2deba`). Per CTA it does the same work as the cluster kernel: four gate
slices of R as WGMMA operand A, hᵀ[D × 16] as operand B in shared memory, and
the same epilogue. Only the h exchange differs:

* each CTA writes its rounded h slice into a double-buffered global buffer
  `XBUF[t % 2, group]`;
* one `T.sync_grid()` per step (`cooperative_groups::this_grid().sync()`);
* every CTA then loads its group's full hᵀ for the next step (16-byte vector
  loads).

A CTA can overwrite `XBUF[t % 2]` again only at step t + 2, after the barrier of
step t + 1, which every reader of step t has reached. TileLang's
`PersistThreadblock` pass marks the function `use_cooperative_groups`, and TVM's
CUDA module then launches it with `cuLaunchCooperativeKernel`. The compiled
grid kernels carry the `use_cooperative_launch` tag and cluster kernels do not;
the dynamic shared memory attribute (229,376 B at S = 32) is set before every
launch branch. A grid holds k × min(groups, ⌊132/k⌋) CTAs, one per SM. Groups
beyond that run in slots; every CTA executes the same barriers, idle or not.
`KernelConfig(family="tilelang", sync="grid", slice_width=16|32)`,
`PreparedSequence`, the public `recurrence(..., sync="grid")`, the timing tool
and the qualification tool (`--sync grid`) carry the choice through.

## Batch

Job 47305834 (name 88), one ACC node `as04r5b14`, 15 min 42 s, `COMPLETED`;
driver `ALL_STARTS_COMPLETED`, 100/100 starts; no surviving owned processes;
raw archive verified (`m7-47305834.tar.gz`, SHA256
`e67bbffecff4f9f7af847bd707742828faf185f76932e3f6fac78ebc341964da`, 687 files).
The source was staged with `git archive da2deba` (274/274 files equal to the
commit, plus the pinned `fixture_math.py`); seal READY with 291 sealed files;
login-node dry-runs passed (prefix control, Triton launcher, 52 TileLang
variants). `analysis/` was computed after the archive and is reproducible from
the archived `science/` tree. Evidence:
`evidence/flashrnn2-rtiled-m7-20261014/job47305834/`.

## Qualification (phase 0)

20 configurations, all with `--sync grid`: D = 768 at S = 16 and 32, and
D = 256 (S = 16, 32) and D = 512 (S = 32), where the cluster path also exists,
each for LSTM/sLSTM × FP16/BF16. The first (LSTM, BF16, D = 768, S = 32, the
largest shared-memory footprint) ran alone.

| Gate | Result |
|---|---|
| E1, fixture_math `CELL_BUDGET`, FP64 oracle; history, FP32 carry, hidden, last | 20/20 configurations, 80/80 cases; worst normalized error 0.0034 (BF16 sLSTM D512), 0.0029 at D = 768 |
| Protocol, bitwise (per-step launches, ragged prepared chunks, public chunked calls, hidden/last = full, 20 repeats, side stream, CUDA Graph replay and bound initial state) | 720/720 |
| Stress, 50 bit-identical repeats, (64,128,4) and (128,64,8) | 40/40; at D = 768 S = 32 the grids ran 4 and 13 slots, at S = 16 8 and 32 slots |
| Grid = cluster, bitwise (D = 256 S = 16/32, D = 512 S = 32; full FP32, full input dtype, hidden, last; three shapes) | 144/144, maximum difference 0 |
| compute-sanitizer memcheck / synccheck (LSTM FP16 D768 S32, small case) | exit 0, `ERROR SUMMARY: 0 errors` |
| compute-sanitizer racecheck (same case, non-blocking) | exit 0, `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`; counted correctly this time |

Distance to the Triton default at D = 768 (informational): history at most
5.5e-5 (FP16 LSTM), 2.9e-4 (FP16 sLSTM), 4.3e-4 (BF16 LSTM), 2.2e-3 (BF16 sLSTM).

## Selection and timing

Split-0 selection per shape among the six M0-qualified Triton configurations at
D = 768 and the two qualified TileLang grid configurations. Three roles were
selected independently: `policy` (all eight), `triton` (Triton only),
`tilelang` (TileLang only). Frozen table SHA256
`53afc336a6fed08b95de3e25ebd6aa514fd19d0761739719bf723492139d1708`. Split-1
timing, 5 starts × 20 ABBA/BAAB blocks, 10000-draw hierarchical bootstrap;
speedup = baseline time / candidate time.

The policy selected TileLang grid for all 10 shapes: S = 16 at B ≤ 32 and
S = 32 at B = 64. The best Triton configuration was S = 32 with four warps for
all 10 shapes.

| Experiment | Rows | Range | Geomean | Interval above 1 |
|---|---|---|---|---|
| Kernel, policy vs default (Triton S32, 8 warps) | 4 | 1.347–2.064 | 1.669 | 4/4 |
| Classifier, policy vs default | 6 | 1.301–1.799 | 1.505 | 6/6 |
| Kernel, TileLang grid vs best Triton | 4 | 1.151–1.730 | 1.410 | 4/4 |
| Classifier, TileLang grid vs best Triton | 6 | 1.050–1.557 | 1.269 | 6/6 |

| Shape | Default ms | Best Triton ms | Policy ms | vs default | vs best Triton |
|---|---|---|---|---|---|
| LSTM kernel B16 T1024 | 6.465 | 5.415 | 3.311 | 1.95× | 1.64× |
| LSTM kernel B64 T1024 | 6.724 | 5.734 | 4.983 | 1.35× | 1.15× |
| sLSTM kernel B16 T1024 | 6.449 | 5.393 | 3.123 | 2.06× | 1.73× |
| sLSTM kernel B64 T1024 | 6.680 | 5.660 | 4.674 | 1.43× | 1.21× |
| LSTM B1/C8/T32 | 7.142 | 5.800 | 5.507 | 1.30× | 1.05× |
| LSTM B16/C32/T32 | 2.012 | 1.694 | 1.368 | 1.47× | 1.24× |
| LSTM B32/C64/T128 | 7.544 | 6.486 | 4.331 | 1.74× | 1.50× |
| sLSTM B1/C8/T32 | 7.426 | 6.158 | 5.689 | 1.30× | 1.08× |
| sLSTM B16/C32/T32 | 2.057 | 1.754 | 1.375 | 1.49× | 1.28× |
| sLSTM B32/C64/T128 | 7.648 | 6.611 | 4.248 | 1.80× | 1.56× |

(Each comparison has its own paired timing, so the policy column is taken from
the policy-vs-default rows and the best-Triton column from the other rows. Both
rows ran the same TileLang configuration, and their candidate times agree within
0.5%.)

## SM footprint

The selected configurations trade SMs for latency. One cooperative grid
occupies k × resident groups SMs (one CTA per SM) for the whole call:

| Selected | k | Groups at B = 1/16, 32, 64 | SMs |
|---|---|---|---|
| S = 16 (B ≤ 32) | 48 | 1, 2 | 48, 96 |
| S = 32 (B = 64) | 24 | 4 | 96 |

That is the right choice for one request stream on an otherwise idle GPU, and
the open question for high concurrency; the footprint is recorded per entry in
the table file (`sm_footprint`).

## Registry

`flashrnn/flashrnn2/tables/h100-rtiled-m7-20261014.json` holds the 10 D = 768
entries with the same identity as the M6 table. `ExecutionPolicy.from_json`
now accepts several files and merges them only if they share one identity and
no shape key:

```python
ExecutionPolicy.from_json(TABLES / "h100-rtiled-m6-20261014.json", TABLES / "h100-rtiled-m7-20261014.json")
```

gives 50 exact-shape entries (D = 192–768). As before, the table takes effect
through `ExecutionPolicy.select → KernelConfig → PreparedSequence(config=…)`.

## Not claimed

* Shapes outside the timed set; BF16 is qualified but not timed.
* Throughput under concurrent requests (the SM footprint above is the input
  for that work).
* Other GPUs, drivers or compiler versions; backward; multi-GPU; model quality.
* Sanitizer coverage beyond one small shape.
