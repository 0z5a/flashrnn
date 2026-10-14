# R tiling: prepared execution (M1) and TileLang cluster spike (M3)

Job `47298769` (name 62), one ACC node `as07r3b04`, four H100 lanes, 4 min 34 s.
Receipts and stage JSON: `evidence/flashrnn2-rtiled-m13-20261014/job47298769/`.
The complete raw archive stays on MN5 with an independent re-hash check
(`ARCHIVE_VERIFIED`, 41 files). Timings here are exploratory and unpaired.

## M1: `mma_rtiled` through `PreparedSequence`

`KernelConfig(family="mma_rtiled", slice_width=S, sync="grid")` runs the R-tiled
kernel inside `PreparedSequence` with a preallocated exchange buffer and per-chunk
barrier counters (zeroed once per run, inside a captured graph). An unknown shape
with D > 128 now selects `default_config(D)` (S=64 up to D=256, else S=32,
eight warps) instead of failing; D ≤ 128 keeps MMA16 / four warps.

48 cases (LSTM/sLSTM × FP16/BF16 × D ∈ {128, 192, 256, 384, 512, 768} ×
(B,T,H) ∈ {(3,37,2), (40,64,1)}), each in full, hidden and last output modes:

| Check | Result |
| --- | --- |
| Prepared eager = standalone R-tiled call, bitwise (output and FP32 carry) | 144/144 |
| Prepared ragged chunks = one call, bitwise | 144/144 |
| CUDA Graph capture + 20 replays = one call, bitwise; error flag 0 | 144/144 |

So a cooperative launch inside CUDA Graph capture works on this stack
(Triton 3.6.0, driver 595.91).

## M3: TileLang 0.1.15 cluster + DSMEM spike

`flashrnn/flashrnn2/tilelang_rtiled.py`: one cluster of k ≤ 16 CTAs per
(batch tile, head); each CTA writes its rounded h slice into every cluster
peer's next-step shared buffer (`copy_cluster(dst_block=r)`, lowered to
`cluster.map_shared_rank` stores) and one `cluster_sync` orders the step. The
spike returns the hidden history and the FP32 final carry for one full logical
call; it does not yet implement chunk continuation or full-state output.

| Check | Result |
| --- | --- |
| FP64 oracle (`fixture_math`, `CELL_BUDGET`), hidden and carry; D ∈ {128, 192, 256, 384, 512}, k = 2–16; FP16/BF16; sLSTM zero/mixed/warm | LSTM 24/24, sLSTM 72/72 PASS; worst normalized error 0.108 |
| Bitwise against Triton R tiling on the same inputs | 0/96 (different MMA lowering); max hidden difference 4.9e-4 |

Exploratory per-step time (LSTM FP16, H=1, T=256, both arms allocate per call):

| D | S (k) | B | TileLang µs/step | Triton µs/step |
| ---: | --- | ---: | ---: | ---: |
| 128 | 64 (2) | 16 / 64 | 4.58 / 5.20 | 3.30 / 3.71 |
| 192 | 64 (3) | 16 / 64 | 5.86 / 6.31 | 4.15 / 5.41 |
| 256 | 64 (4) | 16 / 64 | 7.04 / 7.63 | 3.49 / 4.59 |
| 384 | 32 (12) | 16 / 64 | 8.53 / 9.04 | 3.13 / 4.62 |
| 512 | 32 (16) | 16 / 64 | 11.17 / 11.40 | 4.53 / 5.42 |

The spike is numerically qualified but slower than the Triton L2-barrier path,
increasingly so with k. Its DSMEM write is a SIMT loop of remote stores to all k
peers, and it uses neither TMA multicast nor WGMMA, so this measures the spike,
not TileLang's ceiling.

## Prepared-graph candidate screen

Captured `PreparedSequence` graphs, FP16, H=1, T=256, hidden output, 12 rotating
rounds; median µs per step (fastest arm):

| D | LSTM B=16 | LSTM B=64 | sLSTM B=16 | sLSTM B=64 |
| ---: | --- | --- | --- | --- |
| 64 | single MMA8 1.35 | single MMA8 1.27 | — | — |
| 96 | R S32/w4 2.39 (single MMA8 2.44) | R S32/w4 2.39 (2.45) | — | — |
| 128 | R S32/w4 2.43 (single MMA8 2.48) | R S32/w4 2.81 (2.99) | — | — |
| 192 | R S32/w4 2.81 | R S64/w8 4.03 | R S32/w4 2.55 | R S32/w4 3.83 |
| 256 | R S32/w4 3.04 | R S64/w8 4.37 | R S64/w4 3.01 | R S16/w4 4.22 |
| 384 | R S32/w4 3.05 | R S32/w4 4.54 | R S32/w8 3.90 | R S32/w8 5.24 |
| 512 | R S32/w4 4.50 | R S32/w4 5.31 | R S32/w4 4.54 | R S32/w8 5.44 |
| 768 | R S32/w4 5.38 | R S32/w4 5.67 | R S32/w4 5.34 | R S32/w4 5.62 |

These screens feed candidate selection; the D ≤ 128 switch point and every
speedup claim come only from the paired E2–E4 protocol.
