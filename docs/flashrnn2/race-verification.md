# Race verification of the cross-CTA exchanges

compute-sanitizer's racecheck tracks shared-memory hazards inside a CTA; it does not model TMA/bulk copies into a
peer CTA's shared memory (DSMEM), mbarrier transaction counts, or global-memory exchanges ordered by a grid barrier.
Its clean reports on the TileLang cluster kernel (M6) therefore cover less than the exchange itself. This round adds
a verification build that checks every exchanged operand directly and shows, with deliberately broken variants,
that the check would see a race.

## Instrumentation (compiled out by default)

Every R-tiled kernel takes a race-verification switch (`_verify`; Triton: a `VERIFY` constexpr bitmask; TileLang: a
Python-level option, with the zero-normalizer flag argument widened to four integers only in that build):

* **content check**: after each wait, the consumer compares the whole received operand (all D x BM values of its
  group) with what the senders stored for that step (the FP32 hidden output, rounded the same way; for the
  backward, the stored FP32 dG); a second comparison after the MMAs detects an early overwrite of the buffer still
  in use. Mismatches are counted in global memory.
* **delay injection**: pseudo-random sleeps of 0-4 us from a runtime seed, at the producer before it publishes
  (skewed per warp on TileLang, so a missing block barrier shows) and at the consumer after its wait.
* **broken variants** (negative controls): single buffer instead of double; skipped wait (cluster; the last step
  still waits so that no copy is in flight into an exited CTA) or skipped barrier (grid, Triton); arrive before the
  stores; no block barrier before thread 0 sends (cluster).
* **positive control**: the two-warpgroup cluster kernel, excluded after 0/12 bitwise twins.

With the switch off, the generated code is unchanged: identical PTX for the Triton forward and backward kernels
(`tools/verify_ptx_unchanged.py`, DWARF sections excluded) and identical CUDA source for the TileLang cluster and grid
kernels, including BM > 16 and two-warpgroup variants (`tools/verify_tl_source_unchanged.py`), both PASS on the
login node; every verification variant compiles.

## Rule (pre-registered) and runs

`tools/race_verify.py`, 18 tasks (Triton forward D = 256/768, Triton backward D = 256/768, TileLang cluster D = 256
BM 16/32, TileLang grid D = 768 BM 16 one warpgroup / BM 64 two warpgroups, LSTM and sLSTM, B = 40, T = 48, plus the
positive control), 21 runs per variant (no delay and 20 seeds):

* a correct kernel passes when every run counts zero violations and its outputs are bitwise equal to the
  uninstrumented reference;
* a broken variant is detected when violations are counted in at least 80% of its runs, and no run may change the
  output without counting a violation.

## Round 1 (run race verification, 4 min 52 s, archive verified, 85 files)

* **Every correct kernel is clean.** 16 configurations (Triton forward and backward at D = 256/768; TileLang cluster
  D = 256 BM 16/32; TileLang grid D = 768 BM 16 one warpgroup and BM 64 two warpgroups; LSTM and sLSTM), 21 runs each
  with and without delays: zero violations, outputs bitwise equal to the uninstrumented kernel in all 336 runs.
* **Positive control.** The two-warpgroup cluster kernel was flagged in 21/21 runs for both cells (outputs differ
  in all of them): what it consumes differs from what its peers stored, so its defect is in the exchange, not in the
  cell arithmetic.
* **Controls that removed or reordered synchronization were caught:** skipped barrier (Triton forward/backward,
  TileLang grid) 21/21 everywhere; arrive before the stores 19/21-21/21 (the Triton backward's outputs happened to
  stay equal, yet every run was flagged); skipped wait (cluster, LSTM) 21/21. No run anywhere changed its output
  without being flagged.
* **Weak or invalid controls under the pre-registered 80% rule:**
  * single buffer: flagged in 0-21 of 21 runs depending on the configuration, never with a changed output: with
    0-4 us delays the hazard window rarely opened (when it did, the post-MMA check saw the overwrite);
  * cluster without the explicit block barrier: never flagged because it is not broken -- TileLang's generated
    source still contains a `__syncthreads()` at that point (it inserts one for the shared-memory write/read itself);
  * sLSTM cluster skipped wait: the variant corrupted the CUDA context ("unspecified launch failure") after 7 and 1
    flagged runs, and the later variants of the same process could not run.

## Round 2 (run race verification round 2, 8 min 2 s, archive verified, 229 files)

Changes, written into the rule before submission: delays 0-65 us; the cluster consumer delay after the wait (inside
the window a single buffer exposes); every variant in its own process (a run that corrupts the CUDA context ends only
that variant); the cluster no-barrier variant reported only. 66 tasks, 65 PASS under the rule.

| Path (D, configuration; LSTM and sLSTM) | correct | single buffer | skipped barrier / wait | arrive before store |
| --- | --- | --- | --- | --- |
| Triton forward, D 256 S32 / D 768 S32 | 0/21 flagged, bitwise | 20/21 | 21/21 | 21/21 |
| Triton backward, D 256 S32 / D 768 S16 | 0/21 flagged, bitwise | 20/21 | 21/21 | 21/21 (outputs unchanged) |
| TileLang cluster, D 256 S16 BM32 / S32 BM16 | 0/21 flagged, bitwise | 20/21 | wait: 1/1 completed, then context corrupted | -- |
| TileLang grid, D 768 S16 BM16, one warpgroup | 0/21 flagged, bitwise | 21/21 | 21/21 | LSTM 11/21 (FAIL), sLSTM 19/21 |
| TileLang grid, D 768 S16 BM64, two warpgroups | 0/21 flagged, bitwise | 21/21 | 21/21 | 21/21 |
| positive control: cluster, two warpgroups | -- | -- | -- | 21/21 flagged, outputs differ |

* All 16 correct configurations again count zero violations with outputs bitwise equal to the uninstrumented kernel
  (336 runs, with and without delays).
* No completed run in either round changed its output without being flagged.
* The one control under 80%: arrive-before-store on the one-warpgroup grid kernel (LSTM) was flagged in 11/21 runs;
  the other 10 runs neither flagged nor changed the output (the late store still landed before the readers' copy).
* The skipped-wait cluster control corrupts the CUDA context ("unspecified launch failure") after its first run: a
  peer's bulk copy that nobody waits for still lands in the shared memory of a CTA that has moved on. The completed
  runs were flagged.
* The cluster variant without the explicit block barrier compiles to the same synchronization (TileLang inserts the
  `__syncthreads()` itself), as expected: 0/21, reported only.

Conclusion: the content checks see every reordering of the exchange that changed a result, in every path, including
TMA/DSMEM copies and global exchanges that compute-sanitizer's racecheck does not model; the shipped kernels show no
exchange violation under 0-65 us perturbations.

## Fast backward (run fix confirmation)

The bitwise-identical fast backward variant, same rule as round 2 (0-65 us delays, one process per variant), D 256 S32
and D 768 S16, eight warps, LSTM and sLSTM:

* correct: 0/21 flagged, outputs bitwise equal to the uninstrumented kernel, all four configurations;
* single buffer and skipped barrier: caught under the rule in all four;
* arrive before store: LSTM caught; sLSTM 9/21 (D 256) and 1/21 (D 768) -- under the 80% rule a weak control -- with no
  changed output in any run (the late store landed before the readers' load) and no silent corruption.
