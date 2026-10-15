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

Results: pending.
