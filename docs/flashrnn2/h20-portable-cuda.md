# H20 portable CUDA correctness

The frozen portable-r3 gate ran once on NVIDIA H20 (GPU `d952768b`, boot
`4620b8a9`) using source commit `b6cb215e`. It checks the public Torch CUDA
recurrence contract for LSTM, sLSTM, GRU and Elman. This is a correctness gate
for portable arithmetic, not a timing or optimized Hopper-kernel result.

| Cell | FP64 mathematical | FP32 state / BF16 operands | Total passed | Full-model speedup |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 4/4 | 4/4 | 8/8 | Unmeasured |
| sLSTM | 7/7 | 7/7 | 14/14 | Unmeasured |
| GRU | 4/4 | 4/4 | 8/8 | Unmeasured |
| Elman | 4/4 | 4/4 | 8/8 | Unmeasured |
| **Total** | **19/19** | **19/19** | **38/38** | **Unmeasured** |

The runner saved all 38 JSONL rows and 38 tensor snapshots. Its controller and
natural-exit wrapper returned zero. A separate CPU Torch audit recomputed 456
tensor pairs and 176 per-state comparisons; an independent NumPy audit decoded
the saved tensor archives and recomputed the same comparisons without importing
Torch. Both passed. The complete offbox archive contains 206 SHA-verified files
plus its manifest (SHA-256 `89d7cef436e00ab6c860933efd8f34bf752eb1e65c321b9c15ae4f54cf2646c3`).
Fresh boot/GPU/lock checks found no compute applications, and the original
machine supervisor accepted the WHOLE handback and exited naturally with code
zero. The [audited archive](../../evidence/flashrnn2/h20-portable-r3/complete.tar.gz),
[rows](../../evidence/flashrnn2/h20-portable-r3/portable-cuda-r3.jsonl),
[metadata](../../evidence/flashrnn2/h20-portable-r3/portable-cuda-r3.meta.json),
[Torch audit](../../evidence/flashrnn2/h20-portable-r3/portable-cuda-r3-audit.json),
[NumPy audit](../../evidence/flashrnn2/h20-portable-r3/audit_numpy.result.json),
[independent evidence-chain check](../../evidence/flashrnn2/h20-portable-r3/evidence-chain-audit.json)
and [WHOLE receipt](../../evidence/flashrnn2/h20-portable-r3/whole.json) preserve
the result and its provenance.

| E2E workload | Accelerated baseline | FlashRNN2 candidate | Paired speedup |
| --- | ---: | ---: | ---: |
| Complete models, batch 1/4/16/32/64, concurrency 1/8/32/64/128 | Unmeasured | Unmeasured | Unmeasured |

No model checkpoint, native FlashRNN1/Haste kernel, high-concurrency serving
cohort or B200-specific Tensor Memory path was exercised in this H20 gate.
