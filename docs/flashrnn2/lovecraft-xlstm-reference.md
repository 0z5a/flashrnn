# Mixed mLSTM/sLSTM language-model checkpoint CPU reference

The public [PatrickHaller/lovecraft_xlstm](https://huggingface.co/PatrickHaller/lovecraft_xlstm) checkpoint at `34d473516f5391ffb59df94313b4acb6fd0745b6` contains one mLSTM block and one sLSTM block. All 29 serialized tensors load strictly into the pinned [NX-AI/xlstm](https://github.com/NX-AI/xlstm/tree/ab22eadbd245f293dd8dec38ed29963d73758a12) CPU model after restoring the tied LM-head alias and converting the sLSTM recurrent kernel from CUDA layout to official vanilla layout. The conversion round-trips bitwise. The model has 2,134,018 unique parameters.

The author's pinned `modeling_xlstm.py` block-stack `forward` iterates over the blocks without applying `post_blocks_norm`, although its tensor is stored in the checkpoint. The reference gate follows that actual language-model forward path. Its token-by-token adapter uses the same pinned block `step` methods and also omits the terminal norm. The author's `xLSTMForCausalLM.step` references `self.token_embedding` outside `self.model`, so it cannot itself serve as the cached reference. No package or runtime environment was updated.

At BS1/2/4, 12 five-token prompts and four greedy steps per case give 36 complete-vocabulary batch/serial and cached/full-prefix comparisons, 84 token choices, and 504 recurrent/convolution-state tensor comparisons. The budget is `1e-4 + 1e-4 * abs(reference)` in FP32 on CPU.

| Batch | Comparisons | Token choices | State tensor pairs | Batch/serial max abs | Cached/full max abs | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 12/12 | 12/12 | 72/72 | 0 | 0.0000114441 | Pass |
| 2 | 12/12 | 24/24 | 144/144 | 0.0000133514 | 0.0000123978 | Pass |
| 4 | 12/12 | 48/48 | 288/288 | 0.0000114441 | 0.0000123978 | Pass |

The [independent audit](evidence/lovecraft-xlstm-reference/lovecraft-xlstm-reference-r2.audit.json) recomputes all 36 comparisons, 84 token choices, and 504 state pairs from the 42 MB local binary snapshot; every element is within budget. [Raw rows](evidence/lovecraft-xlstm-reference/lovecraft-xlstm-reference-r2.jsonl), [run metadata](evidence/lovecraft-xlstm-reference/lovecraft-xlstm-reference-r2.meta.json), [source/checkpoint manifest](evidence/lovecraft-xlstm-reference/lovecraft-xlstm-reference-source-manifest.json), and [weight-eviction receipt](evidence/lovecraft-xlstm-reference/lovecraft-xlstm-weight-eviction-r2.json) retain the exact scope and hashes.

| Required GPU E2E comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Native xLSTM CUDA versus FlashRNN2, BS1/4/16/32/64 | — | — | Unmeasured |
| Serving concurrency 1/8/32/64/128 | — | — | Unmeasured |
| Hopper SM90 and B200 SM100 | — | — | Unmeasured |

This tiny checkpoint fills the sLSTM language-model CPU coverage gap; it is not a throughput proxy for larger models. Native CUDA, BF16 equivalence, backward, long context, and high-concurrency full-model GPU E2E remain unqualified. Its 8,539,632-byte weight was removed after hash, audit, no-reader, and completed-process checks; pinned redownload is required for GPU work.
