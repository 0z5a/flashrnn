# TiRex 35M complete-checkpoint sLSTM CPU forecast reference

The public TiRex checkpoint ran its official full-model forecast API with all
12 sLSTM blocks at BS=1/2/4. Across nine cases and 21 distinct series, its
64-step, nine-quantile forecasts match a one-series-at-a-time execution and a
separate composition of two 32-step forecast calls **bitwise**. The median
equals the 0.5 quantile. All four recurrent state tensors from every block
match the serial execution bitwise after each forecast patch. The independent
audit recomputed 12,096 quantile values per execution path and 864 complete
layer-state tensor comparisons from nine saved snapshots.

| Batch | Cases | Series | Quantile values per path | State tensor pairs | Output/state result |
|---:|---:|---:|---:|---:|---|
| 1 | 3 | 3 | 1,728 | 288 | Bitwise pass |
| 2 | 3 | 6 | 3,456 | 288 | Bitwise pass |
| 4 | 3 | 12 | 6,912 | 288 | Bitwise pass |

The [independent audit](evidence/tirex-slstm-35m-reference/tirex-slstm-35m-reference-r1-audit.json)
checks the checkpoint hash, pinned upstream source blobs, executed-source
hashes, snapshot hashes, all saved forecasts and each hidden/cell/normalizer/
stabilizer state tensor. The nine binary snapshots remain in the local
execution workspace, with hashes in the
[evidence inventory](evidence/tirex-slstm-35m-reference/tirex-slstm-35m-reference-evidence-manifest.json).

The [`NX-AI/TiRex` checkpoint](https://huggingface.co/NX-AI/TiRex/tree/63c740922493f5fbe60b277609ec62babfba2762)
is pinned to `63c740922493f5fbe60b277609ec62babfba2762`. Its
141,230,262-byte `model.ckpt` has SHA256
`b8c3f5a036c63272ce4b91c00187e26922a394cb6cb49d4e16db070ad0422314`.
The 157 stored tensors load strictly into 35,291,200 parameters after the
official `block_stack.` prefix removal. The
[upstream TiRex source](https://github.com/NX-AI/tirex/tree/91b67bc6e5d4d5d1e68a302dd848dd1d612f4c97)
is pinned to commit `91b67bc6e5d4d5d1e68a302dd848dd1d612f4c97`;
its 15 source blob hashes are in the
[source pins](evidence/tirex-slstm-35m-reference/official-source-pins.json).

Execution used Torch 2.14.1 CPU with one Torch/OMP/MKL thread. Each input has
128 observed time points; the official API pads it to the checkpoint's 2,048
point context and forecasts 64 time points in two 32-point patches. The sLSTM
Torch cell uses BF16 internally. The local runtime lacked `scikit-learn`, so
the gate supplied an import shim for its training-only split function; that
function was never called, and the upstream inference source was unchanged.
No package or environment was installed or updated.

```sh
python tools/flashrnn2/tirex_slstm_reference_gate.py \
  --model "$PINNED_TIREX_CHECKPOINT_DIRECTORY" \
  --source "$PINNED_TIREX_SOURCE_DIRECTORY" \
  --output "$RESULTS/tirex-slstm-35m-reference.jsonl"
```

| Required E2E comparison | Baseline series/s | Candidate series/s | Speedup |
|---|---:|---:|---|
| Native FlashRNN sLSTM CUDA versus Torch | — | — | Not measured |
| ONNX versus Torch forecast | — | — | Not measured |
| Multi-batch, high-concurrency GPU forecast | — | — | Not measured |

TiRex is a **time-series forecasting** checkpoint. These outputs cannot be
substituted for language-model token-generation baselines; a complete public
sLSTM language-model checkpoint and its E2E qualification remain separate
work. The completed local weight was evicted after the audit, SHA256 check and
no-reader check; a GPU test requires pinned re-download.
