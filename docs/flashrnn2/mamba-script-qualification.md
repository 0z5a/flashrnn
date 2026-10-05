# Full Mamba TorchScript qualification

The existing RTX 5090 runtime has Torch but no Transformers. A task-owned
TorchScript export provides a route to test the complete checkpoint without
installing packages. This is a separate baseline; it does not claim the
native `mamba_ssm` fast path or apply the FlashRNN LSTM kernel to Mamba.

The checkpoint remains `state-spaces/mamba-130m-hf` at
`1e76775f628fbf1350fbe4dbb3d971ba64af25a1`, with all 24 layers and
129,135,360 parameters. Strict native loading has no missing, unexpected or
mismatched weights. Each method returns logits, convolution cache and SSM
cache. Caches are request-owned mutable inputs, not immutable initial states.

| Qualification | Cases | Result |
| --- | --- | --- |
| Native CPU vs CPU TorchScript, prefill | Three independent B1/T5 token inputs | All three outputs bitwise equal |
| Native CPU vs CPU TorchScript, decode | One greedy step after each prefill | All three outputs bitwise equal |
| CUDA device retarget | 50 traced CPU device literals changed to CUDA | Complete artifact loaded and executed on RTX 5090 |
| Native CPU oracle generation | Three separate inputs, new seed 20261006 | Complete logits and caches saved |
| Existing Torch 2.12 CUDA runtime | Three full-model B1/T5 prefill/decode cases | One PASS; two prefill SSM-cache failures retained |
| Multiple batch sizes / long generation / high concurrency | B1/2/4 CPU generation; queued requests | CPU evidence only; full CUDA campaign pending |

The successful CPU export used the existing Torch 2.13.0 / Transformers
4.54.1 runtime. Its controller naturally exited 0. An earlier attempt
incorrectly treated the traced batch dimension as dynamic: B1 passed,
then B2 failed in indexed cache assignment. That failure is retained.
Exports now declare a static batch and prompt length; they must be
qualified separately for every additional shape.

The CUDA artifact changes only device constants and must be loaded with
`map_location="cuda"`. Cross-version serialization and CUDA execution now
work, but numerical qualification fails. The CUDA gate checks all
three complete outputs and both greedy choices on each independent input.
The frozen oracle metadata uses the upstream pretrained fixture's logits
budget (atol/rtol 0.001), and the CPU export cache budget (1e-5 each).
These budgets were fixed before CUDA execution and do not change the
failed stricter cached/full-prefix experiment in the native CPU report.

On GPU1 with Torch 2.12.1+cu130, all 24 layers and 129,135,360 parameters
execute for each case. Both greedy choices match the native CPU oracle in
all three cases. All six complete-logit comparisons and all convolution
cache checks pass. Two prefill SSM cache checks fail the unchanged
`atol=rtol=1e-5`; all three decode SSM cache checks pass.

| Case | Prefill logits max error | Prefill SSM max error | Prefill SSM relative L2 | Both greedy choices | Overall |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 0.000480652 | 0.0000782013 | 1.977e-6 | Match | FAIL |
| 1 | 0.000308990 | 0.0000720024 | 1.898e-6 | Match | FAIL |
| 2 | 0.000228882 | 0.0000190735 | 1.033e-6 | Match | PASS |

This run does not isolate CPU/GPU arithmetic differences from cross-version
or serialization effects. Small relative norms do not override the failed
elementwise checks. The controller naturally exits 1 after all three cases;
raw results are archived and the GPU returned. See
[cases](evidence/mamba-script-r1.jsonl) and
[metadata](evidence/mamba-script-r1.meta.json).

No model speedup or overall GPU correctness is claimed here. Model files and
derived artifacts stay task-owned until the whole model campaign finishes.
The evidence directory contains hashes and small logs, not weights or caches.

Reproduction, using the existing environment:

```sh
python tools/flashrnn2/export_mamba.py --model "$MODEL" --batch 1 --prompt-length 5 --output "$CPU_ARTIFACT"
python tools/flashrnn2/target_mamba_artifact.py --input "$CPU_ARTIFACT" --output "$CUDA_ARTIFACT"
python tools/flashrnn2/mamba_script_goldens.py --model "$MODEL" --output "$ORACLES"
# This CUDA gate has executed and exits 1 with the two retained cache failures.
python tools/flashrnn2/mamba_script_gate.py --model "$CUDA_ARTIFACT" --goldens "$ORACLES" --output "$RESULTS"
```
