# Layer baseline qualification

The adapter maps FlashRNN LSTM, GRU and Elman parameters to `torch.nn` recurrent layers. Input projection belongs to the layer workload; gate packing and parameter construction belong to setup. The observable contract is hidden history plus final states. LSTM intermediate cell history is not returned by torch.nn, so these modules cannot stand in for a full-state recurrence API baseline.

LSTM uses the same i/f/z/o order and puts the single bias into the input bias. GRU reorders recurrent gates from candidate/reset/update to reset/update/candidate; its fourth bias maps to the input candidate bias and its recurrent candidate bias remains inside the reset multiplication. Each head has separate parameters.

| Qualification | Result | Evidence |
| --- | --- | --- |
| CPU FP64 outputs and all gradients, history+final and final-only losses | 6 combinations passed | [CPU log](evidence/cpu-layer-gate-r1.log) |
| RTX 5090 BF16 LSTM, GRU and Elman; B3/T17/H2/D64 and B16/T128/H1/D128 | 6 cases passed | [GPU JSONL](evidence/layer-gate-r1.jsonl) |
| cuDNN kernel trace / actual operator dispatch | Not yet inspected | `cudnn_enabled` alone is not dispatch proof |
| Candidate layer speedup against torch.nn | Not measured | No performance claim |
| Full pretrained model / concurrent serving | Not run | Not established by layer qualification |

The GPU gate used the existing Python 3.12.3, Torch 2.12.1+cu130, cuDNN 92000 runtime on SM120. Maximum hidden/final-state error was 0.0009765625, below the frozen absolute 0.003 gate; atol=0.002 and rtol=0.02 also passed. CPU tests check input, projection, recurrent, bias and initial-state gradients against the independent FP64 reference. GPU backward is not yet qualified.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 python -m unittest discover -s tests/flashrnn2 -p test_torch_layer.py -v
PYTHONPATH=. OMP_NUM_THREADS=1 python tools/flashrnn2/layer_gate.py --output artifacts/layer-gate.jsonl
```
