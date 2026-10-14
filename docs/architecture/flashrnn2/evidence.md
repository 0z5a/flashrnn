# 架构证据、版本与未知项

本文件由 architecture.json 生成。Implemented 只表示指定本地路径有可达源码；built 只适用于两个保留的代表 PTX。没有把源码、编译、历史测量或出图验收当作本轮 GPU 正确性/性能 PASS。

## 工作区与选择的基线

输出仓库：.，master @ 74a8b31b918f3a1b3fecf310ebc1269cdd21d489；原有 tracked 改动保留。
二代：flashrnn2-single-gpu-opt-20261013，codex/flashrnn2-single-gpu-opt-20261013 @ 1e162f03edad876d1e9c7d7471795464fc40ee2d。
二代 git status/diff/rev-parse 的部分命令超时，dirty 状态记 Unknown；HEAD/ref 只读解析得到 commit。所有图按当前文件 SHA 取证，五个核心文件与冻结 experiment manifest 匹配。
冻结 manifest SHA256：dc3b85003a41743daf99d2cdf362e8fb35a824660d60db9678fbf7a1e3f4ad69。
选定基线：triton_fused; inference forward @ 0b84fc30120ecccbde97976bc5b086f3215572d2。
同名根目录执行文件不存在，使用用户提供的 Downloads 附件，保存在 evidence-snapshots/EXECUTION_SPEC.md。
执行规范 SHA256：0839ec7eb86e81fce8aacbae9a0dd7f4ca20d0652e78a080774125c5ef69e6df。

## 范围与模型

{
  "cells": [
    "lstm",
    "slstm"
  ],
  "phase": "generic forward / inference",
  "dtype": [
    "float16",
    "bfloat16"
  ],
  "hardware": "CUDA; representative compiled samples are H100 / sm_90a, not an all-GPU claim",
  "input": "Wx [B,T,4,H,D] = preprojected gate inputs U_t; R [4,H,D,D]; b [4,H,D]",
  "state": "LSTM S=(h,c); sLSTM S=(h,c,n,m); only h enters recurrent matrix multiplication",
  "output": "Full-state history remains default; explicit hidden/last modes and independent FP32 compute carry are opt-in",
  "not_in_scope": [
    "TiRex / xLSTM arithmetic contracts",
    "backward implementation",
    "distributed execution",
    "measured pipeline overlap",
    "speedup claims in these figures"
  ]
}

## 架构证据表

| claim_id | claim | status | scope | source_kind | source_locator / symbol | verification | figure_ids |
|---|---|---|---|---|---|---|---|
| I001 | The selected baseline already fuses the timestep loop in one batch/head CTA, reuses recurrent weights and keeps FP32 state variables between steps; h is rounded for tl.dot. | implemented | pinned 0b84fc3 / triton_fused / LSTM+sLSTM inference / FP16+BF16 | local_source | flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/lstm_fw.py:85-108 / _forward_sequence_kernel<br>flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/lstm_fw.py:135-249 / _forward_sequence_kernel<br>flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/lstm_fw.py:336-338 / _forward_sequence_kernel<br>flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/slstm_fw.py:239-285 / _forward_sequence_kernel<br>flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/slstm_fw.py:419-423 / _forward_sequence_kernel | source_observed | fig01_architecture_overview |
| I002 | Baseline inference writes full state history in its selected dtype and returns the last-state view of that history; the API view is not a separate FP32 computation carry. | implemented | pinned 0b84fc3 / triton_fused / LSTM+sLSTM inference / FP16+BF16 | local_source | flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/lstm_fw.py:251-280 / _forward_sequence_kernel<br>flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/lstm_fw.py:456-473 / forward_sequence<br>flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/slstm_fw.py:324-363 / _forward_sequence_kernel<br>flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/slstm_fw.py:543-560 / forward_sequence | source_observed | fig01_architecture_overview |
| I003 | Baseline gate cache C_t is conditional on OUTPUT_GATES; the inspected inference branch omits it while still writing state history. | implemented | pinned 0b84fc3 / triton_fused / LSTM+sLSTM inference / FP16+BF16 | local_source | flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/lstm_fw.py:282-334 / _forward_sequence_kernel<br>flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/lstm_fw.py:440-473 / forward_sequence<br>flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/slstm_fw.py:365-417 / _forward_sequence_kernel | source_observed | fig01_architecture_overview, fig02_state_residency |
| I004 | V2 MMA uses one CTA per batch tile/head, loads R/b before the chunk loop and feeds local state variables to later steps; it adds no explicit cross-CTA state dependence in this path. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:39-83 / _sequence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:69-90 / PreparedSequence._execute | source_observed | fig01_architecture_overview, fig02_state_residency, fig03_dependency_pipeline, fig04_execution_dispatch |
| I005 | Generic full state and dot accumulators use FP32; only h is converted to the input dtype before the next recurrent MMA. Carry state and observable snapshots are distinct. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/state_contract.py:20-47 / state_contract<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:45-83 / _sequence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:105-153 / _sequence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:89-100 / _sequence gate preactivations | source_observed | fig01_architecture_overview, fig02_state_residency, fig03_dependency_pipeline, fig04_execution_dispatch |
| I006 | Opt-in compute carry is stored from FP32 computation state at a chunk boundary, independently of history dtype. Default recurrence still returns a final history snapshot. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:167-172 / _sequence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:194-205 / recurrence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:252-266 / recurrence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:286-339 / recurrence | source_observed | fig01_architecture_overview, fig02_state_residency |
| I007 | Full outputs every state component; hidden outputs h each step; last outputs only h at the logical final step. Reduced modes require explicit caller intent and preserve FP32 compute carry. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:154-172 / _sequence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:199-205 / recurrence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:22-59 / PreparedSequence.__init__ | source_observed | fig01_architecture_overview, fig02_state_residency, fig03_dependency_pipeline, fig04_execution_dispatch |
| I008 | sLSTM initialization uses one full-call zero-normalizer predicate, and the context advances its logical step offset only after completed chunks. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/state_contract.py:50-97 / SequenceContext; begin_sequence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:117-121 / _sequence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:63-81 / PreparedSequence._execute | source_observed | fig01_architecture_overview |
| I009 | PreparedSequence allocates stable output and two FP32 carry buffers; chunks execute serially and address the original Wx/output with TOTAL_T and STEP_BASE. Carry A/B do not demonstrate asynchronous overlap. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:45-59 / PreparedSequence.__init__<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:69-98 / PreparedSequence._execute<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:38-67 / _sequence | source_observed | fig01_architecture_overview, fig02_state_residency, fig04_execution_dispatch |
| I010 | Prepared results are borrowed and overwritten by the next execution; a plan is used serially on its bound stream. Optional CUDA Graph capture replays the same prepared execution. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:1-6 / module contract<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:100-126 / run; capture; close | source_observed | fig02_state_residency, fig04_execution_dispatch |
| I011 | ExecutionPolicy checks identity for nonempty tables and selects an exact cell/dtype/B/T/H/D key; an absent key uses KernelConfig() MMA16/4/B. Identity mismatch raises an error. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/execution_policy.py:9-26 / KernelConfig<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/execution_policy.py:32-50 / ExecutionPolicy.select<br>flashrnn2-single-gpu-opt-20261013/tests/test_execution_policy.py:16-32 / PolicyContract | source_observed | fig04_execution_dispatch |
| I012 | MMA configurations permit tile16/32 and 4/8 warps; SIMT permits one batch row/head per CTA and 4/8 warps, with a different FP32 reduction tree. Config availability is not universal speedup. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/execution_policy.py:9-26 / KernelConfig<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/simt_persistent.py:18-49 / _sequence_simt<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:75-90 / PreparedSequence._execute | source_observed | fig04_execution_dispatch |
| I013 | Prepared admission requires generic LSTM/sLSTM FP16/BF16, four gates, nonempty B/T/H, 16<=D<=128, matching shapes/dtypes, contiguous same-device CUDA tensors, no gradients, valid output mode and chunk sum. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/state_contract.py:20-47 / state_contract<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:22-47 / PreparedSequence.__init__ | source_observed | fig04_execution_dispatch |
| I014 | Two preserved representative H100 compiler outputs contain warp MMA, shared-memory ldmatrix and CTA bar.sync; they do not contain WGMMA/TMA/cp.async. Exact feedback allocation is not inferred from these counts. | implemented | only the two named compiler samples / Triton 3.6.0 / sm_90a | compiler_output, local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:65-83 / _sequence<br>evidence-snapshots/compiler/lstm-float16-b16-t128-d64-h2-split0--mma4-0.ptx / _sequence<br>evidence-snapshots/compiler/slstm-bfloat16-b16-t128-d64-h2-split0--mma8-0.ptx / _sequence | source_observed, built | fig01_architecture_overview, fig02_state_residency |
| I015 | This V2 persistent/prepared path is forward-only and does not allocate a backward-only gate cache C_t. Required Y_t output stores still occur. | implemented | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:154-172 / _sequence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:194-233 / recurrence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:35-59 / PreparedSequence.__init__ | source_observed | fig01_architecture_overview, fig02_state_residency, fig03_dependency_pipeline |
| I016 | The measured caller explicitly invokes policy.select and passes the result as config to PreparedSequence. PreparedSequence itself does not call the selector or benchmark requests. | implemented | frozen optimization experiment caller + current plan components | local_source | flashrnn2-distributed-p0-20261011-evidence/paper-optimizations-20261013-v2/helpers.py:92-110 / Model.__init__<br>flashrnn2-distributed-p0-20261011-evidence/paper-optimizations-20261013-v2/helpers.py:50-64 / cell_functions<br>flashrnn2-distributed-p0-20261011-evidence/paper-optimizations-20261013-v2/worker.py:62-80 / select<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:48-50 / PreparedSequence.__init__ | source_observed | fig04_execution_dispatch |
| U001 | Exact physical mapping of S_t, R/b and A_t to registers/shared memory/spills is configuration dependent and not established for every admitted shape; all-register residency is not claimed. | unknown | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_measurement, local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:328-338 / recurrence diagnostics<br>flashrnn2-paper-20261012/analysis/ablations-20261013/compiler-resources.csv | none | fig01_architecture_overview, fig02_state_residency, fig03_dependency_pipeline |
| U002 | No named input-staging slot protocol or effective input/compute overlap has been established in the inspected current generic source. A candidate producer/consumer protocol remains TBD. | unknown | current SHA-bound generic persistent forward / FP16+BF16 / 16<=D<=128 | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py:65-100 / _sequence<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:63-98 / PreparedSequence._execute | none | fig02_state_residency, fig03_dependency_pipeline |
| P001 | A future independent U_(t+1) prefetch may overlap current computation, but must complete before consumption and cannot reuse a slot before all consumers finish. Recurrent MMA remains dependent on state readiness. | proposed | logical candidate only; instruction, independent resources and synchronization protocol TBD | task_design | evidence-snapshots/EXECUTION_SPEC.md:section-7.2-7.5 | none | fig01_architecture_overview, fig03_dependency_pipeline |
| P002 | A future capability/resource-aware selector may extend explicit planning; it must preserve the output contract and choose a qualified fallback or error. No resource threshold is implemented here. | proposed | future dispatch design; threshold and qualified fallback registry TBD | task_design | evidence-snapshots/EXECUTION_SPEC.md:section-8 | none | fig04_execution_dispatch |
| P003 | Cluster-resident sharding, WGMMA operand remapping and FP8 are later candidates, isolated from the implemented single-CTA path. Sharding axes, protocols and qualification are unspecified. | proposed | later candidates, not current FlashRNN2 support | hardware_documentation, task_design | evidence-snapshots/EXECUTION_SPEC.md:section-2.2<br>https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator | none | fig04_execution_dispatch |
| D001 | The FlashRNN paper describes WMMA fusion/weight reuse and grid synchronization in some recurrent sharding situations; that documented CUDA design is not substituted for the selected local Triton baseline. | documented | FlashRNN paper Algorithm 1 / section5; backend-dependent | paper | https://arxiv.org/html/2412.07752v2#S5 | none | fig01_architecture_overview |
| U003 | The explicit planner has no runtime resource-budget query or automatic fallback registry; feasible compiled resources and future thresholds remain unspecified. | unknown | current ExecutionPolicy + PreparedSequence; not all upstream backends | local_source | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/execution_policy.py:36-50 / ExecutionPolicy.select<br>flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py:22-50 / PreparedSequence.__init__ | none | fig04_execution_dispatch |

## 所有权 / 存储 / 生命周期 / 访问

| 对象 | owner | location | lifetime | access pattern | claims |
|---|---|---|---|---|---|
| S_t | one batch tile/head CTA (MMA), or one batch row/head CTA (SIMT) | loop-carried FP32 SSA values; full physical mapping TBD | until next-step MMA(h) and state-update consumers(c/n/m) finish; boundary carry exported separately | read/update each step; h rounded for MMA | I004, I005, U001 |
| R_b | one batch/head CTA instance | global backing; compiler-mapped loaded values | loaded before a chunk loop; reused through chunk; a later kernel loads again | read-only repeated use | I004, U001 |
| U_t | input tensor / consuming CTA tile | global Wx, then loaded values; named staging slots not specified | load then pointwise consumption within a step | four gate inputs per step; masks on B/D tails | I004, U002 |
| A_t | current CTA | FP32 dot accumulators; exact allocation TBD | dot result through pointwise gate consumption | produce by MMA; consume by update | I005, U001 |
| Y_t | caller selects output mode; CTA writes disjoint tile | global output array | public result; prepared result borrowed until next run overwrites it | full/hidden every step, last only logical final h | I007, I010 |
| carry | prepared plan / caller | global FP32 arrays A/B at chunk boundaries | read by next serial chunk; reuse only after it completes | store full S then read as next initial | I006, I009 |
| C_t | no owner in this inference scope | absent in inspected inference paths | not allocated; training baseline is outside scope | backward-only conditional gate stores on baseline training branch | I003, I015 |

## 源文件快照与内容 SHA

源码证据包括原路径、实际行范围、symbol 与 SHA。快照是文档证据文本，不是新 kernel 或生产模块。若原工作树仍存在，validator 同时校验原文件与快照。

| source_id | origin | documentary snapshot | SHA256 |
|---|---|---|---|
| v2_triton_persistent | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/triton_persistent.py | [evidence-snapshots/source/v2/triton_persistent.py.txt](evidence-snapshots/source/v2/triton_persistent.py.txt) | c83a6255f01e1db30e5ea6bc9a50f5f7a6d24cffe26b7bd8a89faffe7b2148e1 |
| v2_state_contract | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/state_contract.py | [evidence-snapshots/source/v2/state_contract.py.txt](evidence-snapshots/source/v2/state_contract.py.txt) | b99311ca231c478860b55ecae216e88ecb3a55f9d8aa212428b82157f8b9485e |
| v2_prepared | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/prepared.py | [evidence-snapshots/source/v2/prepared.py.txt](evidence-snapshots/source/v2/prepared.py.txt) | 4201ab2390c845178f9c6585b64867b3459a1acb860a659c17882acee6d24c13 |
| v2_execution_policy | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/execution_policy.py | [evidence-snapshots/source/v2/execution_policy.py.txt](evidence-snapshots/source/v2/execution_policy.py.txt) | 5b3f86a1877dcd75108db2e5f722b6a1e34aa056f646dee9704d1cc8241b3759 |
| v2_simt_persistent | flashrnn2-single-gpu-opt-20261013/flashrnn/flashrnn2/simt_persistent.py | [evidence-snapshots/source/v2/simt_persistent.py.txt](evidence-snapshots/source/v2/simt_persistent.py.txt) | 8418da5de7497dd05ba5f161dad2ff5ae16ee8b11dc1038000131c84ff1f6f8e |
| v1_lstm_fw | flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/lstm_fw.py | [evidence-snapshots/source/v1/lstm_fw.py.txt](evidence-snapshots/source/v1/lstm_fw.py.txt) | 9d5a940a13b95f2df603b5d61d44d5d4b6778a65e4ee7ab0455b909b528b1cc6 |
| v1_slstm_fw | flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/slstm_fw.py | [evidence-snapshots/source/v1/slstm_fw.py.txt](evidence-snapshots/source/v1/slstm_fw.py.txt) | 4b25ef542b80ab0e0ac7cf17b3cde891a0efcb5932866d9fc7dfef6577bb894c |
| v1_fwbw | flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/triton_fused/fwbw.py | [evidence-snapshots/source/v1/fwbw.py.txt](evidence-snapshots/source/v1/fwbw.py.txt) | e62bb13bf84d90e08df1e283de6ae934a15c99c0956427f9f21490065e17bc56 |
| v1_config | flashrnn2-distributed-p0-20261011-evidence/paper-baselines-20261013-v1/pinned/flashrnn/flashrnn/flashrnn.py | [evidence-snapshots/source/v1/flashrnn.py.txt](evidence-snapshots/source/v1/flashrnn.py.txt) | 08d4fe1f4f5051cc296431b9696d91e16a74d1a07af41f331a841d70783d7a50 |
| policy_tests | flashrnn2-single-gpu-opt-20261013/tests/test_execution_policy.py | [evidence-snapshots/source/tests/test_execution_policy.py.txt](evidence-snapshots/source/tests/test_execution_policy.py.txt) | 6a4f1a81dc626c08bafde5307640755e98e3f5002ecef6c7e74b3edee0ed5364 |
| caller_helpers | flashrnn2-distributed-p0-20261011-evidence/paper-optimizations-20261013-v2/helpers.py | [evidence-snapshots/source/caller/helpers.py.txt](evidence-snapshots/source/caller/helpers.py.txt) | 7ea229c2eedcf6039a3949b41894b90d2a175d1688cb724d51e19003f8a8b079 |
| selection_worker | flashrnn2-distributed-p0-20261011-evidence/paper-optimizations-20261013-v2/worker.py | [evidence-snapshots/source/caller/worker.py.txt](evidence-snapshots/source/caller/worker.py.txt) | c82273c37d3fbf5b225da7f412c955b1ca7cc49b5fe94dc58e55b85ad1a00822 |
| v2_readme | flashrnn2-single-gpu-opt-20261013/README.md | [evidence-snapshots/design/README.md.txt](evidence-snapshots/design/README.md.txt) | 78f22eaf5142d3b8746352e66c0bf132c9cfec9d40c695bc64cae432b220818d |
| continuation_design | flashrnn2-single-gpu-opt-20261013/docs/flashrnn2/persistent-continuation.md | [evidence-snapshots/design/persistent-continuation.md.txt](evidence-snapshots/design/persistent-continuation.md.txt) | 6e001748f85ba7fd3cd7336e578baaa534ef4af5f76f7b811a6b156720a797c0 |
| resource_cohort | flashrnn2-paper-20261012/analysis/ablations-20261013/compiler-resources.csv | [evidence-snapshots/measurements/compiler-resources.csv](evidence-snapshots/measurements/compiler-resources.csv) | 1e99a1a48afb89708bbcf9aab00a35122377efde3394be37520fb142e7216445 |

## 与任务默认候选的差异

找到了 SHA 绑定的二代实现，采用 implementation-focused 模式，主图选择泛型 LSTM/sLSTM FP16/BF16 forward。单 CTA、融合、循环重用权重、FP32 内部状态与局部 feedback 是所选一代已有机制。二代实际增量聚焦独立 compute carry、输出契约、full-call context、固定地址的准备执行和离线 exact-shape 策略。

## 仍需实现或验证

- 当前路径没有新一次 GPU launch；图中没有新的性能数字或有效 overlap 测量。
- 全形状 S_t/R/b/A_t 物理映射、layout 转换成本与 spill 归因为 Unknown。旧 resource CSV 是旧 ablation cohort，不能充当当前全形状的资源证明。
- input prefetch 的独立资源、issue/complete/wait、slot ready/reuse 协议与 trace 为 Unknown；候选关系为 Proposed。
- capability/resource-aware 自动分派、fallback registry、cluster 轴与通信、WGMMA 操作数、FP8 为 Proposed / TBD。
- TiRex/xLSTM 的特定算术、训练/backward 与分布式传输不属于此泛型图的实现范围。
- global 地址访问不换算 HBM 事务；loop-carried SSA 不等于全寄存器、无 spill；CTA 不独占 SM。

## 一手参考（仅为对应机制的依据）

- [FlashRNN paper v2, Algorithm 1 / section 5 / Appendix E](https://arxiv.org/html/2412.07752v2): D001 的 CUDA 文献背景，与所选本地 Triton 基线严格分开。
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html): 后续 cluster/TMA 硬件机制；不证明 FlashRNN2 已采用。
- [CairoSVG API documentation](https://cairosvg.org/documentation/): 实际 PNG/PDF 导出接口，不作为 GPU 架构证据。
