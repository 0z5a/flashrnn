# 实际源码到架构判断的映射

对应源代码已保存为 evidence-snapshots 下的文本；完整原路径、行范围和 SHA 在 evidence.md。没有新建 StateManager/PipelineEngine 等虚构组件。

| component / source | role / evidence | figures |
|---|---|---|
| [v2_triton_persistent](evidence-snapshots/source/v2/triton_persistent.py.txt) | I004, I005, I006, I007, I008, I009, I014, I015, U001, U002 | fig01_architecture_overview, fig02_state_residency, fig03_dependency_pipeline, fig04_execution_dispatch |
| [v2_state_contract](evidence-snapshots/source/v2/state_contract.py.txt) | I005, I008, I013 | fig01_architecture_overview, fig02_state_residency, fig03_dependency_pipeline, fig04_execution_dispatch |
| [v2_prepared](evidence-snapshots/source/v2/prepared.py.txt) | I004, I007, I008, I009, I010, I012, I013, I015, I016, U002, U003 | fig01_architecture_overview, fig02_state_residency, fig03_dependency_pipeline, fig04_execution_dispatch |
| [v2_execution_policy](evidence-snapshots/source/v2/execution_policy.py.txt) | I011, I012, U003 | fig04_execution_dispatch |
| [v2_simt_persistent](evidence-snapshots/source/v2/simt_persistent.py.txt) | I012 | fig04_execution_dispatch |
| [v1_lstm_fw](evidence-snapshots/source/v1/lstm_fw.py.txt) | I001, I002, I003 | fig01_architecture_overview, fig02_state_residency |
| [v1_slstm_fw](evidence-snapshots/source/v1/slstm_fw.py.txt) | I001, I002, I003 | fig01_architecture_overview, fig02_state_residency |
| [policy_tests](evidence-snapshots/source/tests/test_execution_policy.py.txt) | I011 | fig04_execution_dispatch |
| [caller_helpers](evidence-snapshots/source/caller/helpers.py.txt) | I016 | fig04_execution_dispatch |
| [selection_worker](evidence-snapshots/source/caller/worker.py.txt) | I016 | fig04_execution_dispatch |

显式实验调用者：helpers.Model.__init__ 调用 policy.select 后将 config 传给 PreparedSequence；PreparedSequence 自身没有自动 selector。Public default recurrence 的 legacy 输出行为单独保留，不能把实验 caller 画成公共 API 已接入的全局调优器。
