# FlashRNN2 跨 SM 的 R 切分：H100 迭代报告

日期：2026-10-14（Europe/Berlin）。分支 `flashrnn2-rtiled-20261014`。全部 GPU 结果来自 MareNostrum 5 ACC 节点（H100 SXM，驱动 595.91），Triton 3.6.0、PyTorch 2.4.0a0（cu118），TileLang 0.1.15。本报告逐项对应《FlashRNN2 迭代计划：跨 SM 的 R 切分与多后端》的 G1–G4 与 E1–E4。

## 结论

* **D > 128 已可用，且比原版快。** FlashRNN2 的 R 切分路径（Triton，L2 计数器屏障 + cooperative launch）支持 LSTM/sLSTM FP16/BF16，D = 192–768，数值全部通过 PR #79 原预算。在原有分类器负载上与钉住的 FlashRNN1 CUDA fused（0b84fc3）配对计时，30/30 个对比更快，加速 1.12×–1.72×，95% 置信区间全部在 1 以上，30 个对比的几何平均 1.37×。
* **D ≤ 128 不切换。** 在 D = 96、128 上，现有单 CTA 路径在 24/24 个对比中都快于 R 切分（0.80×–0.96×，区间全部低于 1）。按计划规则，D ≤ 128 的默认保持单 CTA。
* **TileLang spike 合格但更慢。** cluster + DSMEM 的 TileLang spike 数值 96/96 通过，配对计时 6/6 慢于 Triton R 切分（0.71×–0.88×）。它是未优化的 spike，结论只针对这一实现。
* **契约不变。** full / hidden / last、FP32 compute carry、分块续跑、sLSTM full-call 初始化、PreparedSequence 与 CUDA Graph 在 R 切分路径上全部逐位一致。

## 目标完成情况

| 目标 | 状态 | 证据 |
| --- | --- | --- |
| G1 D > 128 可运行并过数值门槛 | 达成：LSTM/sLSTM FP16/BF16，D = 192、256、384、512、768 | M0 job 47298120，E1 1120/1120 PASS |
| G2 输出契约不变 | 达成：full / hidden / last、FP32 compute carry、分块续跑、sLSTM full-call 初始化；PreparedSequence 与 CUDA Graph | M0 protocol 1260/1260；M1 job 47298769 432/432 |
| G3 对 v1 CUDA fused 的大 D 配对计时 | 达成：30/30 更快，1.12×–1.72×，区间均高于 1 | M4 job 47299690，5 启动 × 20 配对块 |
| G4 合格 fallback 后端 | TileLang cluster + DSMEM spike 数值合格（96/96），但比 Triton 路径慢；尚未实现完整契约 | M3 job 47298769；E4 |

## 设计决定（M0）

Triton 3.6 提供 cooperative launch（`launch_cooperative_grid`），但没有用户可用的 DSMEM 或 cluster barrier（`num_ctas` 只做 CGA 张量切分）。因此主路径采用：

* 沿 gate/输出维把 R 切成 `k = ceil(D/S)` 片，S 为 2 的幂；每个 CTA 常驻四个 gate 的 R 切片（共享内存）和自己的 c/n/m 切片。
* 每步只交换按输入 dtype 舍入后的 h_t：双缓冲 L2 全局缓冲，读取用 `ld.global.cg` 绕过 L1。
* 每组 k 个 CTA 每步一次计数器屏障：`bar.sync` 后单线程 `atom.release.gpu` 到达、`ld.acquire.gpu` 自旋；自旋有上限，超时置错误标志，不挂死。
* cooperative launch 由驱动保证同驻；网格限制为每 SM 一个 CTA，超出的 (batch tile, head) 组在槽位内串行。
* D 不是 2 的幂但为两个 2 的幂之和时（96、192、384、768），K 精确拆成两段，不补零。

编译资源（sm_90a，FP16）：最小可行 k 为 D=192 → 2、256 → 4、384 → 6、512 → 16、768 → 24，与计划表一致（768 需 k=24，走 grid barrier）。

## 数值与契约（M0、M1）

| 检查 | 范围 | 结果 |
| --- | --- | --- |
| 资源启动 | 140 个配置，LSTM/sLSTM，D=64–768 | 140/140，错误标志 0，屏障计数精确 |
| 协议（逐位）| 网格屏障 = 每步 kernel 边界；槽位串行组；分块续跑；hidden/last = full | 1260/1260 |
| E1 对 FP64 oracle | PR #79 `CELL_BUDGET` 原值，history/carry/hidden/last，FP16/BF16，sLSTM zero/mixed/warm | 1168/1168，最差占预算 0.53% |
| 屏障压力 | 3 种 B/T/H × 50 次重复 | 48/48，结果逐位一致，无超时 |
| PreparedSequence | eager / 不规则分块 / CUDA Graph 20 次重放，三种输出模式 | 432/432 逐位一致 |

与单 CTA 路径的重叠点：D=64、D=128 全部切片宽度逐位相等（含 k=2、4、8）；D=96 只有把 K 补到 128 时逐位相等，精确两段 K（64+32）不逐位，在预算内。

## TileLang（M3）

cluster + DSMEM spike：每个 CTA 把自己的 h 切片写进 cluster 内所有 CTA 的下一步共享缓冲（`copy_cluster` 降为 `cluster.map_shared_rank` 远程写），每步一次 `cluster_sync`。对 FP64 oracle 96/96 通过（最差 0.108 预算），与 Triton 输出不逐位（最大差 4.9e-4）。探索计时比 Triton L2 屏障路径慢 1.4–2.5 倍，k 越大差距越大；spike 的远程写是逐元素 SIMT 循环，未用 TMA multicast 或 WGMMA，量的是 spike 而非 TileLang 上限。

## 配对计时（M4）

负载沿用外部基线分类器：4 层、词表 512、FP16，(B, C, T) ∈ (1, 8, 32)、(16, 32, 32)、(32, 64, 128)，只改变 D。配置在 split 0 上从 M0 合格配置中选出并冻结（表格哈希后才开始 split 1 计时）；每个对比 5 次进程启动 × 20 个 ABBA/BAAB 配对块，10000 次分层 bootstrap（种子 20261013 + 序号）。计时前每个分支都对 FP64 模型 oracle 和对方分支检查 logits 与每层 hidden 历史（原模型预算 atol 0.015、rtol 0.03、max 0.03），每个计时块再检查一次输出。300/300 次启动通过，原始输出归档已独立校验。

### E3：对 FlashRNN1 CUDA fused（加速比与 95% CI）

| D | LSTM B1/C8/T32 | LSTM B16/C32/T32 | LSTM B32/C64/T128 | sLSTM B1/C8/T32 | sLSTM B16/C32/T32 | sLSTM B32/C64/T128 |
| ---: | --- | --- | --- | --- | --- | --- |
| 192 | 1.61 [1.60, 1.61] | 1.37 [1.36, 1.38] | 1.28 [1.28, 1.28] | 1.57 [1.56, 1.58] | 1.32 [1.31, 1.33] | 1.43 [1.43, 1.44] |
| 256 | 1.58 [1.58, 1.59] | 1.33 [1.32, 1.34] | 1.28 [1.28, 1.28] | 1.53 [1.52, 1.54] | 1.29 [1.28, 1.30] | 1.27 [1.27, 1.27] |
| 384 | 1.58 [1.57, 1.59] | 1.28 [1.27, 1.29] | 1.55 [1.55, 1.55] | 1.39 [1.39, 1.40] | 1.14 [1.14, 1.15] | 1.30 [1.30, 1.31] |
| 512 | 1.40 [1.40, 1.41] | 1.12 [1.11, 1.13] | 1.32 [1.31, 1.32] | 1.36 [1.35, 1.37] | 1.12 [1.11, 1.13] | 1.33 [1.33, 1.34] |
| 768 | 1.35 [1.34, 1.36] | 1.21 [1.20, 1.23] | 1.67 [1.66, 1.67] | 1.33 [1.32, 1.34] | 1.21 [1.19, 1.23] | 1.72 [1.71, 1.72] |

v1 的 30 个 (cell, D, B) 构建全部成功并通过单层 FP64 检查；ConstrINT 选出的 tile（gate 方向切成 24–128 份等）逐个记录在 `docs/flashrnn2/rtiled-m4.md`。

### E2：D ≤ 128 切换点（单 CTA 为基线）

| D | 候选 | LSTM 三种负载 | sLSTM 三种负载 |
| ---: | --- | --- | --- |
| 96 | k = 2（S = 64） | 0.84 / 0.83 / 0.85 | 0.82 / 0.84 / 0.88 |
| 96 | k = 3（D=96）/ 4（D=128），S = 32 | 0.87 / 0.87 / 0.94 | 0.84 / 0.84 / 0.96 |
| 128 | k = 2（S = 64） | 0.84 / 0.87 / 0.89 | 0.80 / 0.86 / 0.93 |
| 128 | k = 3（D=96）/ 4（D=128），S = 32 | 0.84 / 0.87 / 0.96 | 0.81 / 0.85 / 0.96 |

数值低于 1 表示 R 切分更慢；24 个对比的 95% 区间全部低于 1。先前未配对的 CUDA Graph 内核筛查在 D = 96、128 上略偏向 R 切分，但在这里两边都走公开的逐次调用 API，R 切分每次调用多分配交换缓冲、两份 carry 和清零的屏障计数器；这部分主机侧开销可能是原因之一，本次未单独隔离。

### E4：Triton R 切分对 TileLang spike（D = 256）

| Cell | B1/C8/T32 | B16/C32/T32 | B32/C64/T128 |
| --- | --- | --- | --- |
| LSTM | 0.80 [0.79, 0.80] | 0.73 [0.73, 0.74] | 0.74 [0.74, 0.74] |
| sLSTM | 0.88 [0.87, 0.88] | 0.84 [0.84, 0.85] | 0.71 [0.70, 0.71] |

数值为 Triton 时间 / TileLang 时间，低于 1 表示 TileLang 更慢。完整表格（每个对比的基线/候选毫秒数、所选配置）见 `docs/flashrnn2/rtiled-m4.md`。

## 失败与修复记录

| 作业 | 现象 | 原因 | 修复 |
| --- | --- | --- | --- |
| 47297792（58） | 四路 canary 全部异常，62 秒结束，无 kernel 执行 | module 激活设置 `CC=nvc`，Triton 启动器 C 模块需要 gcc | runtime 导出 `CC=/usr/bin/gcc`；登录节点 dry run 实际构建启动器（无修复时复现原错误） |
| 47299215（63） | phase 0 前几个 v1 构建失败，2 分 47 秒后由本人取消，未计时 | 并发构建争用共享 `gpu_info` 模块；缺少外部基线队列已有的 `torch.amp` 命名空间适配 | 首个 v1 检查单独运行；沿用原适配；未修改任何钉住的 v1 文件 |

两次失败的原始输出都已归档并独立校验。

## 偏离计划之处

* E2 在 D = 96 上用 k = 2（S = 64，最后一片半满）和 k = 3（S = 32）：2 的幂切片宽度给不出 k = 4。
* 分支之间不比较最终状态（FP32 compute carry 与输入 dtype 快照的 dtype 不同），改为比较 logits 和每层 hidden 历史。
* 保存的资格张量只含 logits 与每层最后一步 hidden（体积原因）；完整历史只在 GPU 上检查。
* 四个 GPU 通道并发计时，各自绑定本地 NUMA 核；此前的外部基线队列是单 GPU 串行。
* E4 的 TileLang 分支是未优化的 M3 spike（逐元素 SIMT 远程写 + 每步 cluster_sync），不代表 TileLang 上限。
* 计划中的 D/B/T 网格尚未与原作者确认；本次用计划暂定的 D = 192–768 与现有分类器负载。

## 下一步

* **ThunderKittens 与 TileLang 对比（进行中）。** 已按固定提交 67845f5 取得 ThunderKittens 源码（经 VPS 流式传到 MN5，SHA256 38386831…），写好 `tk_rtiled.cu`：WGMMA（权重作 A 操作数）+ `cp.async.bulk` 跨 CTA 共享内存拷贝 + mbarrier，每步无 cluster barrier；sm_90a 编译通过，尚未上 GPU。为公平比较，TileLang 也写了同一算法的 v2（`tma_store_cluster` + mbarrier，wgmma），编译通过。下一批 GPU 作业做两者的数值资格和配对计时，并与 Triton R 切分对照。
* D ≤ 128：保持单 CTA 默认；若要再探索，需在 PreparedSequence 路径（预分配、CUDA Graph）上单独做配对计时，而不是逐次调用 API。
* 与原作者确认目标 D、B、T 网格；D > 768 与 grid barrier 的 TileLang `sync_grid` 路径尚未覆盖。
