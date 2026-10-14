# 图注与无障碍说明

## fig01_architecture_overview

选定基线是固定 0b84fc3 的 triton_fused 推理路径，二代是 SHA 绑定的泛型 persistent / PreparedSequence 实现。两侧都有融合时间循环、权重重用、FP32 内部状态及局部反馈；图不把这些一代能力归为二代首创。默认 full 模式保留逐步完整状态历史。二代的实际变化是可选的独立 FP32 compute carry、full/hidden/last 输出契约和 full-call 初始化上下文，使 chunk 继续执行不依赖低精度最后快照。局部 feedback 线只表示源程序依赖，不保证所有配置全寄存器驻留；布局与 spill 仍按配置核实。独立预取 overlap 是 Proposed，C_t 是训练专用缓存，在两侧选定推理路径中均不分配。

Alt text: 左右同构机制图：R/b 进入 MMA，预投影 U_t 进入状态更新；S_t 在局部反馈，同时写出 Y_t。左侧最后状态是快照视图，右侧可选择独立 FP32 chunk carry。下方标明继承的局部性、真实新增契约和候选重叠。

## fig02_state_residency

上半图将执行范围与物理存储区分：CTA 是 batch/head 工作的拥有者，不代表独占 SM；代表 H100 PTX 已见寄存器 MMA fragments、共享内存 ldmatrix 交换和 CTA barrier。源程序 S_t、R/b、A_t 的全形状物理分配仍为 Unknown，不能由 loop-carried SSA 推断全寄存器、无 spill。global backing、逐步必需输出和 chunk carry 是独立对象。下半生命周期按 init、t、t+1、finish 说明读取、消费与复用；R/b 在一个 chunk 的循环中重用，下个 kernel 重新加载；FP32 carry 跨 chunk 写入并供下个顺序 kernel 使用。没有把双 carry 缓冲画成异步 input staging。输出遵守 full/hidden/last 模式，PreparedSequence 的借用结果需在下一次覆盖前消费。C_t 在当前推理范围不分配。

Alt text: CTA 范围内显示寄存器 fragments、shared exchange 和计算资源；global 区显示权重 backing、输入和输出/carry。下方六条生命周期说明状态消费、输入 ready、accumulator 消费及输出写出，标明未知的完整物理布局。

## fig03_dependency_pipeline

主面板将同一条 recurrence 展开成三个 timestep。源码中 MMA 结果与当前 U_t 都要就绪才能更新 S_t；required Y_t store 与更新同属实际执行路径。h_t 的下一步转换以及 compiler layout/visibility 在下一 MMA 前发生，c/n/m 也为下一 update 保留；图不将相邻完整 MMA 任意并行。时序是源码依赖的 logical schedule，未用微秒、cycle 或柱宽暗示实测速度；compiled 内部交换可随配置而变。C_t 在该推理范围不分配。下方独立 Proposed 面板只展示已知 U_(t+1) 的候选预取：issue、ready、consumer complete、reusable。输入与状态均 ready 才能消费，所有 consumer 完成后才可复用，协议和独立资源仍 Unknown。两个 FP32 chunk carry 缓冲遵守同一 stream 的顺序消费，不能当成异步重叠的证明。

Alt text: 四条泳道按三个 timestep 展示 MMA、输入装入、更新与输出、状态转换/ready。紫色连线保留下一 MMA 与下一 update 的依赖。下方虚线候选条带展示输入 issue、ready、consume、reuse，不带时间单位。

## fig04_execution_dispatch

上半图是当前 ExecutionPolicy 的真实 exact-shape 分派：非空离线表须匹配执行 identity，key 包含 cell、dtype、B/T/H/D；未命中的合法请求使用既有 MMA16/4/B，而 identity 错误明确报错。MMA 和 SIMT 的资格需来自固定数值门槛与离线表，图中不猜 batch 阈值。下半图是调用者显式传入 config 的 PreparedSequence：shape、dtype、contiguity、单 CUDA device、forward-only、output mode、chunk sum 等检查通过后选择实际 MMA/SIMT kernel 并保留对应 Y/carry 契约。非法请求没有自动转往一代所有 backends。资源可行性由编译/启动限定，当前没有 runtime budget 查询或完整自动 fallback registry；下方能力/资源感知策略以及 cluster/WGMMA/FP8 均为 Proposed。该图表示实验调用者和组件的实际连接，不声称公开 API 内置自动调优。

Alt text: 上方 exact-shape 表将匹配请求送往 table config，未命中送往默认 MMA，身份错误送往 error。调用者将 config 交给下方 PreparedSequence，经语义与布局验证后运行 MMA/SIMT 并返回输出和carry。未知资源门槛与后续策略用独立 TBD/Proposed 框表示。
