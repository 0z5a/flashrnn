# FlashRNN2 新架构出图：Codex 执行文档

版本：1.0  
资料核对日期：2026-10-14  
任务类型：读取代码与设计材料，生成可编辑、可复现、有证据的架构图。  
默认交付语言：图内英文；讲解、验收报告中文；每张图附中英文 caption。

> 核心任务不是“画一个 RNN 模块框图”，而是说明：FlashRNN2 相对所选 FlashRNN 基线，如何改变循环状态的驻留位置、反馈路径、同步范围和执行调度。画图不得反过来成为臆造实现的理由。
>
> 本文提供的是执行规范与设计候选，不是 FlashRNN2 已实现功能的声明。当前没有随任务提供二代的专用仓库、分支或实现 diff。运行本任务的 Codex 应以其本地实际仓库及设计材料取证；若只有一代代码，仍完成明确标为 Proposed 的二代方案图。

---

## 0. 直接交给 Codex 的启动指令

把本文放进目标仓库，在 Codex 中发送以下内容。不要用本文覆盖已有 `AGENTS.md`。

```text
阅读 FLASHRNN2_ARCHITECTURE_FIGURES_EXECUTION.md，并遵守当前仓库适用的 AGENTS.md。

请直接执行这份文档，不要只返回计划、Mermaid 文本或图片生成提示词。
先读取本地代码与设计材料，建立架构证据表，再完成以下四张图：
1. FlashRNN 与 FlashRNN2 的机制对比总览；
2. 循环状态驻留、存储位置和生命周期；
3. 保留真实递归依赖的执行流水时序；
4. workload-aware 执行路径选择与 fallback。

采用本文规定的论文式机制图模板，以结构化 JSON 和 Python 生成可编辑 SVG，
并导出 PNG、矢量 PDF；提交渲染源码、中英文 caption、证据映射和验收报告。

沿用本文中的“状态常驻、同步局部化、依赖感知流水、按 workload 选择执行方式”
作为待核实的设计主线。单 CTA BF16 forward 是默认的第一阶段候选，
cluster、WGMMA 操作数重映射、FP8 是后续候选，不得直接标为已实现。
本地新代码或明确设计材料有不同决策时，以它们为准，并记录差异。

每个关键模块、数据流、同步点和性能判断必须能追溯到证据。
区分 Implemented / Documented / Proposed / Unknown。
找不到二代实现时，继续输出 Proposed architecture，不要编造源码、测试、
性能数字或当前硬件支持。缺少数值就用符号条件，不要追问用户补齐画图细节。

图只能修改 docs/architecture/flashrnn2/ 内的文档、图和绘图工具。
不要修改 kernel、接口、构建系统或已有用户改动，不要自动提交或推送。
实际运行绘图和校验，并查看渲染结果；修复遮挡、错误连线、错误依赖后交付。
不具备某种导出或视觉检查能力时，明确报告缺项，不能把未执行项写成通过。
```

运行环境无需 GPU。不要为了画图安装整套 CUDA/PyTorch，或发起 GPU 租赁与性能测试。已有的测试、编译与 profiling 结果可以作为证据读取。

---

## 1. 选定的模板与工具路线

### 1.1 推荐模板：论文式“机制对比 + 空间映射 + 时间调度”

不要把所有事情塞进一张巨图。主交付四张图，每张只回答一个问题。

| 图 | 核心问题 | 选用的表达模板 |
|---|---|---|
| 01 总览 | 二代究竟改了什么？ | 同构的新旧机制对比，两栏使用相同的数据与计算节点 |
| 02 驻留 | 状态在哪，谁拥有，什么时候可复用？ | 物理存储位置图 + 数据生命周期条带 |
| 03 时序 | 哪些能重叠，哪些必须等待？ | 带依赖边的泳道时序图，参考 FlashAttention-3 的调度表达方式 |
| 04 分派 | 哪类请求走哪条执行路径？ | 带能力约束、资源约束和 fallback 的决策图 |

可选附图 05：源码组件映射。仅在模块与符号已查明、确实有助于代码评审时生成。

模板借鉴的是组织信息的方法，不是复制别人的图片：

- FlashAttention-3 的 Figure 1–2 把执行资源、迭代与重叠关系分开表达，适合作为时序图的参考；不要把其 attention 的可并行关系直接套给 RNN。[R3]
- C4 component/dynamic 图适合解释软件组件及交互；本任务只借用其分层和编号思想，不把 kernel 执行图冒充严格 C4 图。[R4][R5]
- FlashRNN 原论文的 Algorithm 1 和 kernel 说明用于核对基线语义，不提供二代实现的证据。[R1][R2]

### 1.2 工具路线

**主路线：结构化 JSON → Python 原生 SVG → PNG / PDF。**

这是本任务的工程选择，不是对所有画图工具的排名。

- 使用 Python 标准库生成 SVG 元素；语义模型与版式坐标分离。
- 使用环境已有的 CairoSVG 导出 PNG、PDF；它提供相应转换接口。[R8]
- Mermaid 可作为内部草稿，但不作为最终主图。
- 不以 Graphviz 的一次自动布局直接交付论文主图；确有需要时，仅用于初始拓扑检查。
- 不增加 Figma、在线白板、外部图片生成服务、API key 等前置要求。
- 不生成伪 3D GPU、芯片插画、云服务图标、装饰性神经元或科技感背景。

最终图片必须是可重建的技术图，不是一张不可编辑的位图。

---

## 2. 架构上下文：先区分事实与候选

### 2.1 可用于取证的公开基线

公开参考仓库为 `NX-AI/flashrnn`。README 区分 `cuda`、`cuda_fused`、`triton` 等后端，并介绍 `cuda_fused` 的 WMMA 融合及 recurrent matrix / bias 的寄存器和共享内存缓存。[R1]

原论文 Algorithm 1 包含状态及 backward 所需数据的写出，也描述在部分分片情形下通过全局内存与 grid synchronization 合并结果。[R2]

**因此，不能预设以下错误对比：**

```text
错误：FlashRNN1 没有 Tensor Core，FlashRNN2 才有。
错误：FlashRNN1 没有融合，FlashRNN2 才开始融合时间循环。
错误：FlashRNN1 每个 timestep 都重新从设备内存加载全部 recurrent weights。
错误：所有 FlashRNN1 后端、所有尺寸都有同样的 grid.sync 路径。
```

这些是取证起点，不是对 Codex 本地某个 commit 的最终结论。当前源码、编译条件、后端和输入尺寸可能不同，必须在证据表中限定范围。

### 2.2 本任务携带的二代设计候选

沿用此前讨论的主线，但全部先作为 Proposed：

> 循环状态常驻、同步局部化、依赖感知流水、按 workload 选择执行方式。

候选推进顺序：

| 阶段 | 设计候选 | 出图处理 |
|---|---|---|
| 第一阶段 | state-resident、BF16、single-CTA forward | 默认主图的二代候选路径；仍需核实当前实现 |
| 第二阶段 | cluster-resident 分片、WGMMA 操作数重映射 | 仅有设计/实现依据时展开；否则作为清楚隔离的后续候选 |
| 后续阶段 | FP8、更多硬件后端 | 不放进主图的主执行路径，不宣称已经支持 |

不要把“单 CTA state-resident”自动翻译成“全部权重也都在寄存器”。权重缓存策略、状态位置、累加器位置是三个独立问题。

第一阶段默认聚焦 forward/inference 路径，但不能擅自删除现有 API 的序列输出。训练缓存、backward 支持和推理输出接口必须分别核对。

### 2.3 四类状态标签

| 标签 | 使用条件 | 图中表示 |
|---|---|---|
| Implemented | 在指定本地快照的可达实现路径中找到相应代码 | 实线框 + `IMPL` 标签 |
| Documented | 只在公开论文、README 或外部材料中有描述，尚未核对本地实现 | 实线框 + `DOC` 标签；caption 说明来源 |
| Proposed | 来自本任务候选或本地设计文档，未发现落地实现 | 虚线框 + `PROPOSED` 标签 |
| Unknown | 证据不足、互相矛盾或细节未定 | 点划线框 + `TBD` 标签 |

`Implemented` 不等于“已测正确”，更不等于“已测更快”。另设验证等级：`source_observed`、`built`、`tested`、`profiled`，只记录有证据的等级。

论文中的 FlashRNN 基线可以标为 `Documented`，标题应写 `FlashRNN paper baseline`；不能伪装成刚核对过的本地实现。

---

## 3. 仓库勘察与证据建立

### 3.1 先记录工作区，不要切分支或改代码

在仓库根目录运行可用的只读命令：

```bash
pwd
git rev-parse --show-toplevel
git status --short
git branch --show-current
git rev-parse HEAD
git diff --stat
git ls-files '*AGENTS.md' '*README*' '*design*' '*architecture*' '*roadmap*'
```

记录以下内容到 `evidence.md` 与 `architecture.json`：

- 当前仓库、分支、完整 commit、是否 dirty。
- 本任务读取到的相关源文件；dirty 或 untracked 文件应附内容 SHA-256，避免只报 HEAD 却引用未提交代码。
- 图对应的 backend、cell 类型、forward/backward、dtype、输入与输出约定。
- 二代事实来自本地代码、本地设计，还是仅来自本文的候选。

未在 Git 仓库中也继续工作：以当前目录为项目根、commit 记 `null`，生成明确标为 proposal 的交付，不编造版本号。

### 3.2 阅读顺序

按以下顺序阅读，不要只搜索几个关键词就开始画图：

1. 适用的 `AGENTS.md`、README、设计文档和变更记录。
2. 公共 API 与参数规范：输入是否为预计算投影、layout、state 返回方式。
3. backend/dispatch 选择，以及 dtype、设备能力、形状约束。
4. forward kernel 的外层循环、数据装载、矩阵乘、pointwise 更新。
5. 状态读写与存储分配：哪个指针服务 recurrence，哪个服务输出或 backward。
6. 同步、异步复制、分片与 fallback。
7. 测试、基准数据及已有 profiler 记录，用来限定结论，而非猜性能。

建议的导航命令；先通过 `git ls-files` 找到实际源码目录，缩小扫描范围，排除依赖目录与密钥文件：

```bash
rg -n 'cuda_fused|backend|dispatch|forward|inference|training' <实际源码目录>
rg -n 'wmma|mma_sync|wgmma|cp\.async|memcpy_async|TensorMap|tma|tl\.dot' <实际源码目录>
rg -n 'grid\.sync|this_grid|cooperative_groups|__syncthreads|barrier|cluster|map_shared_rank' <实际源码目录>
rg -n 'states|gates|recurrent|padding|scratch|workspace|backward' <实际源码目录>
```

尖括号内容是执行时由 Codex 自动替换的目录，不是交给用户补填。关键词命中只是入口；必须读上下文、模板实例化/配置及调用链。没有搜到某个拼写，不足以证明功能不存在。

### 3.3 必答问题

在 `evidence.md` 中先回答以下问题。没有答案就写 `Unknown`，不要停工追问。

| 类别 | 必须确认的问题 |
|---|---|
| 状态语义 | recurrent input 是整个状态还是一个分量？LSTM / sLSTM 的其他状态如何更新？ |
| 输入语义 | 输入是 raw x、预计算 Wx，还是其他 gate pre-activation？ |
| 时间循环 | 一个 kernel / CTA / cluster 覆盖几个 timestep？是否跨 kernel 保持状态？ |
| 所有权 | 一个 batch/head 的状态由一个 CTA 还是多个 CTA 拥有？沿哪个维度分片？ |
| 存储 | 权重、当前状态、累加器、input staging、输出、训练缓存分别在哪？ |
| 反馈路径 | S_t 到下一步 recurrent MMA 的实际地址空间和 layout 是什么？ |
| 同步 | 哪些参与者在等什么数据？同步范围是 warp、CTA、cluster 还是 grid？ |
| 异步 | 是否真的有异步操作、独立资源和正确的 ready/reuse 协议？ |
| 输出 | 返回所有 timestep、最后状态，还是两者？有哪些 backward-only 缓存？ |
| 分派 | 现有 selector 是否落地？能力检查、资源检查、阈值从哪来？ |

### 3.4 证据表格式

至少建立以下字段：

```text
claim_id | claim | status | scope | source_kind | source_locator | verification | figure_ids
```

例子仅说明格式，不代表实际代码已经存在：

```text
P001 | 主路径让 recurrent feedback 保留在 CTA 作用域内
     | proposed
     | BF16 / forward / candidate single-CTA path
     | task_design
     | 本执行文档 §2.2
     | none
     | fig01, fig02, fig03
```

代码证据使用实际的 `path:line_start-line_end`、符号名与工作区版本。对于干净提交，可附固定 commit 的永久链接。不要在图中印长路径，把路径放在 `evidence.md`；图中保留简短 claim ID 或写入 SVG 的 `data-claim`。

负面结论需要注明检查范围。例如写“在本次检查的 BF16 single-CTA forward 路径未发现显式 TMA 操作”，不要写“FlashRNN 从来不支持 TMA”。

---

## 4. 统一语义：在画图前定义这些对象

### 4.1 状态与输入符号

推荐使用以下图中符号，最终以实际 cell 方程与代码命名为准：

```text
U_t       当前 timestep 的输入项 / gate pre-activation，不能默认是 raw x_t
R, b      recurrent weights 与 bias
S_t       完整循环状态；可能有多个分量
h_t       S_t 中参与 recurrent matrix multiplication 的分量
A_t       recurrent matmul 的中间结果 / accumulator
Y_t       API 需要写出的当前输出
C_t       只在训练/backward 中需要保存的缓存
```

可以用抽象示意表达单步数据依赖：

```text
A_t = recurrent_matmul(R, recurrent_component(S_{t-1}))
S_t = pointwise_update(S_{t-1}, A_t, U_t, b)
```

不要将这一示意扩展成对所有 cell 形式完全一致的数学声明。若加入具体公式，必须核对当前 LSTM / GRU / sLSTM 实现。

### 4.2 三类不能混为一谈的状态写入

1. **Feedback state**：下一时间步运算需要的状态。
2. **Sequence output / state history**：API 要求保存的逐步结果。
3. **Backward-only cache**：反向传播所需的 gates、intermediates 等。

图 01–03 都必须保持这一区分。

例如，`S_t` 留在片上用于下一步计算，同时将 `Y_t` 写入全局输出数组，是完全可以成立的设计。不能为了表现 state-resident 就把合法输出箭头删掉。

同理，“推理不分配 backward-only cache”不等于“推理不写任何序列输出”。只有实际 API 提供并选择 final-state-only 模式时，才能画“只返回最终状态”。

### 4.3 地址空间与硬件流量分开

`global-address-space access` 不自动等于一次实际 DRAM/HBM 事务；是否命中缓存、是否合并、实际字节量需要运行时证据。没有 profiler 数据时，图中使用 `Global state buffer` 或 `Global-memory path`，不写“每步两次 HBM 访问”。

`register-resident` 也要区分设计意图与编译后 spill 结果。只读源码时，可写 `register-resident by design; spill behavior unverified`。

同样，CTA / warp / warpgroup 是执行与协作范围，不是新的物理存储层。不要画出不存在的“warpgroup memory”。

---

## 5. 图 01：新旧机制对比总览

文件基名：`fig01_architecture_overview`  
建议画布：`1920 × 1080`，可因内容适度调整。  
读者目标：不用读代码，能在十几秒内说出二代改变了哪条反馈路径。

### 5.1 版式

上方标题与范围标签；中间左右两栏；下方三条编号机制说明与图例。

```text
                    FlashRNN2: State-Resident Recurrent Execution
                    [cell / backend / dtype / forward / evidence status]

    (a) Selected FlashRNN baseline       (b) FlashRNN2 target / implementation

    同样的输入、权重、MMA、状态更新         同样的输入、权重、MMA、状态更新
    突出实际的状态反馈路径                  突出候选或实际的局部状态反馈路径
    标注该路径的同步范围                    标注该路径的同步范围
    保留 API 要求的输出                    保留相同语义的输出

       ① State feedback locality    ② Synchronization scope    ③ Legal overlap
```

左右栏使用相同节点名称、相同语义颜色、相同阅读方向，避免通过改名或不对称删减制造优势。

### 5.2 必须画出的对象

每栏包含：`U_t`、`R / b`、`Recurrent MMA`、`State update`、`S_t` 反馈、必要输出。

如果核实所选基线反馈通过全局状态缓冲，可画：

```text
State update → Global state buffer → Next recurrent MMA
```

如果核实二代路径或采用 §2.2 的 proposal，可画：

```text
State update → Local state / layout transition → Next recurrent MMA
                     └→ required output store
```

**“Local state / layout transition”不能被省略为魔法箭头。** 没有确定寄存器或 shared-memory 布局时，在其下写 `layout TBD`。不能声称两个 MMA 之间无需重排或转换。

### 5.3 不允许的表现

- 不把一代画成 unfused PyTorch，除非明确另设第三种 baseline。
- 不把可选 cluster 路径塞进 single-CTA 的同一个主框。
- 不把所有同步抹掉后写 `synchronization-free`。
- 不加任何无来源的 speedup、带宽提升或能耗数字。
- 不通过缩短二代流程长度暗示已经测到延迟下降。

### 5.4 可使用的标题与 caption 草稿

标题：`State-Resident Recurrent Execution`。

中文 caption 草稿：

> FlashRNN2 的设计重点是改变循环状态反馈的驻留位置与同步范围，而不是重新发明融合计算。左右两侧采用相同的输入和输出语义；右侧将状态反馈路径与 API 必须保存的输出区分开来。虚线表示尚未实现或尚待验证的设计路径。具体适用范围和证据见正文。

这段 caption 是 proposal 版本的初稿；必须按最终核实结果改写，不可机械保留与图冲突的文字。

---

## 6. 图 02：状态驻留、硬件位置与生命周期

文件基名：`fig02_state_residency`  
建议画布：`1920 × 1200`。  
读者目标：明确“什么常驻、常驻多久、谁拥有、什么时候需要通信”。

### 6.1 上半区：空间映射

主图只展开默认的 single-CTA 路径，至少区分：

```text
Execution scope: one CTA / batch-head tile（实际划分以代码为准）

Thread register state / fragments
CTA shared-memory staging / exchange
Compute resources: recurrent MMA + pointwise update

Global address space:
  inputs | weights backing storage | required outputs | optional training cache
```

将执行范围边界与存储位置边界用不同标题表达，不要暗示“一个 CTA 独占一个 SM”。

每个重要对象附以下四项的简短信息，过长部分放 caption 或表格：

```text
owner / location / lifetime / access pattern
```

示意：

```text
S_t: owner=<actual tile>; location=<verified or TBD>;
     lifetime=until all next-step consumers finish; access=recurrent feedback
```

### 6.2 下半区：生命周期条带

沿 `init → t → t+1 → finish` 展示：

| 对象 | 需要表达的语义 |
|---|---|
| R / b | 初始化加载、在循环中的重用；不预设全量可放入寄存器 |
| S_t | 产生、布局转换、下游消费、被替换/复用的边界 |
| U_t staging | input tile 的装入、ready、consume、reuse |
| A_t | matmul 产生、pointwise 消费、存储释放 |
| Y_t / state history | API 要求的逐步写出或最终写出 |
| C_t | 仅在对应训练路径存在时展示 |

没有实现双缓冲时，不要把两个 slot 画成既成事实。若双缓冲只是设计，标为 `Proposed double buffering`。

### 6.3 cluster 扩展的处理

只有明确设计或实现需要时，增加独立的右侧 inset 或单独附图：

- 显示多个 CTA 及每个 CTA 所属的局部 shared memory。
- 标出沿哪个轴分片：hidden、gate、batch 或其他真实轴。
- 标出通信内容是 state shard、partial accumulator，还是其他对象。
- 标出远端 shared-memory 访问及实际 cluster 同步。
- 不把 DSM 画成无访问成本、容量无限的平坦全局共享池。

NVIDIA 文档说明 cluster 内可以访问其他 block 的 shared memory，但这只能作为硬件机制参考，不能证明 FlashRNN2 已使用该机制，也不能证明任意 GPU 都支持相同配置。[R6]

没有相关证据时，主图不展开 cluster 细节；在 caption 标注其为后续候选即可。

### 6.4 容量与硬件数字

图中优先写符号约束，如 `state + staging + accumulators fit resource budget`。

若填写具体 KB、register/thread、block size、cluster size，必须记录设备型号、compute capability、配置和来源。理论容量、每 SM 容量、每 CTA 上限、实际可分配值不得混用。

---

## 7. 图 03：依赖感知流水时序

文件基名：`fig03_dependency_pipeline`  
建议画布：`1920 × 1080`。  
读者目标：知道“为什么能重叠”以及“为什么不能把整个 RNN 时间维并行化”。

### 7.1 泳道与依赖

至少画三个 timestep，推荐泳道如下。没有独立执行资源的操作不要假画成可同时运行。

```text
Input movement / prefetch
Recurrent MMA
Pointwise state update
State layout / visibility / synchronization
Required output stores（没有异步或独立路径时，合并回真实执行泳道）
```

对一般的完整状态消费路径，保守依赖链为：

```text
MMA_t → Update_t → StateLayout_t → StateReady_t → MMA_(t+1)
```

`StateLayout_t` 可以是寄存器重排、shared-memory staging 或实际无需转换；必须据实现而定。

若实现采用合法的 tile-level 部分状态生产/消费，允许更细粒度重叠，但必须提供 tile 级生产者—消费者依赖证据。不能仅凭“有 WGMMA/TMA”就打破 recurrent dependency。

### 7.2 能表示的候选重叠

以下是需要核实的候选，不是默认已实现能力：

- 已知且独立的 `U_(t+1)` 装载与当前步计算重叠。
- 不同 batch/head 的独立 recurrence 之间交错执行。
- 当前计算与已完成且有独立安全缓冲的输出写出重叠。

“异步接口存在”不足以证明“端到端发生了有效重叠”。需要区分提交、完成和消费。没有运行时 trace 时，只能说 `intended overlap` / `logical schedule`。

### 7.3 必须画出的 ready 与 reuse

若采用 staging buffers，显示：

```text
slot writable → load issued → load complete / ready
              → consumer uses slot → consumer complete / reusable
```

不能在 consumer 还读 slot 时画下一个 producer 覆盖相同 slot。实际使用 barrier phase、wait group 或其他协议时，标出协议名；不明时写 `buffer protocol TBD`，不能装饰性添加几个 barrier 图标。

若状态同时服务下一步消费与异步输出，也要表达全部消费者结束后的复用条件。

### 7.4 横轴与持续时间

没有测量数据时：

```text
Logical execution order — not to scale; no measured speedup implied
```

所有宽度只是排版，不使用 `μs`、cycle、吞吐率等单位。不要让前后对比图的相对长度暗示实测加速。

有 trace 时，需记录设备、输入形状、dtype、版本、测量工具与原始文件，才可以使用真实时间比例。

### 7.5 WGMMA / TMA 命名边界

FlashAttention-3 展示了异步矩阵运算、数据移动与调度组织，但其具体算法依赖不是 RNN 的依赖。[R3]

只有确定当前目标使用这些原语时，图中才写 `WGMMA`、`TMA`。否则写 `MMA engine`、`Input prefetch`，并在注释中说明 instruction/backend TBD。

若描述 WGMMA 操作数位置，必须核对 A/B 操作数形式、layout 与寄存器/SMEM 来源。不能因为一代缓存权重到寄存器，就直接画出“WGMMA 无转换读取全部寄存器权重”。

---

## 8. 图 04：workload-aware 分派与 fallback

文件基名：`fig04_execution_dispatch`  
建议画布：`1600 × 1000`。  
读者目标：知道这些优化适用在哪，以及不满足条件时系统怎么办。

### 8.1 图的输入

按实际 selector 参数组织，不必把所有字段印进一个小框：

```text
B, T, heads, head_dim
cell type, dtype, tensor layout
forward / training / backward
output contract
GPU capability and resource limits
```

`heads` 等符号应与前三张图一致。

### 8.2 决策顺序

```text
Request
  → semantic compatibility
  → hardware/backend capability
  → resource feasibility
  → selected execution policy
  → output contract preserved
```

候选叶节点可包含：

```text
Single-CTA state-resident path
Cluster-resident sharded path（有依据时才展开）
Existing fused / alternating / other supported fallback
Unsupported request → explicit error（仅实际无合法 fallback 时）
```

叶节点必须对应实际后端或清楚标出的设计候选。不要把 baseline 的所有后端一股脑画成二代已接好的自动 fallback。

### 8.3 禁止猜阈值

没有 selector 源码或 benchmark 依据，不得写：

```text
B <= 8 一定选 single CTA
D > 128 一定选 cluster
Hopper 上一定选 TMA
FP8 永远比 BF16 快
```

未确定时使用 `fits CTA resource budget?` 等符号谓词，并标注 `policy proposed / threshold TBD`。

没有实测分派策略时，标题用 `Candidate execution policy`，不是 `Optimal runtime autotuner`。

小 batch 的 padding、tail、mask、专门 kernel 只在有对应证据时展开；不要凭此前讨论给它填一个固定 padding 倍数。

---

## 9. 可选图 05：架构到源码的映射

文件基名：`fig05_code_map`。

仅当当前仓库存在可辨认的二代模块且有必要用于 review 时生成。参考 C4 的组件分层思想，但不要声称这是严格的 C4 container diagram。[R4]

推荐结构：

```text
Public API → request/config validation → dispatch
                                  ├→ kernel path A
                                  ├→ kernel path B
                                  └→ fallback

Data/layout helpers and synchronization helpers
Tests validating the contract
```

每个组件只显示实际的简短符号/文件名与一句职责，完整路径在 `code_map.md`。没有真实的 `StateManager`、`PipelineEngine` 类时，不要为了图好看发明这些组件。

---

## 10. 视觉规范

以下为统一排版参数，不是待用户选择的选项。

### 10.1 整体风格

白底、二维、低饱和填充、清楚的边界、克制的颜色。主图强调计算、状态、数据移动和同步，不使用产品宣传图风格。

每个 panel 建议 5–9 个关键节点。完整主图超过约 14 个主要概念时，先拆图，不靠缩小文字解决。

图内默认英文，以方便论文、README 和报告复用；中文详细解释放 caption 文件。不复制或随交付分发字体文件。

### 10.2 色彩语义

| 对象 | 描边/强调色 | 浅填充 |
|---|---|---|
| recurrent state / feedback | `#7556A6` | `#F2EDF8` |
| compute / MMA / pointwise | `#3568A0` | `#EDF3FA` |
| input / data movement | `#187E77` | `#EAF5F2` |
| synchronization / constraints | `#A66B1F` | `#FFF4E3` |
| storage / scope boundary | `#667085` | `#F7F8FA` |

文本使用深色。颜色只表示对象类别，不表示“已实现/未实现”，实现状态依靠文字 badge 与边框线型。

### 10.3 字号与几何

以 1920 px 宽画布为例：

```text
outer margin: 48–64 px
panel gap: 40–56 px
title: 36–42 px
panel title: 28–32 px
node label: 24–28 px
secondary label: 21–24 px
border / ordinary edge: 2–3 px
feedback emphasis: 3–4 px
```

关键文字不小于 24 px。若制作约 178 mm 宽的论文双栏图，1920 px 画布上的 24 px 字符仅约 6.3 pt；论文版本应采用更大的字号、减少内容或拆成子图，不能把屏幕版直接缩到不可读。

每个节点最多两行主体文字；复杂方程、长约束、路径和测量环境放 caption。

### 10.4 连线规则

- 主数据流从左向右；状态反馈走单独的下方回路，方向必须一眼可见。
- 箭头连接对象边界的固定 anchor，不插进文字。
- 优先正交或少量折线；减少交叉，不跨越其他节点实体。
- 数据流、控制依赖和同步事件分别标明；不要让虚线同时承担多种无法分辨的语义。
- 时序图的同步事件使用带标签的竖线/依赖标记，避免与 proposal 框的虚线混淆。
- panel 外保留图例：状态类型、线型、`not to scale` 和适用范围。

### 10.5 SVG 可编辑性

最终 SVG 必须满足：

```text
有 viewBox、title、desc。
文字默认保留为 <text>/<tspan>，不是整页轮廓化。
每个节点、连线、panel 有稳定的 id。
关键对象带 data-claim / data-kind / data-status。
语义分组使用 <g>。
不嵌入 PNG 冒充矢量图。
不依赖外部图片、网络字体、脚本或 foreignObject。
使用简单 shape、path、marker、text，保证导出兼容性。
```

caption 中提供无障碍 alt text。不要把字体文件打包进产物。

---

## 11. 文件结构与单一事实来源

生成以下目录。已有用户文件不得被无条件覆盖；先检查内容与版本，只更新本任务生成的文件。

```text
docs/architecture/flashrnn2/
├── README.md                    # 看图顺序、范围、架构解释、生成命令
├── evidence.md                  # 证据表、版本、限制与未知项
├── captions_zh.md               # 每图中文 caption 与 alt text
├── captions_en.md               # 每图英文 caption 与 alt text
├── review.md                    # 实际执行的检查、修复与剩余问题
├── code_map.md                  # 可选：源码映射
├── architecture.json            # 语义模型、claims、版本与适用范围
├── layouts.json                 # 版式坐标、anchor、边路由
├── style.json                   # 全套图共用的样式参数
├── requirements-diagrams.txt    # 实际使用的额外绘图依赖，避免改项目依赖
├── environment.txt              # 实际工具版本与字体选择；不包含秘密信息
├── src/
│   ├── render_figures.py         # 命令行入口与导出
│   ├── svg_primitives.py         # 框、箭头、泳道、图例、文本布局
│   └── validate_figures.py       # 语义约束与几何检查
└── figures/
    ├── fig01_architecture_overview.svg / .png / .pdf
    ├── fig02_state_residency.svg / .png / .pdf
    ├── fig03_dependency_pipeline.svg / .png / .pdf
    ├── fig04_execution_dispatch.svg / .png / .pdf
    └── fig05_code_map.svg / .png / .pdf     # 可选
```

`architecture.json` 是事实/设计语义的来源；`layouts.json` 只描述布局。不要在渲染源码中硬编码另一套架构结论。

### 11.1 最小语义模型示例

下面是格式样例，内容全部为 proposal；执行时扩充为实际架构，不要把样例当作取证完成。

```json
{
  "schema_version": 1,
  "project": "FlashRNN2",
  "repository_snapshot": {
    "commit": null,
    "dirty": null,
    "relevant_file_hashes": {}
  },
  "mode": "proposal",
  "scope": {
    "cell": "TBD",
    "phase": "forward",
    "dtype": "BF16 candidate",
    "hardware": "TBD",
    "output_contract": "preserve verified API; details TBD"
  },
  "claims": [
    {
      "id": "P001",
      "text": "Keep recurrent feedback local to the chosen CTA execution scope",
      "status": "proposed",
      "verification": [],
      "evidence": [
        {
          "kind": "task_design",
          "locator": "execution-document:section-2.2"
        }
      ]
    }
  ],
  "nodes": [
    {
      "id": "state_next",
      "label": "S_t",
      "kind": "state",
      "status": "proposed",
      "claim_ids": ["P001"],
      "attributes": {
        "owner": "candidate CTA tile",
        "address_space": "TBD",
        "lifetime": "until all next-step consumers finish"
      }
    },
    {
      "id": "mma_next",
      "label": "Recurrent MMA (t+1)",
      "kind": "compute",
      "status": "proposed",
      "claim_ids": ["P001"]
    }
  ],
  "edges": [
    {
      "id": "state_feedback",
      "source": "state_next",
      "target": "mma_next",
      "kind": "feedback",
      "status": "proposed",
      "label": "layout / visibility transition TBD",
      "claim_ids": ["P001"]
    }
  ],
  "schedule": {
    "scale": "logical_not_to_scale",
    "events": [],
    "dependencies": []
  }
}
```

实际模型还应记录：各图包含哪些节点、事件的资源/时间坐标、数据 slot、ready/reuse 条件、输出契约、分派谓词与来源。不要为了满足模型字段假造数值。

### 11.2 脚本接口约定

编写脚本，使这些命令在仓库根目录可运行：

```bash
python docs/architecture/flashrnn2/src/render_figures.py \
  --model docs/architecture/flashrnn2/architecture.json \
  --layout docs/architecture/flashrnn2/layouts.json \
  --style docs/architecture/flashrnn2/style.json \
  --out docs/architecture/flashrnn2/figures \
  --formats svg,png,pdf

python docs/architecture/flashrnn2/src/validate_figures.py \
  --root docs/architecture/flashrnn2
```

脚本应有类型注解、参数检查、清楚的异常消息、稳定的输出顺序。读取/写入路径应相对传入参数解析，不依赖开发者机器的绝对路径。

默认使用环境现成的绘图依赖。确需安装时，遵守仓库权限和依赖规则，用独立虚拟环境，不修改生产依赖，不用 `sudo`。

CairoSVG 官方文档列出 `svg2png` 与 `svg2pdf` 接口。[R8] 安装和导出不能只在文档中写“应该能运行”，需要实际运行并记录所用版本。

PNG 默认导出 2 倍尺寸；PDF 从 SVG 直接转为矢量，不把 PNG 嵌入 PDF 冒充矢量版本。PDF 的页面物理尺寸需明确，论文版与屏幕展示版分别设置字号和尺寸，避免仅修改 DPI。

渲染应尽量确定性：不把当前时间或随机数写进布局。相同模型、布局、样式和工具版本应产生相同 SVG。构建日志时间放在 `review.md`，不要破坏 SVG 内容复现。

---

## 12. 逐步执行流程

### 阶段 A：完成取证与范围界定

先产出 `evidence.md` 和 `architecture.json` 的初稿。

判断模式：

```text
有二代实现：implementation-focused + 必要的 proposed extension。
只有设计：proposal-focused + 可核对的一代基线。
只有一代：documented/source-observed baseline + 本文的明确 proposal。
什么仓库都没有：依据公开基线和本文候选生成 proposal；实现细节标未知。
```

不要为缺少二代代码中断任务，不要将论文描述自动升级成本地实现。

### 阶段 B：写四句图的结论

在 README 中为每张图先写一句“读者看完应理解什么”，不要超过两行。

检查四张图是否分别承担：机制、空间、时间、适用条件。若重复，删减。

### 阶段 C：建立结构化图模型

先建立语义节点、边、claims 和 schedule dependencies，再写布局。

图 03 必须把 recurrent loop 按时间展开为 DAG 做依赖检查；静态总览中的 feedback loop 可以有环，不得对整套静态图错误应用 DAG 检查。

### 阶段 D：渲染第一版

实现共用图元与四张版式，生成 SVG，然后导出 PNG/PDF。不要到此结束。

### 阶段 E：查看与修改

在当前环境具备图像查看能力时，打开每张 PNG 与导出的 PDF 页面渲染，检查整体图和关键局部。

至少检查：

```text
整体阅读顺序
状态反馈回路
节点边界与箭头端点
所有小字和角标
图例与 proposal 标识
时序的 ready/reuse 与真正依赖
```

修复第一轮发现的问题，重新生成后再次查看。环境不具备图像查看能力时，在 `review.md` 明确写 `visual inspection not performed`，不能用 XML 解析成功替代视觉验收。

### 阶段 F：一致性复核与交付

对照证据逐条审查各图和 caption，更新 README 的讲解顺序。执行验证脚本与 `git diff --check`（若有 Git），确认未修改 kernel 或用户其他文件。

最终答复必须列出已生成文件、实际运行命令、检查结果和剩余未知项；不要只说“已按要求完成”。

---

## 13. 验收标准

### 13.1 语义验收：阻断级

出现以下任一问题，视为未完成：

- 把 Proposed / Unknown 内容画成当前已实现。
- 错画 baseline，把一代已有融合、WMMA 或权重缓存包装成二代首次支持。
- 把 recurrent feedback、sequence output、backward-only cache 混成同一条箭头。
- 省略 state-layout / visibility 的必要转换而暗示零成本直接反馈。
- 在没有 tile-level 依赖证明时，把同一条 recurrence 的相邻整步 MMA 画成任意并行。
- 将 producer 的 buffer reuse 安排在 consumer 完成之前。
- 把 CTA 内同步直接替代仍有跨 CTA 数据依赖的全局同步。
- 图中出现无证据的硬件支持、精度支持、分派阈值或性能数字。
- 将仅有代码存在写成测试通过或实测加速。
- caption、图和 evidence 中对范围、状态、输出语义互相矛盾。

### 13.2 程序化验收

校验脚本至少检查：

```text
[ ] 模型字段、状态枚举与文件引用有效。
[ ] 所有关键节点/边都指向存在的 claim。
[ ] implemented claim 至少有本地 source evidence；profiled claim 有原始测量来源。
[ ] unknown/proposed 在最终 SVG 中可见，而不只是藏在 JSON。
[ ] 所有 SVG 可解析、id 唯一、viewBox 有效。
[ ] 不包含 script、foreignObject、外部资源或嵌入位图。
[ ] 布局对象在画布内；非包含关系的节点不意外重叠。
[ ] 边不穿越非目标节点；箭头与标签不相互覆盖。
[ ] 对时间展开后的 schedule，依赖边在逻辑顺序中成立。
[ ] 声称复用的 buffer，复用发生在所建模的全部消费者完成后。
[ ] 四张主图的 SVG/PNG/PDF 均存在且可打开。
[ ] caption 与 figure basename 一一对应。
```

文本溢出应使用实际渲染器的字体度量或浏览器 bbox 辅助检查，不能仅按字符数断言没有裁切。自动几何校验发现不了全部视觉问题，必须保留人工/视觉复核结果。

自动依赖检查只检查已建模的约束；它不是 kernel 正确性证明。

### 13.3 视觉验收

- 在常规 README 浏览宽度下能读清关键标签。
- 论文版按目标印刷宽度检查，而不是只在 200% 放大时看起来清楚。
- 左右对比的节点命名与语义颜色一致。
- 反馈箭头方向明显；没有意外交叉和悬空端点。
- 图例能解释所有线型与状态标签。
- 去掉颜色仍能靠文字、边界和线型读懂。
- PDF 的文字和线条放大后仍清晰；SVG 保留可编辑对象。

### 13.4 诚实的降级规则

SVG 是最低限度必须完成的原始图。若安装权限、网络或导出依赖阻止生成 PNG/PDF，仍提交 SVG、源码、模型和 caption，并把缺失格式明确列为未完成。

同理，未能打开 PDF、未能视觉检查、未做 profiling，都必须直写，不得用计划中的命令或虚构日志填“通过”。

---

## 14. README 中应提供的讲解顺序

按以下逻辑组织中文解释，不要先讲 GPU 指令清单：

**第一段：问题。** 指定基线和 workload，解释当前瓶颈候选是状态反馈、同步或资源组织中的哪一项。没有测量就写“优化目标”，不是“已证实瓶颈”。

**第二段：核心变化。** 指着图 01 解释哪条路径发生改变，以及哪些已有能力被继承。

**第三段：如何成立。** 指着图 02 解释所有权、驻留位置、生命周期、容量与布局代价。

**第四段：时间上是否合法。** 指着图 03 解释不可打破的递归依赖和候选重叠窗口。

**第五段：边界。** 指着图 04 解释哪些请求不能走主路径、fallback 怎样保留语义。

结尾列出真正需要下一步实现或验证的问题，不宣称所有设计风险都已解决。

推荐使用的一句核心表述：

> FlashRNN2 的候选架构把优化重点放在循环状态反馈的局部性与依赖感知调度上，在不改变输入输出语义的前提下，为不同 workload 选择合适的执行范围。

只有代码和测试支持时，才能把“候选架构”改成“实现”。

---

## 15. 最终交付回复模板

```text
已生成的图：
- fig01：机制对比；范围 / 状态为……
- fig02：状态驻留与生命周期；范围 / 状态为……
- fig03：依赖时序；logical schedule 或 measured timeline 为……
- fig04：执行路径选择；已实现 selector 或 proposed policy 为……

入口：docs/architecture/flashrnn2/README.md
源文件：architecture.json、layouts.json、style.json、src/
产物：每张图实际生成的 SVG / PNG / PDF

实际运行：列出构建、校验和 diff 检查命令及结果。
视觉检查：列出确实查看过的格式、发现并修复的问题。
尚未确认：列出仍为 Proposed / Unknown 的架构要素。
未完成：如有缺少的导出格式或验证，明确列出。
```

不要只交出一张 PNG，不要只交出本文的复述，不要用“架构已验证”概括仅完成了出图。

---

## 16. 参考资料与使用边界

以下是执行文档编写时核对过的公开资料。Codex 执行时应优先核对本地快照；公开文档只说明所引机制或基线，不代表 FlashRNN2 的实现状态。

### [R1] NX-AI / FlashRNN 源仓库与 README

URL: https://github.com/NX-AI/flashrnn  
用途：核对已有 backend、接口示例、WMMA 融合和权重缓存说明。  
限制：默认分支会变化；本地实现图必须固定 commit 与实际工作区。

### [R2] FlashRNN 原论文

URL: https://arxiv.org/html/2412.07752v2  
重点：Algorithm 1、§5.2、相关 kernel 说明。  
用途：区分权重重用、状态反馈、结果合并与输出保存。  
限制：论文伪代码不是每个本地模板实例化的逐行实现。

### [R3] FlashAttention-3 原论文

URL: https://arxiv.org/html/2407.08608v2  
重点：§3.1–3.2、Figure 1–2。  
用途：借鉴资源泳道、依赖与迭代调度的表达方式。  
限制：不复制图，不把 attention 的算法依赖当成 RNN 的依赖。

### [R4] C4 Component Diagram

URL: https://c4model.com/diagrams/component  
用途：可选源码组件映射的职责分层参考。

### [R5] C4 Dynamic Diagram

URL: https://c4model.com/diagrams/dynamic  
用途：编号交互、区分静态结构与运行时协作的表达参考。

### [R6] NVIDIA Hopper Tuning Guide

URL: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html  
重点：TMA、Thread Block Clusters、Distributed Shared Memory。  
用途：核对相关硬件机制的含义与约束。  
限制：不是所有 GPU 的统一能力表，也不是二代已经采用这些机制的证据。

### [R7] OpenAI Codex 官方执行与项目指引

URL: https://developers.openai.com/cookbook/articles/codex_exec_plans  
URL: https://developers.openai.com/codex/agent-configuration/agents-md  
用途：明确任务文档、可检查结果和仓库指令的组织方法。  
注意：本文作为显式读取的任务文件使用；不要求替换已有 AGENTS.md，也不假定仅凭文件名就自动加载全文。

### [R8] CairoSVG 官方文档

URL: https://cairosvg.org/documentation/  
URL: https://cairosvg.org/svg_support/  
用途：SVG 到 PNG/PDF 的导出接口与兼容性限制。  
注意：使用简单 SVG 图元、实际安装字体与导出检查，不假定复杂 CSS 或字体排版在所有渲染器中一致。

---

## 17. 自包含完成标准

本任务完成后，另一个开发者应能在不重新设计图的情况下：

1. 读 README 理解四张图的架构故事与边界。
2. 通过 evidence 找到每条关键架构判断的来源。
3. 修改模型中的真实设计变化，运行脚本重新出图。
4. 将 SVG 继续编辑，或将 PDF 放入论文与报告。
5. 明确区分现有实现、文献基线、设计提案和未知细节。

**优先级：架构语义正确 > 证据清楚 > 图能解释机制 > 可编辑可复现 > 装饰性美观。**
