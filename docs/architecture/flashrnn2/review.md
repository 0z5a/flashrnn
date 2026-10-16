# 四张架构图：实际执行与验收

日期：2026-10-14。此验收针对出图与已建模依赖，不是 kernel 正确性、性能或有效异步 overlap 的新 GPU 验证。

## 实际执行

- 已读取 infra/AGENTS.md、当前二代五个核心模块、默认公开接口、PreparedSequence/ExecutionPolicy 调用链、固定一代 triton_fused forward、设计与代表 PTX。根目录同名执行文件缺失，使用用户附件；已保存原字节与 SHA。
- 先产出 evidence.md 与 architecture.json，再添加语义节点/边/claim/schedule，最后写 layouts.json 和独立 style.json。
- 使用标准库 XML 生成 SVG；CairoSVG 2.9.1 + 已有 Cairo 1.18.4 导出 2x PNG 与直接矢量 PDF。仅在本目录隔离环境添加绘图依赖。

实际构建与检查命令：

```bash
docs/architecture/flashrnn2/.venv/bin/python docs/architecture/flashrnn2/src/write_docs.py --root docs/architecture/flashrnn2
docs/architecture/flashrnn2/.venv/bin/python docs/architecture/flashrnn2/src/run_diagrams.py all
```

run_diagrams.py 依次执行约定的 render_figures.py、validate_figures.py 与 inspect_exports.py 入口。inspect 使用 Poppler `pdftoppm -png -singlefile -r 150` 独立渲染全部八份 PDF。每次报告都保留真实 input SHA、退出码与输出文件。

## 已修复的问题

1. accessible title 与可见标题的 SVG id 重复：分开 stable id。
2. 图 01 继承能力脚注在论文版过宽：缩短标签；没有缩小论文最低字号。
3. 图 01 最后状态的 baseline 画法原暗示独立写出：改为从 history 到 last-state view 的别名关系。
4. 图 01 补充完整状态对下一 state update 的真实依赖，并明确 bias 进入 gate preactivation，R 进入 MMA。
5. 图 02 文本越界、脚注压住图例/边框、连续空格丢失对齐：调整框高和脚注位置，改成独立时间列与生命周期条带。
6. 图 02 生命周期明确 full/hidden 的逐步写出与 last 的逻辑 final h 写出，避免暗示 last 也保存全部 history。
7. 图 03 将候选 input protocol 与当前 recurrence 的 state ready 连接，保留 issue/complete/consume/reuse 与三个 timestep 的真实依赖；相邻 update 的输入和输出使用不同 anchor。
8. 图 04 config 连线穿过文字：引入实际 caller composition 节点并重路由；两种配置路由避免相交。
9. SVG 对象类别和实现状态采用独立图例；未知和候选都有可见文字/线型。一次查看恰逢 PNG 正在覆盖读取失败，等待完成后成功查看，并将导出改为完整 .partial 文件原子替换。

## 最终程序化验收

- **PASS**：15 份相关 source/design/measurement 快照 SHA、23 条 claim（Implemented 16 / Documented 1 / Proposed 3 / Unknown 3）。
- **PASS**：8 个 SVG 的 XML、unique id、viewBox、title/desc、data claim/status、文字可编辑；无 script/foreignObject/外部图片/嵌入位图。
- **PASS**：548 行文字用真实 Cairo 字体度量检查；没有节点越界、非目标节点穿线或文字/箭头遮挡。
- **PASS**：22 个逻辑事件、24 个依赖、3 个 buffer version。静态 feedback 图不错误地套用 DAG 检查。
- **PASS**：5 个负控制确实被拒绝，覆盖 input 未 ready 就消费、input 提前覆盖、carry 提前复用、下一 MMA 提前执行和时间展开图成环。
- **PASS**：8 个 SVG 在相同模型/布局/样式下重复生成，内容 SHA 全部不变。
- **PASS**：8 PNG + 8 PDF 可打开；PNG 全部 2x。8 PDF 均有真实可提取文字且 embedded image count 为 0。screen 宽 254 mm，paper 宽 178 mm；paper 最小字体为 8.409 pt。
- **Git 限制**：实际运行全仓 `git diff --check`，12 秒超时，没有把它写成 PASS。目录的 git status 确认为新 untracked 文档；已另外实际检查所有本任务编写文本的行尾空白，通过。原源码和历史文档快照保留原字节。

## 最终视觉验收

已逐张查看四张屏幕 PNG、四张论文 PNG、八份独立 PDF 页渲染；也查看普通 README 宽度、关键局部、灰度 proof。

- 图 01：反馈方向、完整状态消费者、bias/R 分流、baseline history view 与二代 compute carry 清楚。
- 图 02：CTA scope 与真实地址空间分开，布局 TBD 可见，生命周期分列和借用输出脚注清楚。
- 图 03：三个 timestep 依赖、候选输入 ready/reuse、下一 MMA 的 state dependency 清楚。
- 图 04：未知 key 的 MMA fallback、identity/非法输入 error、显式 caller composition、隔离的 Proposed 策略清楚。
- 按目标 178 mm 印刷宽度的页渲染和实际字号检查；未发现剩余文字裁切或边路由遮挡。黑白下靠 badge 与边界线型仍可区分状态。

视觉 proof 位于 checks/proofs；独立 PDF 页在 checks/pdf-render。它们是验收图，不是科学测量结果。

## 交付范围与剩余未知

四张主图的两套 SVG/PNG/PDF、中英 caption/alt text、源码、单一语义 JSON、独立布局/样式、证据与实际检查均完成；无缺失格式。可选图 05 用 code_map.md 的实际符号映射代替，避免重复四张主图。

全形状物理 layout/spill、独立异步资源与 slot 协议、有效 overlap、自动 capability/resource policy、fallback registry、cluster/WGMMA/FP8 和分布式/backward 资格仍未确认或未实现。没有新 GPU 运行、没有新性能数字、没有将文献机制升级为当前二代支持。

本轮出图只写 docs/architecture/flashrnn2；kernel、公共接口、生产依赖、原论文编辑器和已有用户改动未修改。随后用户单独授权的 PR 发布准备另外记录在 checks/pr-preflight，不改变本次架构验收的科学范围。
