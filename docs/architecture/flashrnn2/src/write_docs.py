"""Export evidence/captions/code maps from the semantic JSON, without new claims."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();root=args.root
    model=json.loads((root/"architecture.json").read_text())
    snap=model["repository_snapshot"]
    for lang in ("zh","en"):
        lines=["# "+("图注与无障碍说明" if lang=="zh" else "Captions and accessible descriptions"),""]
        for fig in model["figures"]:
            lines += ["## "+fig["id"],"",fig["caption_"+lang],"","Alt text: "+fig["alt_"+lang],""]
        (root/f"captions_{lang}.md").write_text("\n".join(lines).rstrip()+"\n")
    lines=["# 架构证据、版本与未知项","",
        "本文件由 architecture.json 生成。Implemented 只表示指定本地路径有可达源码；built 只适用于两个保留的代表 PTX。没有把源码、编译、历史测量或出图验收当作本轮 GPU 正确性/性能 PASS。","",
        "## 工作区与选择的基线","",
        f"输出仓库：{snap['root']}，{snap['branch']} @ {snap['commit']}；原有 tracked 改动保留。",
        f"二代：{snap['v2']['root']}，{snap['v2']['branch']} @ {snap['v2']['commit']}。",
        "二代 git status/diff/rev-parse 的部分命令超时，dirty 状态记 Unknown；HEAD/ref 只读解析得到 commit。所有图按当前文件 SHA 取证，五个核心文件与冻结 experiment manifest 匹配。",
        f"冻结 manifest SHA256：{snap['v2']['source_manifest_sha256']}。",
        f"选定基线：{snap['baseline']['backend']} @ {snap['baseline']['commit']}。",
        "同名根目录执行文件不存在，使用用户提供的 Downloads 附件，保存在 evidence-snapshots/EXECUTION_SPEC.md。",
        f"执行规范 SHA256：{snap['task_document']['sha256']}。","",
        "## 范围与模型","",
        json.dumps(model['scope'],ensure_ascii=False,indent=2),"",
        "## 架构证据表","",
        "| claim_id | claim | status | scope | source_kind | source_locator / symbol | verification | figure_ids |",
        "|---|---|---|---|---|---|---|---|"]
    for c in model["claims"]:
        loc="<br>".join(e['locator']+(" / "+e['symbol'] if e.get('symbol') else '') for e in c['evidence'])
        row=[c['id'],c['text'],c['status'],c['scope'],", ".join(sorted({e['kind'] for e in c['evidence']})),loc,", ".join(c['verification']) or 'none',", ".join(c['figure_ids']) or 'context-only']
        lines.append("| "+" | ".join(x.replace("|","\\|") for x in row)+" |")
    lines += ["","## 所有权 / 存储 / 生命周期 / 访问","",
        "| 对象 | owner | location | lifetime | access pattern | claims |",
        "|---|---|---|---|---|---|"]
    for obj in model["objects"]:
        lines.append("| "+" | ".join([obj[k] for k in ['id','owner','location','lifetime','access_pattern']]+[", ".join(obj['claim_ids'])])+" |")
    lines += ["","## 源文件快照与内容 SHA","",
        "源码证据包括原路径、实际行范围、symbol 与 SHA。快照是文档证据文本，不是新 kernel 或生产模块。若原工作树仍存在，validator 同时校验原文件与快照。","",
        "| source_id | origin | documentary snapshot | SHA256 |",
        "|---|---|---|---|"]
    for key,record in snap['sources'].items():
        lines.append(f"| {key} | {record['origin']} | [{record['snapshot']}]({record['snapshot']}) | {record['sha256']} |")
    lines += ["","## 与任务默认候选的差异","",
        "找到了 SHA 绑定的二代实现，采用 implementation-focused 模式，主图选择泛型 LSTM/sLSTM FP16/BF16 forward。单 CTA、融合、循环重用权重、FP32 内部状态与局部 feedback 是所选一代已有机制。二代实际增量聚焦独立 compute carry、输出契约、full-call context、固定地址的准备执行和离线 exact-shape 策略。",
        "", "## 仍需实现或验证","",
        "- 当前路径没有新一次 GPU launch；图中没有新的性能数字或有效 overlap 测量。",
        "- 全形状 S_t/R/b/A_t 物理映射、layout 转换成本与 spill 归因为 Unknown。旧 resource CSV 是旧 ablation cohort，不能充当当前全形状的资源证明。",
        "- input prefetch 的独立资源、issue/complete/wait、slot ready/reuse 协议与 trace 为 Unknown；候选关系为 Proposed。",
        "- capability/resource-aware 自动分派、fallback registry、cluster 轴与通信、WGMMA 操作数、FP8 为 Proposed / TBD。",
        "- TiRex/xLSTM 的特定算术、训练/backward 与分布式传输不属于此泛型图的实现范围。",
        "- global 地址访问不换算 HBM 事务；loop-carried SSA 不等于全寄存器、无 spill；CTA 不独占 SM。","",
        "## 一手参考（仅为对应机制的依据）","",
        "- [FlashRNN paper v2, Algorithm 1 / section 5 / Appendix E](https://arxiv.org/html/2412.07752v2): D001 的 CUDA 文献背景，与所选本地 Triton 基线严格分开。",
        "- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html): 后续 cluster/TMA 硬件机制；不证明 FlashRNN2 已采用。",
        "- [CairoSVG API documentation](https://cairosvg.org/documentation/): 实际 PNG/PDF 导出接口，不作为 GPU 架构证据。",""]
    (root/"evidence.md").write_text("\n".join(lines).rstrip()+"\n")
    lines=["# 实际源码到架构判断的映射","",
        "对应源代码已保存为 evidence-snapshots 下的文本；完整原路径、行范围和 SHA 在 evidence.md。没有新建 StateManager/PipelineEngine 等虚构组件。","",
        "| component / source | role / evidence | figures |",
        "|---|---|---|"]
    for key,record in snap['sources'].items():
        refs=[c for c in model['claims'] if any(e.get('source_id')==key for e in c['evidence'])]
        if not refs:continue
        figs=sorted({f for c in refs for f in c['figure_ids']})
        lines.append(f"| [{key}]({record['snapshot']}) | "+", ".join(c['id'] for c in refs)+" | "+", ".join(figs)+" |")
    lines += ["","显式实验调用者：helpers.Model.__init__ 调用 policy.select 后将 config 传给 PreparedSequence；PreparedSequence 自身没有自动 selector。Public default recurrence 的 legacy 输出行为单独保留，不能把实验 caller 画成公共 API 已接入的全局调优器。",""]
    (root/"code_map.md").write_text("\n".join(lines).rstrip()+"\n")
    counts=Counter(c['status'] for c in model['claims'])
    print(f"Exported bilingual captions, evidence and code map: {len(model['claims'])} claims / {dict(counts)}")


if __name__=="__main__":
    main()
