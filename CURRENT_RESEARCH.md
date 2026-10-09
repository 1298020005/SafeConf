# SafeConf 当前入口

更新：2026-10-09。研究分支为 `exp/e220-reviewer-closure-20260921`。

## 负责人执行口径

SafeConf 是项目名；TCBB、二区和 CCF-B 是投稿目标，不是论文题目。当前工作题目暂定为：

> Public Perturbation Evidence Enables Target-Error-Free Risk Auditing for Single-Cell Perturbation Prediction

本阶段由 Codex 作为研究负责人连续推进：KOLF 完成后自动进入证据冻结、主图、基线表、Methods 合同、Supplement 索引和复现入口整理。普通工程修复、阶段衔接、结果核对和组件取舍不交回用户；正文和 PDF 仍按用户要求暂缓。

GitHub默认展示的`master`仍是8月20日的轻量快照。查看当前研究请进入[研究分支](https://github.com/1298020005/SafeConf/tree/exp/e220-reviewer-closure-20260921)。

## 现在看这里

- [v2.1实际执行与新结果](docs/研究推进/20261009_投稿证据_v21/PROGRESS_REPORT.md)：标签效率、20次内容归因、Source固定修复及外部真实作业状态。
- [KOLF负责人执行决定](docs/研究推进/20261009_投稿证据_v21/KOLF_PRIMARY_EXECUTION_DECISION.md)：训练、能力门、风险冻结、确认读取和最终证据交付顺序。

## 既有汇报与历史安排

1. [本周周报](docs/周报_20261008.md)：四条简短说明。
2. [下一阶段实验工作单](docs/研究推进/20261008_下一阶段/EXECUTION_PLAN.md)：已启动的统计、后续实验、采用条件及资源。
3. [完整研究说明与汇报](docs/组会汇报/20261008_最终报告/SafeConf_研究主线与汇报.md)：演变、架构、结果和图。

## 当前研究决定

- McFaline当前默认是PublicRule(合法历史实验的加权距离)。同样复核43项，命中22项大误差；幅度命中4项。
- Source(旧模型错误监督)和Target(本模型反馈)保留各自已验证范围，候选结果与默认配置分别记录。
- 10月8日新完成542任务、377个基因的重复端点配对统计：公共规则相对幅度的8项区间为正；相对历史支持量的8项区间均跨零。下一项是更严格的内容归因。
- 修正下一阶段资产安排：Norman采用CRISPRa(基因激活)，GWPS采用CRISPRi(基因抑制)。原周报中的Norman方案按本工作单修订；优先核准Adamson与GWPS的同干预类型比较。
- McFaline主实验、Source和已有风险学习结果已冻结，不再扩展网络或继续搜索特征。当前唯一新增主线是 KOLF 独立确认：600 train、300 feedback、300 confirmation、1400固定输出轴；confirmation truth 在风险分数冻结前保持封存。
- KOLF 结束后停止新实验，直接生成最终结果矩阵、coverage 分层、fallback sanity check、claim-evidence matrix、四张主图 source data、Methods 合同和 reproduction entry。

最新结果见[本轮结果目录](docs/研究推进/20261008_下一阶段/)。原始实验、失败结果和旧稿原地保留。

## 仓库整理状态

本轮核查时有34项已跟踪修改、105个未跟踪条目(含目录)，索引无暂存内容、无合并冲突。多数属于旧论文构建、SAMS状态和Orion原型；它们仍需逐项核对所属工作后归档或提交。

当前入口只指向最新说明；旧日期入口按其原实验范围阅读。下载包的最新周报版本是`SafeConf_本周周报_20261008_口语版.zip`，下一阶段实验安排以本页链接的工作单为准。
