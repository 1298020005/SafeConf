# SafeConf 当前入口

更新：2026-10-09。研究分支为 `exp/e220-reviewer-closure-20260921`。

## 负责人执行口径

SafeConf 是项目名；TCBB、二区和 CCF-B 是投稿目标，不是论文题目。当前工作题目暂定为：

> Public Perturbation Evidence Enables Target-Error-Free Risk Auditing for Single-Cell Perturbation Prediction

本阶段由 Codex 作为研究负责人连续推进：KOLF 已完成，证据冻结、主图、基线表、Methods 合同、Supplement 索引和复现入口已经生成。普通工程修复、阶段衔接、结果核对和组件取舍不交回用户；正文和 PDF 仍按用户要求暂缓。

GitHub默认展示的`master`仍是8月20日的轻量快照。查看当前研究请进入[研究分支](https://github.com/1298020005/SafeConf/tree/exp/e220-reviewer-closure-20260921)。

## 现在看这里

- [v2.1实际执行与新结果](docs/研究推进/20261009_投稿证据_v21/PROGRESS_REPORT.md)：标签效率、20次内容归因、Source固定修复及外部真实作业状态。
- [KOLF负责人执行决定](docs/研究推进/20261009_投稿证据_v21/KOLF_PRIMARY_EXECUTION_DECISION.md)：训练、能力门、风险冻结、确认读取和最终证据交付顺序。
- [证据冻结包](docs/研究推进/20261009_投稿证据_v21/evidence_freeze_v1/MORNING_REPORT.md)：KOLF独立确认、Frangieh跨模型压力证据、统一表格和图源数据。

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
- KOLF 已完成：Ridge 通过能力门，MLP 方差门失败；确认集300个基因在风险冻结后读取，Public覆盖228个、72个无历史任务回退幅度。正式独立结果是单 predictor 范围。
- Frangieh native512 的 GEARS↔scGPT 压力测试保留为补充证据：PublicHGB相对幅度在两个方向的全量 paired U20 增量为 +0.460 [0.317, 0.704] 与 +0.309 [0.079, 0.440]；六个上游能力门均失败，因此不作为合格独立确认。
- 当前唯一科学缺口是预测器特异信息能否稳定超过历史效应能量/支持量；不再通过堆网络掩盖这项边界。目标仍是形成可投二区/CCF-B的完整稿件，当前交付是证据冻结而非投稿完成。

最新结果见[本轮结果目录](docs/研究推进/20261008_下一阶段/)。原始实验、失败结果和旧稿原地保留。

## 仓库整理状态

本轮核查时有34项已跟踪修改、105个未跟踪条目(含目录)，索引无暂存内容、无合并冲突。多数属于旧论文构建、SAMS状态和Orion原型；它们仍需逐项核对所属工作后归档或提交。

当前入口只指向最新说明；旧日期入口按其原实验范围阅读。下载包的最新周报版本是`SafeConf_本周周报_20261008_口语版.zip`，下一阶段实验安排以本页链接的工作单为准。
