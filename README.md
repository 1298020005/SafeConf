# SafeConf

## 当前研究入口（2026-10-08）

请先读[当前研究入口](CURRENT_RESEARCH.md)。本周周报、最新结果和下一阶段实验均从该页进入。

研究分支为`exp/e220-reviewer-closure-20260921`；GitHub默认展示的`master`是8月20日轻量快照。直接查看[当前研究分支](https://github.com/1298020005/SafeConf/tree/exp/e220-reviewer-closure-20260921)。

## 2026-10-02研究记录（历史快照）

当前执行分支为 `exp/e220-reviewer-closure-20260921`。本轮研究分开检验公共真实实验、源预测器错误监督和目标预测器反馈的增量。下方 2026-09-06 的进度与分支说明属于历史快照。

- [当前实验状态与运行记录](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/EXECUTION_STATUS.json)
- [实验完成要求与实际证据](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/REQUIREMENT_COMPLETION_AUDIT.csv)
- [论文主张与证据](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/CLAIM_EVIDENCE_MATRIX.md)
- [复现入口与已知限制](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/REPRODUCIBILITY_REPORT.md)
- [Deep Research 报告复核](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/DEEP_RESEARCH_REVIEW.md)

当前事实：独立 Orion 研究的已发表线性上游通过预登记 validation 非劣能力门；不支持其优于简单基线。一次性 TEST 读取与固定 13 方法、5000 次簇抽样已完成，共同历史比较为 232 个任务、144 个基因簇，占全部 2993 个合格 TEST 任务约 7.75%。主比较的源风险 HGB 未建立超过同参照历史距离的增量，且一背景未通过安全护栏。详见[当前方法判断](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/METHOD_DECISION.md)。TEST 元数据已见，因此这是前瞻性的真值盲评外部复核，不是完整 pristine confirmation。

现有 McFaline 共同基因空间结果尚未证明源风险学习超过强历史距离和历史数量规则；不能将已运行的实验或工程检查写成投稿质量已经达标。论文正文依用户要求暂不生成。

### 历史入口（事实截止 2026-09-06）

SafeConf 研究单细胞扰动预测完成以后，怎样在不知道真实答案时识别高风险任务，以及能否把风险信息用于改进模型训练。

当前分支：`exp/task-risk-audit-20260611`

事实截止：2026-09-06

## 第一次进入先读这两份

1. [2026-09-06 审核与图解教学](docs/学习导航/20260906_论文审核与从零教学/README.md)：GPT 对照、Nature 风格中文图、一区/二区就绪报告。
2. [当前项目手把手学习](docs/学习导航/00_当前项目手把手学习_20260905.md)：完整教程，从单细胞、扰动、细胞系讲到公式、E199–E205、代码和自测题。

文中所有仓库文件都使用相对链接，下载到其他电脑后仍可跳转。

## 当前进度

| 工作 | 状态 | 当前结论 |
| --- | --- | --- |
| E199：K562 未见基因 | 完成 | 三个公开 TxPert 模型（GAT / Exphormer / Exphormer-MG）的分歧能识别一部分难任务 |
| E200：整个 K562 背景留出 | 完成 | 预测幅度明显强于固定风险分 |
| E201：四背景 × 四种子盲测 | 完成 | SafeConf 与误差稳定正相关，也有幅度之外的信息；但单独排序弱于幅度 |
| E202：相对基线失败归因 | 完成，主门失败 | 当前 source dispersion 不能解释 GAT 相对简单基线多犯的错 |
| E204：风险指导训练 | 四个背景的工程验收通过 | 权重确实进入训练且无覆盖缺口；正式 80 轮对照尚未运行完 |
| E205：跨架构分歧 | 协议已设计 | 尚未运行，当前四个成员只是同架构四个随机种子 |

详细数字以 [当前研究判断](docs/实验结果/CURRENT_RESEARCH_DECISION_20260905.md)、[E201 正式报告](docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md) 和 [E204 工程验收](docs/实验结果/E204_risk_guided_training_20260830/PROFILE_ACCEPTANCE_20260905.md) 为准。

## 目录

| 路径 | 内容 |
| --- | --- |
| `code/` | SafeConf 正式代码和测试 |
| `tools/` | E 编号实验运行脚本、审计和维护工具 |
| `docs/实验结果/` | 冻结协议、结果表、报告和负结果 |
| `docs/学习导航/` | 当前教程和历史学习材料 |
| `workspace/` | 近期汇报与工作材料，不高于正式实验报告 |
| `agents/` | Grok、GLM、Qoder 等原始意见，不是事实来源 |
| `/home/yyf/archive/safeconf/` | 服务器历史归档，不属于 Git 仓库 |

大型 H5AD、模型权重和运行缓存不提交 Git；仓库主要保存代码、协议、精简结果和可核验记录。

## 远程同步

```text
GitHub: https://github.com/1298020005/SafeConf
Gitee:  https://gitee.com/librety/safe-conf
分支:   exp/task-risk-audit-20260611
```

完整拉取和新 Codex 初始化见 [REMOTE_CODEX_INIT.md](REMOTE_CODEX_INIT.md)。
