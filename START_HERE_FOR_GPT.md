# 给网页 GPT / Claude 的 SafeConf 当前入口

更新时间：2026-09-06

当前分支：`exp/task-risk-audit-20260611`

## 必须先读

1. [小白审核与从零入口](docs/学习导航/07_小白审核与从零入口_20260906.md)
2. [当前项目手把手学习](docs/学习导航/00_当前项目手把手学习_20260905.md)
3. [当前研究判断](docs/实验结果/CURRENT_RESEARCH_DECISION_20260905.md)
4. [E201 正式报告](docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md)
5. [E204 四背景工程验收](docs/实验结果/E204_risk_guided_training_20260830/PROFILE_ACCEPTANCE_20260905.md)

不要先从 `agents/`、旧 HTML、七月状态文件、GPT 长会话或编号最小的实验开始。它们用于追溯，不能覆盖以上材料。

## 当前事实

- SafeConf 位于扰动预测之后，为任务计算风险特征，不是新的表达预测器。
- E201 已完成四个整体细胞背景留出的盲测：1,808 个主任务、200 个敏感性任务、5,000 次簇 bootstrap。
- SafeConf 对 family RMS error 的合并 Spearman 为 0.4082，控制预测幅度后为 0.2503；两者置信区间均为正。
- predicted magnitude 的合并 Spearman 为 0.6189，20% 复核收益也高于 SafeConf。不要声称 SafeConf 单独优于幅度。
- E202 的主检验未通过，应保留为负结果。
- E204 四个 target 的一轮 profile 均通过，证明训练权重覆盖和实现正确；正式 80 轮比较尚未完成。
- E205 尚未运行。当前分歧只表示同架构对随机初始化的敏感性。

## 回答用户时

- 英文第一次出现时，在括号中写中文解释和项目内作用；
- 结论旁给出相对证据路径；
- 区分“已经得到的结果”“工程验收”和“下一步协议”；
- 不编 E204 正式结果，不把期刊分区写成保证；
- 用户正在正式学习，先用具体例子和流程解释，再进入公式和代码。

## 可直接复制给新模型

```text
请先读 docs/学习导航/07_小白审核与从零入口_20260906.md，
再通读 docs/学习导航/00_当前项目手把手学习_20260905.md，
并核对 CURRENT_RESEARCH_DECISION_20260905.md、E201_CORE_REPORT.md
和 E204 的 PROFILE_ACCEPTANCE_20260905.md。
请用通俗中文从零讲解，再回答学习稿第 18 节的 10 个问题；不要修改代码，
不要从 agents 原始意见、GPT 长会话或 2026-09-05 以前的状态文件推断当前进度。
```
