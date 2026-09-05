# SafeConf 当前工作入口

更新时间：2026-09-05

`workspace/` 保存近期汇报和过程材料，不作为最高事实来源。

当前先读：

1. `../docs/学习导航/00_当前项目手把手学习_20260905.md`
2. `../docs/实验结果/CURRENT_RESEARCH_DECISION_20260905.md`
3. `../docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md`
4. `../docs/实验结果/E204_risk_guided_training_20260830/README_先看这个.md`

## 现在做到哪

- E201 四个整体细胞背景留出的正式盲测已完成。
- SafeConf 与误差稳定正相关，并包含预测幅度之外的信息；预测幅度仍是更强的单一排序器。
- E202 主假设没有得到支持，负结果已经保留。
- E204 的四 target 一轮工程 profile 均通过，下一步是正式 80 轮 `risk_weighted` 与 `dispersion_only` 训练。
- E205 跨架构对照尚未运行。

旧组会稿可以追溯当时的问题和说法，不能覆盖上面的新结果。
