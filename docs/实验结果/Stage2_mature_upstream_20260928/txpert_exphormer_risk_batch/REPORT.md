# SafeConf｜TxPert Exphormer 跨结构复现

**完成时间：2026-09-28｜同一 1808 个主任务｜同一四背景｜同一风险协议**

Exphormer 是与 STRING-GAT 结构不同的 TxPert 上游，使用已封存的四种子预测。风险模型和输入没有为 Exphormer 调整。

| 方法 | 宏 U20 | 宏 Spearman |
|---|---:|---:|
| 原始预测幅度 M | 0.7587 | 0.7276 |
| Ridge，P+Q | 0.7732 | 0.7485 |
| Ridge，P+Q+H | 0.7684 | 0.7643 |
| HGB，P+Q+H | 0.7583 | 0.7487 |
| 小 MLP，P+Q+H | **0.7742** | 0.7556 |

Exphormer 上的递增结果：

- 当前预测 P 相对学习幅度基线的 U20 增量 `+0.01402`，四个背景全部正向；
- P→P+Q 的 Ridge U20 增量 `+0.00096`，方向不一致；
- P+Q→P+Q+H 的 Ridge U20 增量 `−0.00483`，四个背景均为负；
- P+Q→P+Q+H 的 Spearman 增量 `+0.01585`，95% 区间 `[+0.00468,+0.02705]`，说明 H 能改善排序的连续性，但不保证固定 20% 截断收益。

这条复现改变了方法选择：**跨 GAT 和 Exphormer 都保留的主输入是 P+Q；H 作为可选内容扩展和解释性分析，不作为第一版统一模型的必选项。** GAT 上 H 有正向 U20 增量，Exphormer 上 H 的截断效用下降，不能把 H 写成跨结构必然有效。

本批仍是同四背景公开标签上的回顾性跨结构复现，不是新数据独立确认；下一步化学线承担模态迁移压力测试。

文件：`SUMMARY.csv`、`TARGET_RESULTS.csv`、`INCREMENTAL_RESULTS.csv`、`OOF_PREDICTIONS.csv.gz`、`RUN_STATUS.json`。执行合同为 `../TXPERT_EXPHORMER_RISK_BATCH_CONTRACT.md`。
