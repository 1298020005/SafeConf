# SafeConf｜跨上游迁移实验

**完成时间：2026-09-28｜TxPert GAT + Exphormer｜3616 prediction records｜1808 个任务｜按基因五折完全留出**

这项实验把两个 TxPert 上游的风险记录放在同一风险学习框架中。每个折按基因切分，评价基因的任何任务都不出现在风险训练标签中；两个上游虽然共享生物任务，因此这是跨上游结构迁移证据，不是新的生物数据确认。

| 上游 | Raw magnitude U20 | 跨上游训练 Ridge(P+Q) | 相对幅度 | 跨上游训练 Ridge(P+Q+H) |
|---|---:|---:|---:|---:|
| TxPert GAT | 0.7682 | **0.7750** | +0.0068，4/4 背景正向 | 0.7857 |
| TxPert Exphormer | 0.7587 | **0.7764** | +0.0177，4/4 背景正向 | 0.7802 |

P+Q 的提升来自完全未用于风险训练的基因组，说明它不是只记住某一个上游的固定误差模式。P+Q+H 在两个上游也提高点估计，但均有一个背景轻微下降，因此 H 继续作为可选扩展。

这项结果将 SafeConf 的主张从“某个 TxPert 模型后面加特征有效”推进到：**同一组预测/公共历史字段和同一个小型风险学习器可以迁移到结构不同的上游预测器。** 后续仍需真正不同数据集的独立确认，化学 CPA 结果目前承担压力测试而非确认。

文件：`SUMMARY.csv`、`TARGET_RESULTS.csv`、`INCREMENTAL_RESULTS.csv`、`OOF_PREDICTIONS.csv.gz`、`SPLIT_MANIFEST.csv.gz`、`RUN_STATUS.json`。
