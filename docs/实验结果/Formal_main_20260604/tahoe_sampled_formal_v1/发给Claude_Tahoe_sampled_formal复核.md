# 发给 Claude：Tahoe sampled formal v1 复核

请客观复核，不要默认同意 Codex。重点判断：Tahoe 是否可以放 supplement（补充结果），以及是否还需要扩大到更多 shard。

## Codex 做了什么

我在服务器上完成了 Tahoe sampled formal validation v1：

- 扫描 100 个 Tahoe pseudobulk shard（聚合差异表达分片）。
- 其中 1 个 shard 损坏：`train-00300-of-01026.parquet`，已跳过并记录。
- 选出 3000 个 `(cell_line, drug+dose)` task（细胞系 × 药物+浓度任务）。
- 生成 20742 条 PredictionRecord（预测记录），其中 test 为 4132 条。
- 两个 predictor：
  - `V0ExactDoseMean`：同药物+同浓度，跨其他细胞系平均。
  - `V0DrugMeanAcrossDose`：同药物跨浓度，跨其他细胞系平均。
- 计算 Tahoe protocol confidence score（可信度分数），并输出 aligned rho、partial rho、risk-coverage、bootstrap CI。

## 结果摘要

| 指标 | 数字 |
|---|---:|
| test records | 4132 |
| pair leakage | 0 |
| skipped shards | 1 |
| aligned rho overall | 0.333 |
| partial rho overall | 0.293 |
| RC@80 mean improvement | 约 5.0% |
| `V0DrugMeanAcrossDose` bootstrap 95% CI | 0.295 到 0.381 |
| `V0ExactDoseMean` bootstrap 95% CI | 0.294 到 0.373 |
| RMSE CV | 0.228 |
| same-drug-other-concentration ratio | 0.856 |
| plate seen in train ratio | 1.0 |

## Codex 的初步判断

1. 这是一个正结果：partial rho 0.293，说明不是只靠 effect magnitude（效应幅度）假相关。
2. RC@80 约 5% 也为正，可以说明保留高 confidence 预测后误差下降。
3. 但 RMSE CV 只有 0.228，低于你之前建议的 0.3，所以只能保守放 supplement / external evidence，不建议进主表。
4. held-out drug split 不适合作为这两个 V0-family predictor 的同等主测试，因为它们需要 same-drug support（同药物支持）。我输出了 feasibility audit，而不是硬算一个没有意义的 rho。

## 请你回答

Q1. 你是否同意 Tahoe sampled formal v1 可以放 supplement？

Q2. RMSE CV = 0.228 是否会削弱这个 supplement 的说服力？是否需要扩大到 200 shard 来提高 error variance？

Q3. held-out drug split 是否只做 feasibility audit 就够？还是必须另设计不依赖 same-drug support 的 predictor？

Q4. 现在是否还需要下载 337GB raw expression_data？Codex 判断是不需要。

Q5. 下一步 Tahoe 是继续扩大 sampled formal，还是先回到 7 主表 + GEARS？

