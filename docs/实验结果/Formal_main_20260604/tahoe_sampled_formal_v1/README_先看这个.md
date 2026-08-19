# Tahoe sampled formal v1 结果说明

这一步做的是 Tahoe-100M（超大药物扰动数据）的 sampled formal validation（抽样正式验证）。它不是 7 主表的一部分，定位是 supplement / external evidence（补充外部证据）。

## 这次跑了什么

- 数据：Tahoe pseudobulk（聚合后的药物效应表，不是 337GB raw expression 原始表达矩阵）。
- 扫描：100 个 shard（数据分片），跳过 1 个损坏 parquet 分片。
- 任务：3000 个 `(cell_line, drug+dose)` pair（细胞系 × 药物+浓度组合）。
- 测试记录：4132 条 test PredictionRecord（预测记录），两个 predictor（预测器）各 2066 条。
- Predictor：
  - `V0ExactDoseMean`：同 drug（药物）+同 dose（浓度）在其他 cell_line（细胞系）里的平均效应。
  - `V0DrugMeanAcrossDose`：同 drug 跨所有 dose 在其他 cell_line 里的平均效应。

## 关键结果

| 指标 | 结果 | 怎么理解 |
|---|---:|---|
| aligned rho | 0.333 | 分数和真实误差方向一致，有正信号 |
| partial rho | 0.293 | 控制 effect magnitude（效应幅度）后仍有信号 |
| RC@80 | 约 5.0% | 保留高 confidence（可信度）预测后，平均 RMSE（误差）下降约 5% |
| bootstrap CI | 两个 predictor 的 95% CI 都不跨 0 | 统计上不是随机抖动 |
| pair leakage | 0 | test pair 没有进入 train，基本防泄漏通过 |

## 限制

- `true_error_rmse` 的 CV（变异系数）是 0.228，低于 0.3；说明误差差异没有 smoke v2 那么宽，结论要保守。
- concentration leakage（同药物不同浓度在 train 出现）比例是 0.856；这是 Tahoe 设计导致的，不等于 pair leakage。
- plate（实验板）和 cell_line（细胞系）高度绑定，不能声称完全排除了 plate batch effect（板效应）。
- held-out drug split（整种药物留出）对这两个 V0-family predictor 不适用，因为它们本来就需要同 drug 支持。

## 当前判断

Tahoe sampled formal v1 可以作为补充外部证据：说明 SafeConf 分数在一个独立超大药物图谱上也有可见信号。不要把它写成主表强结论，也不要下载 337GB raw expression。

