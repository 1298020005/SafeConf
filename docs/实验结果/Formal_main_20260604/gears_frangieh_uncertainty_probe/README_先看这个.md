# GEARS Frangieh uncertainty probe 结果说明

更新时间：2026-06-05

## 一句话

这次确认了：GEARS（图神经网络扰动预测模型）在 Frangieh（黑色素瘤扰动数据集）上可以导出 native uncertainty（原生不确定性）。

但这只是 1 个 seed、21 条 test record 的 probe（探针），不能当正式结论。

## 这次跑了什么

- 数据集：Frangieh。
- 模型：GEARS。
- 参数：`--uncertainty` 打开。
- seed：1。
- epoch：1。
- 输出：PredictionRecord + GEARS native uncertainty。

## 关键结果

| 项目 | 结果 |
|---|---:|
| status | ok |
| PredictionRecord | 21 |
| native uncertainty 非空记录 | 21 |
| test MSE | 0.00150 |
| test Pearson | 0.99565 |
| top20 DE MSE | 0.00491 |
| top20 DE Pearson | 0.93528 |

## uncertainty 快速评估

| score | n | Spearman rho | p value | 判断 |
|---|---:|---:|---:|---|
| `gears_uncertainty_logvar_mean` | 21 | 0.352 | 0.118 | 方向对，但样本太少 |
| `gears_uncertainty_confidence` | 21 | -0.352 | 0.118 | 方向对，但样本太少 |

解释：

- logvar（对数方差）越大，理论上 risk（风险）越高，所以 rho 为正是合理方向。
- confidence（可信度）越大，理论上 error（误差）越小，所以 rho 为负是合理方向。
- 但 n=21 太小，不能写成正式结论。

## 现在能说什么

能说：

> GEARS native uncertainty 可以被导出，下一步可以做公平比较。

不能说：

> GEARS uncertainty 已经验证有效。

## 下一步

建议下一步不是继续单 seed，而是做 GEARS formal probe：

1. 增加 test records。
2. 明确 GEARS split（切分）和 SafeConf held-out pair split 是否兼容。
3. 同时比较：
   - GEARS native uncertainty；
   - SafeConf protocol score；
   - model disagreement；
   - seed ensemble proxy（如果有多 seed）。

