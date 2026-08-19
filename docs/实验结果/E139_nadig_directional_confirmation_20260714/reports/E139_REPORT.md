# E139｜Nadig 第七数据方向风险确认

## 预注册结论：通过

冻结方向风险在 centered Pearson error 上的两 fold 宏平均 Spearman 为 **0.748**，centered cosine error 为 **0.757**。复合方向 rank 的 perturbation-cluster bootstrap 95% CI 为 **[0.693, 0.801]**。

## 分数对照

| score | Pearson ρ | cosine ρ | combined rank ρ |
|---|---:|---:|---:|
| baseline_predicted_magnitude | 0.280 | 0.299 | 0.291 |
| directional_risk_frozen | 0.748 | 0.757 | 0.753 |
| risk_model_disagreement | 0.122 | 0.124 | 0.126 |
| safeconf_calibrated_pair_risk | -0.363 | -0.359 | -0.359 |

## 上游模型与简单质心

| dataset | ensemble RMSE | training perturbed-centroid RMSE | ensemble − simple | ensemble 胜出比例 |
|---|---:|---:|---:|---:|
| Nadig_two_cellline | 0.1315 | 0.4007 | -0.2693 | 79.7% |

## 信息边界

- E135 模型哈希在 E136 合同中冻结；E139 第一阶段只读取四个部署特征并写出风险分数，第二阶段才读取向量计算方向误差。
- 两个主要终点、复合终点、竞争分数和全部 3,000 次 bootstrap 均落盘。
- 若未通过，冻结模型不在 Nadig 上调参后重新宣称确认。
