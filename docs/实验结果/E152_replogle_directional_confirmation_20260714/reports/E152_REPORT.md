# E152｜Replogle K562/RPE1 方向风险确认

## 预注册 gate：通过

冻结方向风险在 centered Pearson error 上的两折宏平均 Spearman 为 **0.194**，centered cosine error 为 **0.187**。复合方向 rank 的 perturbation-cluster bootstrap 95% CI 为 **[0.071, 0.307]**。

相对 predicted magnitude 的复合方向 Δρ bootstrap 95% CI 为 **[-0.162, 0.139]**。

| score | Pearson ρ | cosine ρ | combined rank ρ |
|---|---:|---:|---:|
| baseline_predicted_magnitude | 0.204 | 0.211 | 0.208 |
| directional_risk_frozen | 0.194 | 0.187 | 0.191 |
| risk_model_disagreement | -0.030 | -0.038 | -0.034 |
| safeconf_calibrated_pair_risk | 0.160 | 0.170 | 0.164 |

## 简单预测器

| tasks | ensemble RMSE | training perturbed-centroid RMSE | ensemble − simple | ensemble胜出比例 |
|---:|---:|---:|---:|---:|
| 256 | 0.1031 | 0.5808 | -0.4776 | 100.0% |

## 覆盖与边界

- 主分析包含 256 个唯一 held-out cell-line × perturbation 任务；84 个 source-context 诊断任务不进入 gate。
- E135 模型没有在 Replogle 上重拟合；风险分数先冻结，随后才读取方向真值。
- 两个细胞系来自同一研究，目标细胞系 control 可见。通过 gate 也只表示同研究内跨细胞系复制，不等于跨研究泛化或新湿实验确认。
