# E190 Adamson→Replogle 直接迁移

状态：**PASS**。692 个任务、47 个共同扰动基因、48 个目标 batch。

## 与 zero-effect 比较

| estimator | mean_rmse | zero_mean_rmse | task_win_rate_vs_zero | gene_cluster_mean_delta | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| scGPT | 0.258261 | 0.258283 | 0.570809 | 0.000427 | 0.000126 | 0.000731 |
| GEARS | 0.238505 | 0.258283 | 0.606936 | -0.006386 | -0.014773 | 0.001123 |
| six_model_family | 0.249185 | 0.258283 | 0.598266 | -0.002540 | -0.006747 | 0.001101 |
| source_effect | 0.237058 | 0.258283 | 0.624277 | -0.007340 | -0.016312 | 0.000284 |

负的 delta 表示优于 zero-effect。置信区间按目标基因整簇抽样。

## 预测前风险量与真实误差

| predictor | outcome | spearman | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- |
| diversity_lower_bound | family_rms_error | 0.423882 | 0.134779 | 0.632195 |
| diameter_half_lower_bound | family_worst_error | 0.507335 | 0.239738 | 0.695589 |
| predicted_magnitude | family_rms_error | 0.419763 | 0.128498 | 0.620064 |
| source_effect_magnitude | family_rms_error | 0.419094 | 0.160018 | 0.623813 |

family RMS 下界违例 0；worst-member 下界违例 0。PASS 只表示冻结合同和确定性下界成立，不表示跨研究预测一定优于简单基线。
