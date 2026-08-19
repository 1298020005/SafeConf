# E110｜困难设置内层校准审计

校准器只用 E109 的内层新背景、新扰动和双未见样本拟合。固定 `Ridge(alpha=1, positive=True)`，输入为分歧、预测幅度、背景新颖度、扰动新颖度及结构交互项。外层 837 个测试标签没有参与拟合或阈值。

## 3-fold 宏平均 Spearman

| setting | score | ρ |
|---|---|---:|
| all_test_settings_pooled | baseline_predicted_magnitude | 0.148 |
| all_test_settings_pooled | risk_model_disagreement | 0.137 |
| all_test_settings_pooled | safeconf_calibrated_pair_risk | 0.253 |
| all_test_settings_pooled | safeconf_frozen_pair_risk | 0.242 |
| all_test_settings_pooled | safeconf_nested_hard_risk | 0.176 |
| context_and_perturbation_unseen | baseline_predicted_magnitude | 0.054 |
| context_and_perturbation_unseen | risk_model_disagreement | 0.068 |
| context_and_perturbation_unseen | safeconf_calibrated_pair_risk | 0.070 |
| context_and_perturbation_unseen | safeconf_frozen_pair_risk | 0.068 |
| context_and_perturbation_unseen | safeconf_nested_hard_risk | 0.055 |
| context_unseen_row | baseline_predicted_magnitude | 0.218 |
| context_unseen_row | risk_model_disagreement | 0.210 |
| context_unseen_row | safeconf_calibrated_pair_risk | 0.155 |
| context_unseen_row | safeconf_frozen_pair_risk | 0.181 |
| context_unseen_row | safeconf_nested_hard_risk | 0.208 |
| perturbation_unseen_column | baseline_predicted_magnitude | 0.099 |
| perturbation_unseen_column | risk_model_disagreement | 0.113 |
| perturbation_unseen_column | safeconf_calibrated_pair_risk | 0.129 |
| perturbation_unseen_column | safeconf_frozen_pair_risk | 0.113 |
| perturbation_unseen_column | safeconf_nested_hard_risk | 0.105 |
| random_missing_pair | baseline_predicted_magnitude | 0.053 |
| random_missing_pair | risk_model_disagreement | 0.051 |
| random_missing_pair | safeconf_calibrated_pair_risk | 0.138 |
| random_missing_pair | safeconf_frozen_pair_risk | 0.091 |
| random_missing_pair | safeconf_nested_hard_risk | 0.060 |

## 聚类 bootstrap

| comparator | Δρ | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| safeconf_calibrated_pair_risk | -0.077 | [-0.226, 0.045] | 0.152 |
| safeconf_frozen_pair_risk | -0.066 | [-0.205, 0.053] | 0.187 |
| risk_model_disagreement | 0.038 | [-0.046, 0.130] | 0.800 |
| baseline_predicted_magnitude | 0.028 | [-0.075, 0.146] | 0.655 |

## 结论

困难设置校准没有替代 E108 主分数：pooled ρ 为 0.176，低于随机-pair 校准的 0.253 和冻结规则的 0.242。它在新背景上达到 0.208，接近 magnitude 的 0.218，但新扰动、双未见和随机缺失没有同步改善。当前只有三个背景，每个外层 fold 的内层训练只能使用一个背景，权重迁移不稳定。E110 作为失败边界保留，不用于回调 E108 权重。
