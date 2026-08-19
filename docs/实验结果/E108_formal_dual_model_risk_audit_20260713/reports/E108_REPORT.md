# E108｜正式 scGPT–GEARS 双模型风险审计

E108 使用 E106/E107 的同背景输入正式预测。外层测试共有 3×279=837 个任务、1674 条严格 PredictionRecord，合同问题数为 0。预测、风险分数和 q80 阈值冻结后才读取测试扰动表达。

校准只读取每折 30 个 source validation pair。新背景与新扰动的结构项沿用冻结规则，因此这部分仍需后续 inner row/column/double 校准实验验证，不能把当前校准结果写成跨设置保证。

## 3-fold 宏平均 Spearman

| setting | score | ρ |
|---|---|---:|
| all_test_settings_pooled | baseline_predicted_magnitude | 0.148 |
| all_test_settings_pooled | risk_model_disagreement | 0.137 |
| all_test_settings_pooled | safeconf_calibrated_pair_risk | 0.253 |
| all_test_settings_pooled | safeconf_frozen_pair_risk | 0.242 |
| context_and_perturbation_unseen | baseline_predicted_magnitude | 0.054 |
| context_and_perturbation_unseen | risk_model_disagreement | 0.068 |
| context_and_perturbation_unseen | safeconf_calibrated_pair_risk | 0.070 |
| context_and_perturbation_unseen | safeconf_frozen_pair_risk | 0.068 |
| context_unseen_row | baseline_predicted_magnitude | 0.218 |
| context_unseen_row | risk_model_disagreement | 0.210 |
| context_unseen_row | safeconf_calibrated_pair_risk | 0.155 |
| context_unseen_row | safeconf_frozen_pair_risk | 0.181 |
| perturbation_unseen_column | baseline_predicted_magnitude | 0.099 |
| perturbation_unseen_column | risk_model_disagreement | 0.113 |
| perturbation_unseen_column | safeconf_calibrated_pair_risk | 0.129 |
| perturbation_unseen_column | safeconf_frozen_pair_risk | 0.113 |
| random_missing_pair | baseline_predicted_magnitude | 0.053 |
| random_missing_pair | risk_model_disagreement | 0.051 |
| random_missing_pair | safeconf_calibrated_pair_risk | 0.138 |
| random_missing_pair | safeconf_frozen_pair_risk | 0.091 |

## 聚类 bootstrap：校准风险相对基线

| comparator | Δρ | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| safeconf_frozen_pair_risk | 0.011 | [-0.028, 0.054] | 0.702 |
| risk_model_disagreement | 0.116 | [-0.033, 0.277] | 0.931 |
| baseline_predicted_magnitude | 0.105 | [-0.042, 0.248] | 0.916 |

完整任务表见 `tables/E108_TEST_TASK_RISK_TABLE.csv`，校准参数见 `tables/E108_CALIBRATORS.csv`，白底图见 `figures/F1_formal_risk_by_setting.svg`。
