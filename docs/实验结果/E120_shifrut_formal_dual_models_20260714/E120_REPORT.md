# E120｜Shifrut–Marson 四背景正式双模型复制

E120 使用 E119 事先冻结的两供体 × TCR 刺激/未刺激矩阵。scGPT 加载 whole-human 预训练参数并按外层 fold 微调；GEARS 的共表达图只使用训练背景 control。目标受扰动表达在预测和风险分数落盘后才用于评价。

- 外层 folds：4
- 测试任务：172
- strict PredictionRecord：344；issues=0

| setting | score | macro Spearman |
|---|---|---:|
| context_and_perturbation_unseen | baseline_predicted_magnitude | -0.200 |
| context_and_perturbation_unseen | risk_model_disagreement | 0.025 |
| context_and_perturbation_unseen | safeconf_calibrated_pair_risk | -0.375 |
| context_and_perturbation_unseen | safeconf_frozen_pair_risk | 0.025 |
| context_unseen_row | baseline_predicted_magnitude | 0.176 |
| context_unseen_row | risk_model_disagreement | 0.046 |
| context_unseen_row | safeconf_calibrated_pair_risk | 0.381 |
| context_unseen_row | safeconf_frozen_pair_risk | 0.038 |
| perturbation_unseen_column | baseline_predicted_magnitude | 0.036 |
| perturbation_unseen_column | risk_model_disagreement | -0.109 |
| perturbation_unseen_column | safeconf_calibrated_pair_risk | -0.114 |
| perturbation_unseen_column | safeconf_frozen_pair_risk | -0.109 |
| random_missing_pair | baseline_predicted_magnitude | 0.226 |
| random_missing_pair | risk_model_disagreement | 0.268 |
| random_missing_pair | safeconf_calibrated_pair_risk | 0.107 |
| random_missing_pair | safeconf_frozen_pair_risk | 0.220 |
| all_test_settings_pooled | safeconf_calibrated_pair_risk | 0.173 |
| all_test_settings_pooled | safeconf_frozen_pair_risk | 0.053 |
| all_test_settings_pooled | risk_model_disagreement | 0.051 |
| all_test_settings_pooled | baseline_predicted_magnitude | 0.209 |

该数据在 E115/E117 完成后才解封，适合作为新增外部效用验证和误差界修正的独立测试来源。
