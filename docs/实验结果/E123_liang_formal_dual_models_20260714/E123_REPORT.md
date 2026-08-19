# E123｜Liang–Wang 九背景正式双模型复制

该数据合同在 Shifrut 结果解封以后冻结，但测试真值在本轮预测、风险落盘前未读取。scGPT 和 GEARS 沿用 E112/E120 的固定训练与输入合同。

- folds：9
- test tasks：612
- strict records：1224；issues=0

| setting | score | macro Spearman |
|---|---|---:|
| context_and_perturbation_unseen | baseline_predicted_magnitude | -0.111 |
| context_and_perturbation_unseen | risk_model_disagreement | -0.111 |
| context_and_perturbation_unseen | safeconf_calibrated_pair_risk | -0.225 |
| context_and_perturbation_unseen | safeconf_frozen_pair_risk | -0.111 |
| context_unseen_row | baseline_predicted_magnitude | 0.478 |
| context_unseen_row | risk_model_disagreement | 0.465 |
| context_unseen_row | safeconf_calibrated_pair_risk | 0.448 |
| context_unseen_row | safeconf_frozen_pair_risk | 0.459 |
| perturbation_unseen_column | baseline_predicted_magnitude | -0.011 |
| perturbation_unseen_column | risk_model_disagreement | 0.043 |
| perturbation_unseen_column | safeconf_calibrated_pair_risk | -0.039 |
| perturbation_unseen_column | safeconf_frozen_pair_risk | 0.043 |
| random_missing_pair | baseline_predicted_magnitude | 0.317 |
| random_missing_pair | risk_model_disagreement | 0.298 |
| random_missing_pair | safeconf_calibrated_pair_risk | 0.230 |
| random_missing_pair | safeconf_frozen_pair_risk | 0.301 |
| all_test_settings_pooled | safeconf_calibrated_pair_risk | 0.212 |
| all_test_settings_pooled | safeconf_frozen_pair_risk | 0.202 |
| all_test_settings_pooled | risk_model_disagreement | 0.075 |
| all_test_settings_pooled | baseline_predicted_magnitude | 0.074 |
