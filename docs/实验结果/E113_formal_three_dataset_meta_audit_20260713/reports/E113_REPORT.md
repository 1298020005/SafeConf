# E113｜三套正式 scGPT–GEARS 数据集元分析

三套数据均使用同背景输入、端到端微调 scGPT 和训练背景专属共表达图 GEARS。E113 不重拟合分数，只在每个 fold 内计算相关，再对 fold 和数据集等权平均。

## 双模型平均误差

| dataset | SafeConf calibrated | frozen | disagreement | magnitude |
|---|---:|---:|---:|---:|
| Frangieh | 0.253 | 0.242 | 0.137 | 0.148 |
| Lara_exvivo | 0.387 | 0.355 | 0.176 | 0.148 |
| Santinha | 0.065 | -0.095 | -0.127 | -0.089 |

## Bootstrap

| scope | target | comparator | unit | Δρ | 95% CI | P(Δ>0) |
|---|---|---|---|---:|---:|---:|
| Frangieh | error_two_predictor_mean_rmse | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.011 | [-0.028, 0.056] | 0.699 |
| Lara_exvivo | error_two_predictor_mean_rmse | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.032 | [-0.018, 0.094] | 0.833 |
| Santinha | error_two_predictor_mean_rmse | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.160 | [-0.076, 0.429] | 0.904 |
| three_dataset_macro | error_two_predictor_mean_rmse | safeconf_frozen_pair_risk | fixed_three_datasets | 0.068 | [-0.017, 0.157] | 0.934 |
| three_dataset_macro | error_two_predictor_mean_rmse | safeconf_frozen_pair_risk | dataset_population_plus_fold_plus_perturbation | 0.068 | [-0.008, 0.203] | 0.940 |
| Frangieh | error_two_predictor_mean_rmse | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.116 | [-0.035, 0.283] | 0.924 |
| Lara_exvivo | error_two_predictor_mean_rmse | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.211 | [0.063, 0.355] | 0.998 |
| Santinha | error_two_predictor_mean_rmse | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.191 | [-0.102, 0.479] | 0.902 |
| three_dataset_macro | error_two_predictor_mean_rmse | risk_model_disagreement | fixed_three_datasets | 0.173 | [0.045, 0.294] | 0.998 |
| three_dataset_macro | error_two_predictor_mean_rmse | risk_model_disagreement | dataset_population_plus_fold_plus_perturbation | 0.173 | [0.051, 0.305] | 0.995 |
| Frangieh | error_two_predictor_mean_rmse | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.105 | [-0.045, 0.255] | 0.912 |
| Lara_exvivo | error_two_predictor_mean_rmse | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.240 | [0.083, 0.389] | 1.000 |
| Santinha | error_two_predictor_mean_rmse | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.154 | [-0.159, 0.442] | 0.816 |
| three_dataset_macro | error_two_predictor_mean_rmse | baseline_predicted_magnitude | fixed_three_datasets | 0.166 | [0.030, 0.286] | 0.993 |
| three_dataset_macro | error_two_predictor_mean_rmse | baseline_predicted_magnitude | dataset_population_plus_fold_plus_perturbation | 0.166 | [0.022, 0.298] | 0.993 |

Santinha 是明确的外部失败边界。三数据集固定集合与 dataset-population 区间必须分别解释；只有三个数据集时，后者通常很宽，不能声称对未来数据集已有稳定保证。
