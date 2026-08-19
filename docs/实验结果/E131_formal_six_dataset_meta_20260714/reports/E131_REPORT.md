# E131｜六套正式 scGPT–GEARS 数据元分析

Tian–Kampmann 在提交 `19e5e23` 冻结合同后解封。Tian 的 context 是技术批次，其余五套包含生物背景变化。所有失败数据集均保留。

| dataset | SafeConf | frozen | disagreement | magnitude |
|---|---:|---:|---:|---:|
| Frangieh | 0.253 | 0.242 | 0.137 | 0.148 |
| Lara_exvivo | 0.387 | 0.355 | 0.176 | 0.148 |
| Liang | 0.212 | 0.202 | 0.075 | 0.074 |
| Santinha | 0.065 | -0.095 | -0.127 | -0.089 |
| Shifrut | 0.173 | 0.053 | 0.051 | 0.209 |
| Tian_CRISPRi | 0.134 | 0.111 | -0.018 | 0.067 |

## Bootstrap

| scope | comparator | unit | Δρ | 95% CI | P(Δ>0) |
|---|---|---|---:|---:|---:|
| Frangieh | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.011 | [-0.030, 0.058] | 0.695 |
| Lara_exvivo | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.032 | [-0.018, 0.096] | 0.822 |
| Liang | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.010 | [-0.117, 0.108] | 0.544 |
| Santinha | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.160 | [-0.076, 0.426] | 0.913 |
| Shifrut | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.119 | [-0.043, 0.339] | 0.919 |
| Tian_CRISPRi | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.023 | [-0.102, 0.165] | 0.623 |
| six_dataset_macro | safeconf_frozen_pair_risk | fixed_six_datasets | 0.059 | [0.001, 0.123] | 0.978 |
| six_dataset_macro | safeconf_frozen_pair_risk | dataset_population_plus_fold_plus_perturbation | 0.059 | [-0.007, 0.150] | 0.960 |
| Frangieh | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.116 | [-0.028, 0.280] | 0.931 |
| Lara_exvivo | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.211 | [0.060, 0.348] | 1.000 |
| Liang | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.138 | [-0.081, 0.338] | 0.888 |
| Santinha | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.191 | [-0.098, 0.494] | 0.895 |
| Shifrut | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.121 | [-0.076, 0.359] | 0.879 |
| Tian_CRISPRi | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.152 | [-0.111, 0.416] | 0.869 |
| six_dataset_macro | risk_model_disagreement | fixed_six_datasets | 0.155 | [0.063, 0.246] | 0.999 |
| six_dataset_macro | risk_model_disagreement | dataset_population_plus_fold_plus_perturbation | 0.155 | [0.059, 0.251] | 0.999 |
| Frangieh | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.105 | [-0.036, 0.255] | 0.926 |
| Lara_exvivo | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.240 | [0.086, 0.398] | 1.000 |
| Liang | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.138 | [-0.061, 0.333] | 0.918 |
| Santinha | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.154 | [-0.160, 0.430] | 0.806 |
| Shifrut | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | -0.036 | [-0.225, 0.260] | 0.374 |
| Tian_CRISPRi | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.067 | [-0.115, 0.274] | 0.712 |
| six_dataset_macro | baseline_predicted_magnitude | fixed_six_datasets | 0.111 | [0.025, 0.200] | 0.996 |
| six_dataset_macro | baseline_predicted_magnitude | dataset_population_plus_fold_plus_perturbation | 0.111 | [0.001, 0.215] | 0.976 |
