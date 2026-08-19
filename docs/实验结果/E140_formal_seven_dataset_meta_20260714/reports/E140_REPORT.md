# E140｜七套正式数据 absolute-RMSE 元分析

Nadig 是为 E135 方向风险头冻结的第七数据，同时把其原 SafeConf absolute-RMSE 结果并入总账，不能只报告方向风险成功。

| dataset | SafeConf | frozen | disagreement | magnitude |
|---|---:|---:|---:|---:|
| Frangieh | 0.253 | 0.242 | 0.137 | 0.148 |
| Lara_exvivo | 0.387 | 0.355 | 0.176 | 0.148 |
| Liang | 0.212 | 0.202 | 0.075 | 0.074 |
| Nadig_two_cellline | 0.231 | 0.243 | 0.230 | 0.403 |
| Santinha | 0.065 | -0.095 | -0.127 | -0.089 |
| Shifrut | 0.173 | 0.053 | 0.051 | 0.209 |
| Tian_CRISPRi | 0.134 | 0.111 | -0.018 | 0.067 |

## Bootstrap

| scope | comparator | unit | Δρ | 95% CI | P(Δ>0) |
|---|---|---|---:|---:|---:|
| Frangieh | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.011 | [-0.028, 0.057] | 0.686 |
| Lara_exvivo | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.032 | [-0.018, 0.091] | 0.813 |
| Liang | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.010 | [-0.120, 0.109] | 0.578 |
| Nadig_two_cellline | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | -0.012 | [-0.089, 0.067] | 0.363 |
| Santinha | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.160 | [-0.082, 0.414] | 0.903 |
| Shifrut | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.119 | [-0.053, 0.351] | 0.911 |
| Tian_CRISPRi | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.023 | [-0.102, 0.166] | 0.601 |
| seven_dataset_macro | safeconf_frozen_pair_risk | fixed_seven_datasets | 0.049 | [-0.005, 0.105] | 0.960 |
| seven_dataset_macro | safeconf_frozen_pair_risk | dataset_population_plus_fold_plus_perturbation | 0.049 | [-0.011, 0.132] | 0.935 |
| Frangieh | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.116 | [-0.032, 0.282] | 0.928 |
| Lara_exvivo | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.211 | [0.063, 0.346] | 0.999 |
| Liang | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.138 | [-0.084, 0.329] | 0.882 |
| Nadig_two_cellline | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.001 | [-0.087, 0.106] | 0.496 |
| Santinha | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.191 | [-0.105, 0.491] | 0.907 |
| Shifrut | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.121 | [-0.069, 0.369] | 0.884 |
| Tian_CRISPRi | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.152 | [-0.112, 0.418] | 0.856 |
| seven_dataset_macro | risk_model_disagreement | fixed_seven_datasets | 0.133 | [0.052, 0.211] | 1.000 |
| seven_dataset_macro | risk_model_disagreement | dataset_population_plus_fold_plus_perturbation | 0.133 | [0.045, 0.223] | 0.999 |
| Frangieh | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.105 | [-0.043, 0.256] | 0.914 |
| Lara_exvivo | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.240 | [0.081, 0.407] | 0.999 |
| Liang | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.138 | [-0.071, 0.332] | 0.912 |
| Nadig_two_cellline | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | -0.172 | [-0.419, 0.023] | 0.064 |
| Santinha | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.154 | [-0.174, 0.434] | 0.821 |
| Shifrut | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | -0.036 | [-0.231, 0.236] | 0.382 |
| Tian_CRISPRi | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.067 | [-0.110, 0.281] | 0.733 |
| seven_dataset_macro | baseline_predicted_magnitude | fixed_seven_datasets | 0.071 | [-0.013, 0.149] | 0.950 |
| seven_dataset_macro | baseline_predicted_magnitude | dataset_population_plus_fold_plus_perturbation | 0.071 | [-0.056, 0.186] | 0.868 |

## 边界

Nadig 的 absolute RMSE 中 magnitude 明显强于原 SafeConf。E139 的 Directional-SafeConf 通过不能覆盖这个负结果；两种误差必须使用各自风险头并分别报告。
