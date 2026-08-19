# E124｜五套正式 scGPT–GEARS 数据集元分析

Shifrut 和 Liang 的合同在 E113 三数据集结果之后冻结；Liang 又在 Shifrut 解封后冻结。全部数据使用同背景 control 输入、正式 scGPT/GEARS 和相同风险公式。

| dataset | SafeConf | frozen | disagreement | magnitude |
|---|---:|---:|---:|---:|
| Frangieh | 0.253 | 0.242 | 0.137 | 0.148 |
| Lara_exvivo | 0.387 | 0.355 | 0.176 | 0.148 |
| Liang | 0.212 | 0.202 | 0.075 | 0.074 |
| Santinha | 0.065 | -0.095 | -0.127 | -0.089 |
| Shifrut | 0.173 | 0.053 | 0.051 | 0.209 |

## Bootstrap

| scope | comparator | unit | Δρ | 95% CI | P(Δ>0) |
|---|---|---|---:|---:|---:|
| Frangieh | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.011 | [-0.028, 0.053] | 0.700 |
| Lara_exvivo | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.032 | [-0.017, 0.095] | 0.822 |
| Liang | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.010 | [-0.116, 0.107] | 0.558 |
| Santinha | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.160 | [-0.076, 0.427] | 0.913 |
| Shifrut | safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.119 | [-0.047, 0.341] | 0.909 |
| five_dataset_macro | safeconf_frozen_pair_risk | fixed_five_datasets | 0.066 | [0.000, 0.139] | 0.975 |
| five_dataset_macro | safeconf_frozen_pair_risk | dataset_population_plus_fold_plus_perturbation | 0.066 | [-0.009, 0.170] | 0.951 |
| Frangieh | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.116 | [-0.038, 0.286] | 0.926 |
| Lara_exvivo | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.211 | [0.066, 0.352] | 1.000 |
| Liang | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.138 | [-0.077, 0.335] | 0.895 |
| Santinha | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.191 | [-0.090, 0.479] | 0.912 |
| Shifrut | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.121 | [-0.082, 0.377] | 0.884 |
| five_dataset_macro | risk_model_disagreement | fixed_five_datasets | 0.155 | [0.059, 0.246] | 0.999 |
| five_dataset_macro | risk_model_disagreement | dataset_population_plus_fold_plus_perturbation | 0.155 | [0.055, 0.249] | 1.000 |
| Frangieh | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.105 | [-0.036, 0.245] | 0.922 |
| Lara_exvivo | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.240 | [0.082, 0.405] | 1.000 |
| Liang | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.138 | [-0.064, 0.337] | 0.927 |
| Santinha | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.154 | [-0.188, 0.440] | 0.817 |
| Shifrut | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | -0.036 | [-0.228, 0.258] | 0.378 |
| five_dataset_macro | baseline_predicted_magnitude | fixed_five_datasets | 0.120 | [0.021, 0.221] | 0.992 |
| five_dataset_macro | baseline_predicted_magnitude | dataset_population_plus_fold_plus_perturbation | 0.120 | [-0.006, 0.235] | 0.968 |

Shifrut 中 SafeConf 没有超过 magnitude，Liang 则形成新的正复制。两者均保留在总体推断中，不按结果删除。五数据集总体结论优先于 E113 的三数据集结论。
