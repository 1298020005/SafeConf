# E101｜三套独立多背景遗传扰动矩阵元分析

E101 不重新拟合分数。主分数是 E98/E100 在 test truth 解封前已经计算的 `safeconf_frozen_pair_risk`，由模型分歧、背景 control 新颖度和训练支持组成；强基线为同一双预测器输出的 predicted magnitude。每个 fold 先算 Spearman，再在数据集内和数据集间做等权宏平均。

## 100% 训练量 pooled setting

| dataset | frozen pair risk ρ | magnitude ρ | disagreement ρ | Δρ vs magnitude |
|---|---:|---:|---:|---:|
| Frangieh | 0.688 | 0.643 | 0.596 | 0.045 |
| Lara_exvivo | 0.229 | 0.043 | 0.085 | 0.186 |
| Santinha | 0.357 | 0.385 | 0.342 | -0.028 |
| 三数据集宏平均 | 0.425 | 0.357 | 0.341 | 0.067 |

## Bootstrap

| scope | comparator | unit | Δρ | 95% CI | P(Δ>0) |
|---|---|---|---:|---:|---:|
| Frangieh | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.045 | [-0.039, 0.170] | 0.716 |
| Lara_exvivo | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.186 | [0.017, 0.376] | 0.983 |
| Santinha | baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | -0.028 | [-0.159, 0.135] | 0.375 |
| three_dataset_macro | baseline_predicted_magnitude | fixed_datasets_fold_plus_perturbation | 0.067 | [-0.012, 0.157] | 0.955 |
| three_dataset_macro | baseline_predicted_magnitude | dataset_population_plus_fold_plus_perturbation | 0.067 | [-0.057, 0.213] | 0.847 |
| Frangieh | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.092 | [0.031, 0.176] | 1.000 |
| Lara_exvivo | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.144 | [0.069, 0.213] | 1.000 |
| Santinha | risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.015 | [-0.020, 0.056] | 0.834 |
| three_dataset_macro | risk_model_disagreement | fixed_datasets_fold_plus_perturbation | 0.083 | [0.048, 0.122] | 1.000 |
| three_dataset_macro | risk_model_disagreement | dataset_population_plus_fold_plus_perturbation | 0.083 | [0.017, 0.151] | 0.997 |

固定三数据集 bootstrap 回答这三份数据上的测量不确定性；dataset-population bootstrap 额外重采样数据集，回答推广到未来数据集的不确定性。只有 3 个数据集时，后者应作为主边界。

## Leave-one-dataset-out 敏感性

| removed | kept | comparator | macro Δρ |
|---|---|---|---:|
| Frangieh | Lara_exvivo+Santinha | baseline_predicted_magnitude | 0.079 |
| Frangieh | Lara_exvivo+Santinha | risk_model_disagreement | 0.079 |
| Lara_exvivo | Frangieh+Santinha | baseline_predicted_magnitude | 0.008 |
| Lara_exvivo | Frangieh+Santinha | risk_model_disagreement | 0.053 |
| Santinha | Frangieh+Lara_exvivo | baseline_predicted_magnitude | 0.115 |
| Santinha | Frangieh+Lara_exvivo | risk_model_disagreement | 0.118 |

## 结论边界

Frozen pair risk 在 Frangieh、Lara 为正增量，在 Santinha 略低于 magnitude。若固定这三套数据，宏平均可以衡量现有证据；若把数据集视作未来总体的随机样本，三个数据集仍不足以给出稳定推广保证。E101 不用校准后的分数替换 frozen 主分数，因此没有根据 Santinha 失败回调权重。

- `tables/E101_FOLD_SUMMARY.csv`
- `tables/E101_BOOTSTRAP.csv`
- `tables/E101_LEAVE_ONE_DATASET_OUT.csv`
- `figures/F1_frozen_vs_magnitude_forest.svg`
