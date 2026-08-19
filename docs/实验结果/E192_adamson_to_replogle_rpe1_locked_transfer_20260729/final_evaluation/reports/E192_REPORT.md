# E192 Adamson→Replogle RPE1 锁定确认结果

确定性证书 gate：**PASS**。
经验排序激活 gate：**FAIL / ABSTAIN**。

## 预测器与 zero-effect

| estimator | mean_rmse | zero_mean_rmse | task_win_rate_vs_zero | gene_cluster_mean_delta | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| scGPT | 0.29773 | 0.29789 | 0.60000 | -0.00037 | -0.00056 | -0.00020 |
| GEARS | 0.30862 | 0.29789 | 0.13714 | 0.00466 | -0.00153 | 0.01037 |
| six_model_family | 0.30327 | 0.29789 | 0.13143 | 0.00223 | -0.00088 | 0.00512 |
| source_effect | 0.30848 | 0.29789 | 0.12571 | 0.00517 | -0.00091 | 0.01074 |

## 风险量与真实误差

| predictor | outcome | spearman | ci95_lower | ci95_upper | bootstrap_valid |
| --- | --- | --- | --- | --- | --- |
| diversity_lower_bound | family_rms_error | 0.29976 | -0.03997 | 0.57951 | 5000 |
| diameter_half_lower_bound | family_worst_error | 0.25695 | -0.08092 | 0.54065 | 5000 |
| predicted_magnitude | family_rms_error | 0.34167 | -0.00110 | 0.60247 | 5000 |
| source_effect_magnitude | family_rms_error | 0.18483 | -0.15229 | 0.48697 | 5000 |

## 固定复核预算

| budget | n_selected | high_error_capture | random_expected_capture | selected_mean_error | overall_mean_error | error_lift | oracle_mean_error | oracle_normalized_utility | predictor | utility_ci95_lower | utility_ci95_upper | utility_bootstrap_valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10000 | 18 | 0.50000 | 0.10286 | 0.41337 | 0.30327 | 1.36305 | 0.50908 | 0.53498 | diversity_lower_bound | 0.09724 | 0.86580 | 3000 |
| 0.20000 | 35 | 0.62857 | 0.20000 | 0.40393 | 0.30327 | 1.33193 | 0.44795 | 0.69577 | diversity_lower_bound | 0.11348 | 0.87216 | 3000 |
| 0.30000 | 53 | 0.52830 | 0.30286 | 0.35896 | 0.30327 | 1.18363 | 0.40974 | 0.52304 | diversity_lower_bound | 0.11806 | 0.84714 | 3000 |
| 0.10000 | 18 | 0.50000 | 0.10286 | 0.40712 | 0.30327 | 1.34243 | 0.50908 | 0.50459 | predicted_magnitude | 0.09277 | 0.82491 | 3000 |
| 0.20000 | 35 | 0.62857 | 0.20000 | 0.40812 | 0.30327 | 1.34574 | 0.44795 | 0.72471 | predicted_magnitude | 0.19155 | 0.86103 | 3000 |
| 0.30000 | 53 | 0.66038 | 0.30286 | 0.37489 | 0.30327 | 1.23618 | 0.40974 | 0.67272 | predicted_magnitude | 0.17436 | 0.84688 | 3000 |
| 0.10000 | 18 | 0.44444 | 0.10286 | 0.39608 | 0.30327 | 1.30605 | 0.50908 | 0.45098 | source_effect_magnitude | -0.03355 | 0.77037 | 3000 |
| 0.20000 | 35 | 0.40000 | 0.20000 | 0.34565 | 0.30327 | 1.13974 | 0.44795 | 0.29291 | source_effect_magnitude | -0.09579 | 0.75271 | 3000 |
| 0.30000 | 53 | 0.52830 | 0.30286 | 0.35896 | 0.30327 | 1.18363 | 0.40974 | 0.52304 | source_effect_magnitude | -0.12935 | 0.73535 | 3000 |

证书 gate 与排序 gate 分开裁决。确定性下界成立，不能自动把外部 RPE1 setting 的经验排序改为可用；只有预注册的三个排序条件全部通过才启用。
