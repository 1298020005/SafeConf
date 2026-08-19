# E26 GEARS single-model risk audit

生成时间：2026-07-08T02:59:59

## 结论

- 输入：E25 strict GEARS PredictionRecord 包。
- 记录：54 条，覆盖 3 个数据集。
- E25 strict validator issue_count：0。
- 可部署单模型分数：4 个。
- GEARS native uncertainty：不可用，E25 formal records 全为空。

整体上，GEARS predicted-effect magnitude 与真实误差呈正相关；这说明在 GEARS 单模型场景下，效应幅度仍是重要风险线索。该结果不能写成多模型不确定性验证，因为当前没有 scGPT/CPA 对齐输出，也没有模型间 disagreement。

## Overall score summary

| score_name | score_type | n | direction_aligned_spearman | aurc | random_aurc | oracle_aurc | risk_cov_improve_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true_effect_abs_mean_diagnostic | risk | 54 | 0.7673 | 0.0423 | 0.0710 | 0.0370 | 36.0107 |
| true_effect_l2_diagnostic | risk | 54 | 0.7553 | 0.0424 | 0.0710 | 0.0370 | 36.3052 |
| gears_predicted_effect_abs_mean_risk | risk | 54 | 0.6792 | 0.0426 | 0.0710 | 0.0370 | 34.1988 |
| gears_predicted_effect_l2_risk | risk | 54 | 0.6239 | 0.0437 | 0.0710 | 0.0370 | 33.3607 |
| gears_low_support_risk | risk | 54 | 0.2532 | 0.1206 | 0.0710 | 0.0370 | -2.8913 |
| gears_cell_support_confidence | confidence | 54 | 0.2532 | 0.1206 | 0.0710 | 0.0370 | -2.6111 |

## Dataset-level summary

| dataset_name | score_name | n | direction_aligned_spearman | risk_cov_improve_pct |
| --- | --- | --- | --- | --- |
| adamson | gears_cell_support_confidence | 21 | 0.6582 | 2.0837 |
| adamson | gears_low_support_risk | 21 | 0.6582 | 2.0837 |
| adamson | gears_predicted_effect_abs_mean_risk | 21 | 0.5857 | 0.8130 |
| adamson | gears_predicted_effect_l2_risk | 21 | 0.4221 | 0.8130 |
| adamson | true_effect_abs_mean_diagnostic | 21 | 0.8520 | 18.5317 |
| adamson | true_effect_l2_diagnostic | 21 | 0.8741 | 17.4570 |
| dixit | gears_cell_support_confidence | 3 | 0.8660 | 0.0000 |
| dixit | gears_low_support_risk | 3 | 0.8660 | 0.0000 |
| dixit | gears_predicted_effect_abs_mean_risk | 3 | 0.5000 | 0.0000 |
| dixit | gears_predicted_effect_l2_risk | 3 | 0.5000 | 0.0000 |
| dixit | true_effect_abs_mean_diagnostic | 3 | 0.8660 | 0.0000 |
| dixit | true_effect_l2_diagnostic | 3 | 0.8660 | 0.0000 |
| norman | gears_cell_support_confidence | 30 | 0.2418 | 5.5589 |
| norman | gears_low_support_risk | 30 | 0.2418 | 5.5589 |
| norman | gears_predicted_effect_abs_mean_risk | 30 | 0.6294 | 7.6099 |
| norman | gears_predicted_effect_l2_risk | 30 | 0.6236 | 7.6099 |
| norman | true_effect_abs_mean_diagnostic | 30 | 0.5338 | 9.6973 |
| norman | true_effect_l2_diagnostic | 30 | 0.5342 | 7.5814 |

## Partial Spearman controlling true magnitude

| scope | score_name | n | partial_spearman_control_true_l2 | partial_spearman_control_true_abs_mean |
| --- | --- | --- | --- | --- |
| overall | gears_cell_support_confidence | 54 | -0.2546 | -0.1778 |
| overall | gears_low_support_risk | 54 | 0.2546 | 0.1778 |
| overall | gears_predicted_effect_abs_mean_risk | 54 | 0.5895 | 0.5284 |
| overall | gears_predicted_effect_l2_risk | 54 | 0.5453 | 0.4994 |
| overall | true_effect_abs_mean_diagnostic | 54 | 0.3126 | -0.1136 |
| overall | true_effect_l2_diagnostic | 54 | 0.0617 | 0.0653 |

## Top-20% high-error enrichment

| scope | score_name | score_type | n | top_fraction | k | high_error_threshold_p80 | base_high_error_rate | picked_high_error_rate | enrichment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | gears_cell_support_confidence | confidence | 54 | 0.2000 | 11 | 0.0697 | 0.2037 | 0.4545 | 2.2314 |
| overall | gears_low_support_risk | risk | 54 | 0.2000 | 11 | 0.0697 | 0.2037 | 0.4545 | 2.2314 |
| overall | gears_predicted_effect_abs_mean_risk | risk | 54 | 0.2000 | 11 | 0.0697 | 0.2037 | 0.6364 | 3.1240 |
| overall | gears_predicted_effect_l2_risk | risk | 54 | 0.2000 | 11 | 0.0697 | 0.2037 | 0.6364 | 3.1240 |

## 边界

1. E26 是 GEARS-only，不能代表 GEARS/scGPT/CPA 统一多模型验证。
2. true-effect magnitude 是诊断量，部署时不可用。
3. Dixit 只有 3 条记录，只能保留在 overall 里，不能单独强解释。
