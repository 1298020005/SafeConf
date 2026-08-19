# E85｜CPA 化学四象限选择性路由

E85 不训练模型、不改分数，只把 E84 冻结的 disagreement 与 predicted magnitude 放到预先计划的选择性预测指标中：top-20% 高错误富集、拒绝最高风险 20% 后的剩余误差下降、coverage 50%–100% 的归一化 AURC。随机路由的期望分别为 1、0、1。

| quadrant | metric | n_manifests | model_disagreement_mean | predicted_magnitude_mean | favorable_delta_definition | favorable_delta_mean | favorable_delta_bootstrap_ci95_low | favorable_delta_bootstrap_ci95_high | manifests_where_disagreement_better |
|---|---|---|---|---|---|---|---|---|---|
| new_context_new_perturbation | top20_error_enrichment | 8 | 1.171 | 1.174 | disagreement_minus_magnitude | -0.003 | -0.015 | 0.006 | 2 |
| new_context_new_perturbation | reject20_remaining_error_reduction | 8 | 0.052 | 0.052 | disagreement_minus_magnitude | -0.001 | -0.004 | 0.001 | 3 |
| new_context_new_perturbation | normalized_aurc_50_100 | 8 | 0.953 | 0.953 | magnitude_minus_disagreement | -0.0 | -0.002 | 0.001 | 6 |
| new_context_seen_perturbation | top20_error_enrichment | 8 | 1.196 | 1.218 | disagreement_minus_magnitude | -0.022 | -0.056 | 0.001 | 1 |
| new_context_seen_perturbation | reject20_remaining_error_reduction | 8 | 0.059 | 0.065 | disagreement_minus_magnitude | -0.006 | -0.015 | 0.0 | 1 |
| new_context_seen_perturbation | normalized_aurc_50_100 | 8 | 0.945 | 0.942 | magnitude_minus_disagreement | -0.003 | -0.006 | -0.001 | 1 |
| seen_context_new_perturbation | top20_error_enrichment | 8 | 1.112 | 1.153 | disagreement_minus_magnitude | -0.041 | -0.103 | 0.021 | 3 |
| seen_context_new_perturbation | reject20_remaining_error_reduction | 8 | 0.03 | 0.042 | disagreement_minus_magnitude | -0.012 | -0.029 | 0.005 | 3 |
| seen_context_new_perturbation | normalized_aurc_50_100 | 8 | 0.968 | 0.953 | magnitude_minus_disagreement | -0.016 | -0.03 | -0.005 | 2 |
| seen_context_seen_perturbation_pair_holdout | top20_error_enrichment | 8 | 1.143 | 1.144 | disagreement_minus_magnitude | -0.001 | -0.016 | 0.013 | 1 |
| seen_context_seen_perturbation_pair_holdout | reject20_remaining_error_reduction | 8 | 0.061 | 0.061 | disagreement_minus_magnitude | -0.001 | -0.006 | 0.004 | 1 |
| seen_context_seen_perturbation_pair_holdout | normalized_aurc_50_100 | 8 | 0.963 | 0.964 | magnitude_minus_disagreement | 0.001 | -0.002 | 0.005 | 4 |

选择性指标比 Spearman 更严格：四个象限中，两种分数的 top-20%、reject-20% 和 AURC 大多接近。新 context 与新药的 AURC 由 magnitude 稳定占优；随机缺失 pair 的 disagreement 只有极小、区间跨 0 的优势。E84 的排序相关不能直接换写成明显的资源节省。

正的 favorable delta 表示 disagreement 优于 magnitude。区间按 8 个 manifest 成对重采样，只描述同一 sciPlex3 内冻结 split 的敏感性。图 `figures/F1_risk_coverage_four_quadrants.svg` 为白底，可直接用于汇报。
