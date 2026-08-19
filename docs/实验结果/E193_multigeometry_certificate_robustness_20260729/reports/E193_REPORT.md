# E193 多几何注册家族证书结果

确定性实现 gate：**PASS**。

E193 是开真值后的指标稳健性分析。证书恒等式与下界属于确定性复核；相关性和复核收益全部是探索性结果。

## 证书审计

| dataset | geometry | n_valid_tasks | mean_family_rms_error | mean_diversity_lower_bound | family_rms_lower_violations | family_worst_lower_violations | max_rms_identity_residual | raw_replication_max_abs_diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E190_K562 | absolute_rmse | 692 | 0.24919 | 0.07331 | 0 | 0 | 0.00000 | 0.00000 |
| E190_K562 | cosine | 692 | 0.89054 | 0.59169 | 0 | 0 | 0.00000 | nan |
| E190_K562 | pearson | 692 | 0.91149 | 0.60142 | 0 | 0 | 0.00000 | nan |
| E192_RPE1 | absolute_rmse | 175 | 0.30327 | 0.04731 | 0 | 0 | 0.00000 | 0.00000 |
| E192_RPE1 | cosine | 175 | 0.98079 | 0.58476 | 0 | 0 | 0.00000 | nan |
| E192_RPE1 | pearson | 175 | 0.98839 | 0.60467 | 0 | 0 | 0.00000 | nan |

## 同几何 diversity 与 family error

| dataset | geometry | n_tasks | n_gene_clusters | spearman | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| E190_K562 | absolute_rmse | 692 | 47 | 0.42388 | 0.14088 | 0.63109 |
| E190_K562 | cosine | 692 | 47 | 0.56807 | 0.27846 | 0.78254 |
| E190_K562 | pearson | 692 | 47 | 0.24535 | -0.27664 | 0.60455 |
| E192_RPE1 | absolute_rmse | 175 | 21 | 0.29976 | -0.05666 | 0.58351 |
| E192_RPE1 | cosine | 175 | 21 | 0.04815 | -0.15956 | 0.29007 |
| E192_RPE1 | pearson | 175 | 21 | -0.21019 | -0.50704 | 0.03854 |

## Diversity 相对基线的配对相关差

| dataset | geometry | comparator | predictor_spearman | comparator_spearman | spearman_delta | delta_ci95_lower | delta_ci95_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E190_K562 | absolute_rmse | raw_predicted_magnitude | 0.42388 | 0.41976 | 0.00412 | -0.03345 | 0.04491 |
| E190_K562 | absolute_rmse | raw_diversity_lower_bound | 0.42388 | 0.42388 | 0.00000 | 0.00000 | 0.00000 |
| E190_K562 | absolute_rmse | source_effect_magnitude | 0.42388 | 0.41909 | 0.00479 | -0.04092 | 0.05222 |
| E190_K562 | absolute_rmse | source_to_family_centroid_distance | 0.42388 | 0.41641 | 0.00747 | -0.08546 | 0.10248 |
| E190_K562 | cosine | raw_predicted_magnitude | 0.56807 | -0.83077 | 1.39884 | 0.98576 | 1.63633 |
| E190_K562 | cosine | raw_diversity_lower_bound | 0.56807 | -0.83135 | 1.39942 | 1.00800 | 1.63886 |
| E190_K562 | cosine | source_effect_magnitude | 0.56807 | -0.82744 | 1.39551 | 1.01392 | 1.63427 |
| E190_K562 | cosine | source_to_family_centroid_distance | 0.56807 | 0.74664 | -0.17857 | -0.37545 | 0.00520 |
| E190_K562 | pearson | raw_predicted_magnitude | 0.24535 | -0.83215 | 1.07750 | 0.47296 | 1.43032 |
| E190_K562 | pearson | raw_diversity_lower_bound | 0.24535 | -0.83810 | 1.08345 | 0.47324 | 1.44094 |
| E190_K562 | pearson | source_effect_magnitude | 0.24535 | -0.81826 | 1.06362 | 0.45716 | 1.43283 |
| E190_K562 | pearson | source_to_family_centroid_distance | 0.24535 | 0.60571 | -0.36036 | -0.67501 | -0.13936 |
| E192_RPE1 | absolute_rmse | raw_predicted_magnitude | 0.29976 | 0.34167 | -0.04191 | -0.16799 | 0.01690 |
| E192_RPE1 | absolute_rmse | raw_diversity_lower_bound | 0.29976 | 0.29976 | 0.00000 | 0.00000 | 0.00000 |
| E192_RPE1 | absolute_rmse | source_effect_magnitude | 0.29976 | 0.18483 | 0.11493 | -0.01613 | 0.35861 |
| E192_RPE1 | absolute_rmse | source_to_family_centroid_distance | 0.29976 | 0.00748 | 0.29227 | 0.01708 | 0.61673 |
| E192_RPE1 | cosine | raw_predicted_magnitude | 0.04815 | -0.18758 | 0.23573 | -0.17945 | 0.73602 |
| E192_RPE1 | cosine | raw_diversity_lower_bound | 0.04815 | -0.15155 | 0.19970 | -0.23781 | 0.67480 |
| E192_RPE1 | cosine | source_effect_magnitude | 0.04815 | -0.17355 | 0.22171 | -0.14133 | 0.74444 |
| E192_RPE1 | cosine | source_to_family_centroid_distance | 0.04815 | 0.06594 | -0.01779 | -0.13851 | 0.08095 |
| E192_RPE1 | pearson | raw_predicted_magnitude | -0.21019 | -0.03938 | -0.17081 | -0.62151 | 0.32556 |
| E192_RPE1 | pearson | raw_diversity_lower_bound | -0.21019 | -0.04563 | -0.16456 | -0.66265 | 0.27046 |
| E192_RPE1 | pearson | source_effect_magnitude | -0.21019 | -0.06365 | -0.14654 | -0.52270 | 0.31067 |
| E192_RPE1 | pearson | source_to_family_centroid_distance | -0.21019 | -0.24769 | 0.03750 | -0.06325 | 0.14649 |

## 20% 复核预算

| dataset | geometry | predictor | high_error_capture | error_lift | oracle_normalized_utility | utility_ci95_lower | utility_ci95_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E190_K562 | absolute_rmse | diversity_lower_bound | 0.47482 | 1.14988 | 0.44303 | 0.01728 | 0.80714 |
| E190_K562 | absolute_rmse | raw_predicted_magnitude | 0.47482 | 1.14925 | 0.44117 | -0.02488 | 0.80132 |
| E190_K562 | absolute_rmse | source_effect_magnitude | 0.47482 | 1.15916 | 0.47047 | 0.01430 | 0.80169 |
| E190_K562 | absolute_rmse | source_to_family_centroid_distance | 0.43165 | 1.14098 | 0.41671 | 0.02264 | 0.80825 |
| E190_K562 | cosine | diversity_lower_bound | 0.58273 | 1.09393 | 0.78210 | 0.47942 | 0.92235 |
| E190_K562 | cosine | raw_predicted_magnitude | 0.00000 | 0.90990 | -0.75020 | -1.10990 | -0.42869 |
| E190_K562 | cosine | source_effect_magnitude | 0.00000 | 0.90631 | -0.78010 | -1.12102 | -0.50062 |
| E190_K562 | cosine | source_to_family_centroid_distance | 0.59712 | 1.09752 | 0.81205 | 0.58076 | 0.93623 |
| E190_K562 | pearson | diversity_lower_bound | 0.15827 | 0.99726 | -0.02585 | -0.42814 | 0.43470 |
| E190_K562 | pearson | raw_predicted_magnitude | 0.00000 | 0.91933 | -0.76167 | -1.12174 | -0.48274 |
| E190_K562 | pearson | source_effect_magnitude | 0.00000 | 0.92135 | -0.74261 | -1.13509 | -0.44867 |
| E190_K562 | pearson | source_to_family_centroid_distance | 0.43165 | 1.06715 | 0.63396 | 0.34404 | 0.81972 |
| E192_RPE1 | absolute_rmse | diversity_lower_bound | 0.62857 | 1.33193 | 0.69577 | 0.12553 | 0.87156 |
| E192_RPE1 | absolute_rmse | raw_predicted_magnitude | 0.62857 | 1.34574 | 0.72471 | 0.20189 | 0.86356 |
| E192_RPE1 | absolute_rmse | source_effect_magnitude | 0.40000 | 1.13207 | 0.27684 | -0.09256 | 0.73878 |
| E192_RPE1 | absolute_rmse | source_to_family_centroid_distance | 0.37143 | 1.10536 | 0.22085 | -0.19898 | 0.66296 |
| E192_RPE1 | cosine | diversity_lower_bound | 0.17143 | 1.00458 | 0.11292 | -0.28116 | 0.36108 |
| E192_RPE1 | cosine | raw_predicted_magnitude | 0.25714 | 0.98878 | -0.27629 | -1.03079 | 0.34955 |
| E192_RPE1 | cosine | source_effect_magnitude | 0.05714 | 0.97572 | -0.59807 | -1.05454 | 0.13718 |
| E192_RPE1 | cosine | source_to_family_centroid_distance | 0.14286 | 1.00195 | 0.04814 | -0.24328 | 0.40196 |
| E192_RPE1 | pearson | diversity_lower_bound | 0.05714 | 0.99441 | -0.13519 | -0.41966 | 0.12486 |
| E192_RPE1 | pearson | raw_predicted_magnitude | 0.22857 | 1.00513 | 0.12408 | -0.66947 | 0.54804 |
| E192_RPE1 | pearson | source_effect_magnitude | 0.02857 | 0.98615 | -0.33489 | -0.69514 | 0.34210 |
| E192_RPE1 | pearson | source_to_family_centroid_distance | 0.05714 | 0.99625 | -0.09056 | -0.49560 | 0.14917 |

方向几何中的证书值只能解释该几何定义的家族误差下界，不能改写成模型正确概率，也不能从本项 post-truth 分析推导新的选择性风险保证。
