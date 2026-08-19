# E200 K562 整体背景留出审计

- 完整性：**PASS**。
- 20% 复核路由：**ENABLED**。
- 相对 predicted magnitude 的新增价值：**NOT SUPPORTED**。
- GAT 主 RMSE 优于 general baseline：**YES**。

## 主误差与基线

| baseline | n_tasks | predictor_mean_error | baseline_mean_error | task_win_rate | mean_delta | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| general_baseline | 566 | 0.0542 | 0.0584 | 0.9187 | -0.0042 | -0.0046 | -0.0038 |
| batch_matched_control | 566 | 0.0542 | 0.0523 | 0.5053 | 0.0019 | 0.0006 | 0.0032 |

## 冻结风险量

| predictor | spearman | ci95_lower | ci95_upper |
| --- | --- | --- | --- |
| transfer_risk | 0.4240 | 0.3506 | 0.4953 |
| predicted_magnitude | 0.8797 | 0.8437 | 0.9095 |
| model_baseline_gap | 0.1597 | 0.0751 | 0.2415 |
| training_delta_dispersion | 0.6639 | 0.6075 | 0.7143 |
| negative_log_train_cells | 0.2149 | 0.1309 | 0.2955 |
| support_context_deficit | 0.0170 | -0.0662 | 0.1034 |

| predictor | n_selected | high_error_capture | error_lift | oracle_normalized_utility | utility_ci95_lower | utility_ci95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| transfer_risk | 114 | 0.3509 | 1.2085 | 0.3648 | 0.2356 | 0.4813 |
| predicted_magnitude | 114 | 0.7807 | 1.5220 | 0.9133 | 0.8748 | 0.9520 |

## 五个独立端点

| predictor | endpoint | n | mean_error | median_error |
| --- | --- | --- | --- | --- |
| batch_matched_control | de_auprc | 566 | 0.7580 | 0.7833 |
| batch_matched_control | energy_distance_pca_k=50 | 566 | 2.6133 | 2.1310 |
| batch_matched_control | mse | 566 | 0.0032 | 0.0023 |
| batch_matched_control | pearson_pert | 566 | 0.9028 | 0.9012 |
| batch_matched_control | rank | 566 | 0.4108 | 0.3735 |
| gat | de_auprc | 566 | 0.7477 | 0.7721 |
| gat | energy_distance_pca_k=50 | 566 | 2.4760 | 2.2351 |
| gat | mse | 566 | 0.0034 | 0.0026 |
| gat | pearson_pert | 566 | 0.5880 | 0.5491 |
| gat | rank | 566 | 0.2108 | 0.1009 |
| general_baseline | de_auprc | 566 | 0.7711 | 0.8045 |
| general_baseline | energy_distance_pca_k=50 | 566 | 2.4773 | 2.2416 |
| general_baseline | mse | 566 | 0.0038 | 0.0031 |
| general_baseline | pearson_pert | 566 | 0.5823 | 0.5622 |
| general_baseline | rank | 566 | 0.1938 | 0.0823 |

## 边界

E200 只是 K562 目标背景上的公开 GAT checkpoint。它能回答整行留出的一个真实实例，不代表其他目标细胞系、多模型家族或跨独立数据集已经回答。
