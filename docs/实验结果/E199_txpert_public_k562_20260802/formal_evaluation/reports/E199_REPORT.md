# E199 TxPert 公开 K562 未见扰动审计

确定性证书：**PASS**。
经验路由：**ENABLED**。
相对 predicted magnitude 的新增价值：**SUPPORTED**。

## 五个冻结端点（主分析）

| predictor | endpoint | n | mean | median |
| --- | --- | --- | --- | --- |
| batch_matched_control | de_auprc | 263 | 0.2179 | 0.1865 |
| batch_matched_control | energy_distance_pca_k=50 | 263 | 2.8893 | 2.3859 |
| batch_matched_control | mse | 263 | 0.0025 | 0.0018 |
| batch_matched_control | pearson_pert | 263 | 0.0926 | 0.1088 |
| batch_matched_control | rank | 263 | 0.3996 | 0.3511 |
| exphormer | de_auprc | 263 | 0.3395 | 0.2869 |
| exphormer | energy_distance_pca_k=50 | 263 | 2.3910 | 2.1917 |
| exphormer | mse | 263 | 0.0015 | 0.0011 |
| exphormer | pearson_pert | 263 | 0.5235 | 0.5475 |
| exphormer | rank | 263 | 0.1085 | 0.0382 |
| exphormer_mg | de_auprc | 263 | 0.3435 | 0.2785 |
| exphormer_mg | energy_distance_pca_k=50 | 263 | 2.3405 | 2.1551 |
| exphormer_mg | mse | 263 | 0.0014 | 0.0010 |
| exphormer_mg | pearson_pert | 263 | 0.5282 | 0.5605 |
| exphormer_mg | rank | 263 | 0.1072 | 0.0382 |
| family_centroid | de_auprc | 263 | 0.3482 | 0.2867 |
| family_centroid | energy_distance_pca_k=50 | 263 | 2.3362 | 2.1319 |
| family_centroid | mse | 263 | 0.0014 | 0.0011 |
| family_centroid | pearson_pert | 263 | 0.5490 | 0.5941 |
| family_centroid | rank | 263 | 0.1003 | 0.0344 |
| gat | de_auprc | 263 | 0.3311 | 0.2888 |
| gat | energy_distance_pca_k=50 | 263 | 2.4254 | 2.1910 |
| gat | mse | 263 | 0.0015 | 0.0011 |
| gat | pearson_pert | 263 | 0.4942 | 0.5547 |
| gat | rank | 263 | 0.1453 | 0.0534 |
| general_baseline | de_auprc | 263 | 0.1318 | 0.0826 |
| general_baseline | energy_distance_pca_k=50 | 263 | 2.8264 | 2.4286 |
| general_baseline | mse | 263 | 0.0022 | 0.0017 |
| general_baseline | pearson_pert | 263 | -0.0024 | -0.0327 |
| general_baseline | rank | 263 | 0.3926 | 0.3397 |

## 简单基线对照（centroid RMSE）

| predictor | predictor_mean_rmse | baseline_mean_rmse | task_win_rate | mean_delta | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| gat | 0.0367 | 0.0442 | 0.7643 | -0.0075 | -0.0090 | -0.0060 |
| exphormer | 0.0356 | 0.0442 | 0.8859 | -0.0086 | -0.0098 | -0.0075 |
| exphormer_mg | 0.0353 | 0.0442 | 0.9011 | -0.0089 | -0.0101 | -0.0078 |
| family_centroid | 0.0349 | 0.0442 | 0.9011 | -0.0093 | -0.0105 | -0.0082 |
| family_member_rms | 0.0360 | 0.0442 | 0.8631 | -0.0081 | -0.0093 | -0.0070 |
| general_baseline | 0.0442 | 0.0442 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| batch_matched_control | 0.0466 | 0.0442 | 0.3612 | 0.0025 | 0.0017 | 0.0033 |

## SafeConf 风险量

| predictor | spearman | ci95_lower | ci95_upper |
| --- | --- | --- | --- |
| diversity_lower_bound | 0.3948 | 0.2835 | 0.4969 |
| predicted_magnitude | 0.0955 | -0.0256 | 0.2187 |
| model_baseline_gap | -0.0064 | -0.1332 | 0.1185 |
| negative_string_train_neighbor_count | -0.0822 | -0.2052 | 0.0388 |
| negative_go_train_neighbor_count | -0.1018 | -0.2200 | 0.0165 |
| graph_isolated_risk | -0.1067 | -0.1961 | -0.0029 |

| predictor | high_error_capture | error_lift | oracle_normalized_utility | utility_ci95_lower | utility_ci95_upper |
| --- | --- | --- | --- | --- | --- |
| diversity_lower_bound | 0.2642 | 1.1205 | 0.2084 | 0.1033 | 0.3755 |
| predicted_magnitude | 0.2264 | 1.0230 | 0.0397 | -0.0830 | 0.2255 |
| model_baseline_gap | 0.2075 | 0.9913 | -0.0150 | -0.1278 | 0.1781 |
| negative_string_train_neighbor_count | 0.1887 | 0.9517 | -0.0835 | -0.2408 | 0.0418 |
| negative_go_train_neighbor_count | 0.1132 | 0.9111 | -0.1537 | -0.2578 | 0.0024 |
| graph_isolated_risk | 0.0566 | 0.8554 | -0.2500 | -0.1790 | 0.1222 |

## 边界

E199 只是 K562 内未见单基因扰动。它能回答模型家族是否显示任务级难度，不能代替整个细胞背景留出或跨数据集迁移。公开 Exphormer-MG 只使用 STRING+GO，不包含 TxPert 论文最强配置中未公开的 PxMap/TxMap。

主分析 263 个任务；目标基因效应方向命中率为 0.490，该项是预先写入 formal runner 的失败模式诊断。
