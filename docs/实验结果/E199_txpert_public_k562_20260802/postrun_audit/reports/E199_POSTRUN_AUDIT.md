# E199 运行后稳健性审计

> **POST HOC / EXPLORATORY。** 本报告不改变正式 gate。

## 不含 diversity 代数项的误差

| outcome | spearman | ci95_lower | ci95_upper | oracle_normalized_utility | utility_ci95_lower | utility_ci95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| centroid_error | 0.3251 | 0.2079 | 0.4301 | 0.1283 | 0.0275 | 0.2941 |
| family_worst_error | 0.4940 | 0.3872 | 0.5903 | 0.3519 | 0.2358 | 0.5149 |
| gat_rmse | 0.4255 | 0.3151 | 0.5233 | 0.3065 | 0.1827 | 0.4702 |
| exphormer_rmse | 0.3197 | 0.2020 | 0.4293 | 0.1402 | 0.0307 | 0.3243 |
| exphormer_mg_rmse | 0.2890 | 0.1713 | 0.4024 | 0.0868 | -0.0209 | 0.2553 |

对等权均值真实 RMSE，diversity 的 Spearman 为 0.325 （95% CI 0.208–0.430）；20% 复核效用为 0.128 （95% CI 0.027–0.294）。

## 五个端点的依赖性

| endpoint | spearman | ci95_lower | ci95_upper |
| --- | --- | --- | --- |
| de_auprc | -0.0755 | -0.1910 | 0.0426 |
| energy_distance_pca_k=50 | 0.2338 | 0.1169 | 0.3487 |
| mse | 0.3251 | 0.2145 | 0.4316 |
| pearson_pert | 0.0272 | -0.0904 | 0.1466 |
| rank | 0.0623 | -0.0645 | 0.1911 |

MSE 和群体距离上的方向较清楚；Pearson、检索 rank 与 DE-AUPRC 不显示同样的一致性。因此 E199 支持的是特定误差口径下的风险路由，不支持跨端点通用声明。

## 目标基因方向分母

| analysis_status | stratum | n_total_tasks | n_evaluable_target_genes | n_direction_correct | direction_accuracy | n_direction_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| POST_HOC_DENOMINATOR_CLARIFICATION | all_ge10 | 272 | 107 | 52 | 0.4860 | 55 |
| POST_HOC_DENOMINATOR_CLARIFICATION | primary_ge30 | 263 | 104 | 51 | 0.4904 | 53 |
| POST_HOC_DENOMINATOR_CLARIFICATION | sensitivity_10_29 | 9 | 3 | 1 | 0.3333 | 2 |

主分析 263 个任务中只有 104 个目标基因位于 5,000 基因面板；其中 51 个方向正确，命中率 0.490。

## 仍未回答

整个细胞背景留出和跨数据集迁移仍未回答，不能由 E199 外推。
