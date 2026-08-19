# E13 官方 sciplex3 三细胞系 focused panel

生成时间：2026-07-07 14:29

## 1. 结论

E13 把官方 `sciplex3_A549.h5ad`、`sciplex3_K562.h5ad`、`sciplex3_MCF7.h5ad` 合成为一个三细胞系任务面板：context 为 cell line，perturbation 为 `drug_dose_name`。原始可构建 2,237 个任务，其中 743 个 drug-dose 同时出现在三种细胞系。为避免高维最近邻计算拖垮流程，本轮按每个 drug-dose 的最小细胞数与总细胞数选取 top 80，形成 240 个任务。

正式 test 记录 480 条，无 pair/context/perturbation leakage。最强信号为 `model_disagreement_risk`：aligned Spearman = 0.576，80% coverage RMSE 改善 = 6.42%。`learned_risk_score` 与 disagreement 非常接近：aligned Spearman = 0.572；`simple_combined_confidence` aligned Spearman = 0.366。

## 2. 面板构建

| dataset | n_tasks | n_contexts | n_perturbations | context_col | perturbation_col | n_genes | n_raw_tasks | n_shared_perturbations | n_selected_perturbations | selection_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 240 | 3 | 80 | cell_line | drug_dose_name | 5000 | 2237 | 743 | 80 | top 80 shared perturbations by min_cells, total_cells |

## 3. Split 审计

| dataset_name | fold_id | n_tasks_total | n_candidate_test_pairs | n_train | n_val | n_test | support_check_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 0 | 240 | 240 | 163 | 29 | 48 | True |
| sciplex3_official_3cell | 1 | 240 | 240 | 163 | 29 | 48 | True |
| sciplex3_official_3cell | 2 | 240 | 240 | 163 | 29 | 48 | True |
| sciplex3_official_3cell | 3 | 240 | 240 | 163 | 29 | 48 | True |
| sciplex3_official_3cell | 4 | 240 | 240 | 163 | 29 | 48 | True |

## 4. 主结果

| dataset_name | simple_combined_confidence_aligned_rho | simple_combined_confidence_risk_cov_80_improve_pct | learned_risk_score_aligned_rho | learned_risk_score_risk_cov_80_improve_pct | model_disagreement_risk_aligned_rho | model_disagreement_risk_risk_cov_80_improve_pct |
| --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 0.365698 | 6.90189 | 0.572339 | 3.82844 | 0.57623 | 6.41862 |

## 5. Overall 信号排序

| score_name | score_type | n | direction_aligned_spearman | risk_cov_80_improve_pct | high_low_rmse_gap |
| --- | --- | --- | --- | --- | --- |
| model_disagreement_risk | risk | 480 | 0.57623 | 6.41862 | 0.0159672 |
| learned_risk_score | risk | 480 | 0.572339 | 3.82844 | 0.0116867 |
| support_count_score | confidence | 480 | 0.463493 | 2.39803 | 0.00700604 |
| simple_combined_confidence | confidence | 480 | 0.365698 | 6.90189 | 0.0156391 |
| random_score | confidence | 480 | -0.0116285 | 0.382258 | 0.00116577 |
| context_similarity_score | confidence | 480 | -0.0140814 | 0.537559 | 0.000632898 |
| perturbation_stability_score | confidence | 220 | -0.153238 | -0.120076 | -0.00329546 |
| historical_residual_risk | risk | 480 | -0.292099 | -3.52917 | -0.00382839 |
| prediction_magnitude_risk | risk | 480 | -0.292755 | 0.128309 | -0.00126586 |
| ood_distance_risk | risk | 480 | -0.366869 | -3.96203 | -0.0108359 |

## 6. 必须保留的边界

- 本轮为 focused top-80 shared drug-dose panel；743 shared drug-dose 全量版本留待优化后复跑。
- 全量版本第一次尝试在高维最近邻 feature 阶段过慢，已中断；后续需要优化 `compute_features` 后再跑。
- `prediction_magnitude_risk` 在该 panel 上方向为负：aligned Spearman = -0.293。
- `perturbation_effect_stability` 缺失率为 55.6%，组合分数受该特征覆盖影响。
- 当前仍使用两个轻量参考预测器；GEARS/CPA/scGPT 多模型逐向量验证尚未完成。

## 7. 下一步

1. 优化 `compute_features` 中 context cosine / OOD nearest distance 的矩阵化实现。
2. 复跑官方 sciplex3 全 743 shared drug-dose panel。
3. 把 GEARS/CPA/scGPT 的同任务预测向量接入该三细胞系 panel，比较 task-only、model-only、task+model combined。
