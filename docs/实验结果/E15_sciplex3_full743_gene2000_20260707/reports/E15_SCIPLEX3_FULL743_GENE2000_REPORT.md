# E15 官方 sciplex3 full-743 gene2000 sensitivity

生成时间：2026-07-07 21:09

## 1. 结论

E15 复用官方 `sciplex3_A549/K562/MCF7` 三细胞系面板，保留全部 743 个三细胞系共享 drug-dose。本轮基因面板为 2,000 genes，用于检查 full-743 chemical 外部验证信号随基因数增加是否稳定。

面板规模：2,229 tasks、3 contexts、743 perturbations、4,458 test records。split 审计通过，无 pair/context/perturbation leakage。

最强信号为 `learned_risk_score`：aligned Spearman = 0.899，80% coverage RMSE 改善 = 17.56%。`model_disagreement_risk` aligned Spearman = 0.739，80% coverage 改善 = 17.06%；`simple_combined_confidence` aligned Spearman = 0.710。

## 2. 面板构建

| dataset | n_tasks | n_contexts | n_perturbations | context_col | perturbation_col | n_genes | n_raw_tasks | n_shared_perturbations | n_selected_perturbations | selection_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 2229 | 3 | 743 | cell_line | drug_dose_name | 2000 | 2237 | 743 | 743 | top 743 shared perturbations by min_cells, total_cells |

## 3. Split 审计

| dataset_name | fold_id | n_tasks_total | n_candidate_test_pairs | n_train | n_val | n_test | support_check_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 0 | 2229 | 2229 | 1516 | 267 | 446 | True |
| sciplex3_official_3cell | 1 | 2229 | 2229 | 1516 | 267 | 446 | True |
| sciplex3_official_3cell | 2 | 2229 | 2229 | 1516 | 268 | 445 | True |
| sciplex3_official_3cell | 3 | 2229 | 2229 | 1516 | 267 | 446 | True |
| sciplex3_official_3cell | 4 | 2229 | 2229 | 1516 | 267 | 446 | True |

## 4. 主结果

| dataset_name | simple_combined_confidence_aligned_rho | simple_combined_confidence_risk_cov_80_improve_pct | learned_risk_score_aligned_rho | learned_risk_score_risk_cov_80_improve_pct | model_disagreement_risk_aligned_rho | model_disagreement_risk_risk_cov_80_improve_pct |
| --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 0.709772 | 16.7972 | 0.898593 | 17.5568 | 0.739184 | 17.0648 |

## 5. Overall 信号排序

| score_name | score_type | n | direction_aligned_spearman | risk_cov_80_improve_pct | high_low_rmse_gap |
| --- | --- | --- | --- | --- | --- |
| learned_risk_score | risk | 4458 | 0.898593 | 17.5568 | 0.071087 |
| model_disagreement_risk | risk | 4458 | 0.739184 | 17.0648 | 0.0646876 |
| simple_combined_confidence | confidence | 4458 | 0.709772 | 16.7972 | 0.0634421 |
| support_count_score | confidence | 4458 | 0.257402 | 1.59388 | 0.0100968 |
| historical_residual_risk | risk | 4458 | 0.100594 | 3.70941 | 0.0226198 |
| random_score | confidence | 4458 | 0.00291867 | 0.130788 | 0.000597826 |
| context_similarity_score | confidence | 4458 | -0.0290727 | 0.00120614 | 0.000655069 |
| prediction_magnitude_risk | risk | 4458 | -0.0397301 | 8.82209 | 0.0261021 |
| ood_distance_risk | risk | 4458 | -0.1984 | -3.16287 | -0.0109644 |
| perturbation_stability_score | confidence | 2072 | -0.504437 | -5.07505 | -0.0466467 |

## 6. 连续性判断

- 上一轮：E14 gene1000，learned aligned Spearman = 0.862，80% coverage 改善 = 13.63%
- 本轮：gene2000，learned aligned Spearman = 0.899，80% coverage 改善 = 17.56%。
- 当前判断：风险信号在 full-743 chemical 面板上保持稳定。

## 7. 边界

- 本轮仍属于 lightweight reference prediction system 内的风险评估。
- 真实 GEARS/CPA/scGPT 逐模型向量验证仍是单独任务。
- `prediction_magnitude_risk` aligned Spearman = -0.040。

## 8. 下一步

继续运行 3,000/5,000-gene sensitivity；同时开始 GEARS/CPA/scGPT 逐模型预测向量对齐审计。
