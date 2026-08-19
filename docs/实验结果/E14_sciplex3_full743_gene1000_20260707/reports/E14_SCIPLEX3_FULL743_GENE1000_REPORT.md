# E14 官方 sciplex3 full-743 gene1000 panel

生成时间：2026-07-07 15:21

## 1. 结论

E14 使用官方 `sciplex3_A549/K562/MCF7` 三细胞系面板，保留全部 743 个三细胞系共享 drug-dose。为让全量结构先跑通，本轮将基因面板降为 1,000 genes。因此它是 full perturbation / low-gene 快速验证；5,000-gene 全量正式版仍需单独复跑。

面板规模：2,229 tasks、3 contexts、743 perturbations、4,458 test records。split 审计通过，无 pair/context/perturbation leakage。

最强信号为 `learned_risk_score`：aligned Spearman = 0.862，80% coverage RMSE 改善 = 13.63%。`model_disagreement_risk` aligned Spearman = 0.418，80% coverage 改善 = 12.13%；`simple_combined_confidence` aligned Spearman = 0.339。

## 2. 面板构建

| dataset | n_tasks | n_contexts | n_perturbations | context_col | perturbation_col | n_genes | n_raw_tasks | n_shared_perturbations | n_selected_perturbations | selection_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 2229 | 3 | 743 | cell_line | drug_dose_name | 1000 | 2237 | 743 | 743 | top 743 shared perturbations by min_cells, total_cells |

## 3. Split 审计

| dataset_name | fold_id | n_tasks_total | n_candidate_test_pairs | n_train | n_val | n_test | support_check_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 0 | 2229 | 2229 | 1516 | 267 | 446 | True |
| sciplex3_official_3cell | 1 | 2229 | 2229 | 1516 | 268 | 445 | True |
| sciplex3_official_3cell | 2 | 2229 | 2229 | 1516 | 267 | 446 | True |
| sciplex3_official_3cell | 3 | 2229 | 2229 | 1516 | 267 | 446 | True |
| sciplex3_official_3cell | 4 | 2229 | 2229 | 1516 | 267 | 446 | True |

## 4. 主结果

| dataset_name | simple_combined_confidence_aligned_rho | simple_combined_confidence_risk_cov_80_improve_pct | learned_risk_score_aligned_rho | learned_risk_score_risk_cov_80_improve_pct | model_disagreement_risk_aligned_rho | model_disagreement_risk_risk_cov_80_improve_pct |
| --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 0.338739 | 8.93489 | 0.862259 | 13.6263 | 0.41811 | 12.1345 |

## 5. Overall 信号排序

| score_name | score_type | n | direction_aligned_spearman | risk_cov_80_improve_pct | high_low_rmse_gap |
| --- | --- | --- | --- | --- | --- |
| learned_risk_score | risk | 4458 | 0.862259 | 13.6263 | 0.0507136 |
| model_disagreement_risk | risk | 4458 | 0.41811 | 12.1345 | 0.0400399 |
| simple_combined_confidence | confidence | 4458 | 0.338739 | 8.93489 | 0.028387 |
| support_count_score | confidence | 4458 | 0.272183 | 1.79001 | 0.00812137 |
| context_similarity_score | confidence | 4458 | 0.0912269 | -0.395061 | 0.00337056 |
| historical_residual_risk | risk | 4458 | 0.0754414 | 3.03148 | 0.0158607 |
| random_score | confidence | 4458 | -0.0203764 | -0.384222 | -0.00119101 |
| prediction_magnitude_risk | risk | 4458 | -0.0638103 | 6.91766 | 0.0177875 |
| ood_distance_risk | risk | 4458 | -0.345484 | -3.03869 | -0.0135709 |
| perturbation_stability_score | confidence | 2062 | -0.429849 | -2.99793 | -0.0303272 |

## 6. 必须保留的边界

- E14 是 1,000-gene full-743 快速验证；5,000-gene full-743 仍未完成。
- 旧 gate G10 按 5,000 genes 检查，因此在本轮必然 FAIL。
- learned risk 很强，但这是同一轻量参考预测体系内的 fold-safe learned score；真实 GEARS/CPA/scGPT 逐模型向量验证尚未完成。
- `perturbation_effect_stability` 缺失率约 55.3%，simple combined 仍受覆盖影响。
- `prediction_magnitude_risk` aligned Spearman = -0.064，方向仍弱。

## 7. 下一步

1. 对 `compute_features` 继续做 task-fold 级缓存，减少重复计算。
2. 复跑 5,000-gene full-743，或先跑 2,000/3,000 gene sensitivity。
3. 用 E14 面板作为真实多模型接入的外部 chemical 任务基准。
