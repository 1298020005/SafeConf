# Frozen-protocol blind dataset report

## 主结论表（per-dataset，pooled 只作辅助）

| dataset_name | simple_combined_confidence_aligned_rho | simple_combined_confidence_risk_cov_80_improve_pct | learned_risk_score_aligned_rho | learned_risk_score_risk_cov_80_improve_pct | historical_residual_risk_aligned_rho | historical_residual_risk_risk_cov_80_improve_pct | model_disagreement_risk_aligned_rho | model_disagreement_risk_risk_cov_80_improve_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sciplex3_official_3cell | 0.709772 | 16.7972 | 0.898593 | 17.5568 | 0.100594 | 3.70941 | 0.739184 | 17.0648 |

## 本阶段修了什么

- 用 `n_genes=5000, min_cells=6, max_cells_per_group=2200, seed=5201` 重建三数据集任务。
- PredictionRecord 包含 train/val/test，但所有正式 evaluation 只用 test。
- `perturbation_effect_stability` 保留原始 NaN，不再用全局 median 洗成非缺失。
- 新增 `historical_residual_risk`：train 内 leave-one-context-out 残差，按 perturbation 聚合。
- `simple_combined_confidence` 用同 fold train 作 z-score 参考，权重只根据同 fold val 的 aligned Spearman 网格搜索。
- `learned_risk_score` 改为同 dataset + 同 fold + 同 predictor 的 train+val 训练 HistGradientBoosting，只给 test 打分。

## 数据与防泄漏

| path | dataset | n_tasks | n_contexts | n_perturbations | context_col | perturbation_col | n_genes | source_files | n_raw_tasks | n_shared_perturbations | n_selected_perturbations | selection_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /home/yyf/data/singlecell_perturbation_atlas/official_generalization/sciplex3... | sciplex3_official_3cell | 2229 | 3 | 743 | cell_line | drug_dose_name | 2000 | sciplex3_A549.h5ad;sciplex3_K562.h5ad;sciplex3_MCF7.h5ad | 2237 | 743 | 743 | top 743 shared perturbations by min_cells, total_cells |

- Test records: 4458
- Split leakage counts: {"pair_leak": 0, "pert_missing": 0, "context_missing": 0}

## Feature missingness by dataset

| dataset_name | perturbation_effect_stability | historical_residual_risk | ood_nearest_distance |
| --- | --- | --- | --- |
| sciplex3_official_3cell | 0.552266 | 0 | 0 |

## 口径

这次不把 pooled learned 分数当标题。主方法口径看每个数据集上的 `simple_combined_confidence`；learned 只是辅助/ablation。

## combined regression debug

- Phase 1 KCC combined 是单数据集 min-max 组合，且 KCC 任务较小。
- Phase 2 v2 改成跨三数据集 pooled z-score/median fallback，并把 stability fallback 填满，导致 KCC/Haber combined 方向塌掉。
- Phase 2.1 改回 fold-local 参考分布和 val-tuned fixed-sign weights；如果 G7/G8 仍未过，见 `reports/combined_regression_debug.md`。
