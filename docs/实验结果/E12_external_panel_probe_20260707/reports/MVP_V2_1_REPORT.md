# Frozen-protocol blind dataset report

## 主结论表（per-dataset，pooled 只作辅助）

| dataset_name | simple_combined_confidence_aligned_rho | simple_combined_confidence_risk_cov_80_improve_pct | learned_risk_score_aligned_rho | learned_risk_score_risk_cov_80_improve_pct | historical_residual_risk_aligned_rho | historical_residual_risk_risk_cov_80_improve_pct | model_disagreement_risk_aligned_rho | model_disagreement_risk_risk_cov_80_improve_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KaggleCrossPatient | 0.618747 | 14.3832 | 0.00923478 | -1.9435 | -0.0204212 | -2.00744 | 0.468595 | 12.0946 |
| crossPatient | 0.0753331 | 2.44117 | 0 | 0 | -0.435464 | -3.26371 | 0.597644 | -0.167093 |
| sciplex3 | 0.622404 | 17.6907 | 0.425516 | 7.80326 | 0.451349 | 2.84309 | 0.884577 | 18.1931 |

## 本阶段修了什么

- 用 `n_genes=5000, min_cells=6, max_cells_per_group=2200, seed=5201` 重建三数据集任务。
- PredictionRecord 包含 train/val/test，但所有正式 evaluation 只用 test。
- `perturbation_effect_stability` 保留原始 NaN，不再用全局 median 洗成非缺失。
- 新增 `historical_residual_risk`：train 内 leave-one-context-out 残差，按 perturbation 聚合。
- `simple_combined_confidence` 用同 fold train 作 z-score 参考，权重只根据同 fold val 的 aligned Spearman 网格搜索。
- `learned_risk_score` 改为同 dataset + 同 fold + 同 predictor 的 train+val 训练 HistGradientBoosting，只给 test 打分。

## 数据与防泄漏

| path | dataset | n_tasks | n_contexts | n_perturbations | context_col | perturbation_col | n_genes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_cont... | kangCrossCell | 8 | 8 | 1 | cell_type | perturbation | 5000 |
| /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_cont... | kangCrossPatient | 8 | 8 | 1 | condition1 | perturbation | 5000 |
| /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_cont... | KaggleCrossPatient | 40 | 6 | 10 | cell_type | perturbation | 5000 |
| /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_cont... | crossPatient | 10 | 6 | 2 | condition1 | perturbation | 5000 |
| /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_cont... | TCDD | 6 | 6 | 1 | condition1 | perturbation | 5000 |
| /home/yyf/datasets/singlecell_perturbation_atlas/extra_official/cellular_cont... | sciplex3 | 106 | 3 | 36 | cell_type | perturbation | 5000 |

- Test records: 308
- Split leakage counts: {"pair_leak": 0, "pert_missing": 0, "context_missing": 0}

## Feature missingness by dataset

| dataset_name | perturbation_effect_stability | historical_residual_risk | ood_nearest_distance |
| --- | --- | --- | --- |
| KaggleCrossPatient | 0.26 | 0 | 0 |
| TCDD | 0 | 0 | 0 |
| crossPatient | 0.133333 | 0 | 0 |
| kangCrossCell | 0 | 0 | 0 |
| kangCrossPatient | 0 | 0 | 0 |
| sciplex3 | 0.571698 | 0 | 0 |

## 口径

这次不把 pooled learned 分数当标题。主方法口径看每个数据集上的 `simple_combined_confidence`；learned 只是辅助/ablation。

## combined regression debug

- Phase 1 KCC combined 是单数据集 min-max 组合，且 KCC 任务较小。
- Phase 2 v2 改成跨三数据集 pooled z-score/median fallback，并把 stability fallback 填满，导致 KCC/Haber combined 方向塌掉。
- Phase 2.1 改回 fold-local 参考分布和 val-tuned fixed-sign weights；如果 G7/G8 仍未过，见 `reports/combined_regression_debug.md`。
