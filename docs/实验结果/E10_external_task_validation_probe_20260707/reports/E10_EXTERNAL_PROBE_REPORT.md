# E10 外部任务级验证探针报告

生成时间：2026-07-07 06:26

## 1. 一句话结论

E10 已经从外部 context-generalization 数据中跑通一个小型任务级探针：`KaggleCrossCell`、`Haber`、`Parekh` 共 78 个任务、77 个测试任务，split 支持检查通过。最强信号来自 `model_disagreement_risk`，overall aligned Spearman = 0.559，80% coverage 下平均 RMSE 改善 7.96%。

这不是最终可投稿的外部验证结论。它更像“侦察兵”：证明管线能跑，也暴露出 frozen/simple combined 在 KaggleCrossCell 上较弱、learned risk 未超过 disagreement 的问题。

## 2. 数据与任务

| dataset | n_tasks | n_contexts | n_perturbations | context_col | perturbation_col | n_genes |
| --- | --- | --- | --- | --- | --- | --- |
| KaggleCrossCell | 24 | 5 | 10 | cell_type | perturbation | 5000 |
| Haber | 24 | 8 | 3 | cell_type | perturbation | 5000 |
| Parekh | 30 | 3 | 10 | cell_type | perturbation | 5000 |

## 3. 关键结果

### 3.1 每个数据集的主要分数

| dataset_name | simple_combined_confidence_aligned_rho | simple_combined_confidence_risk_cov_80_improve_pct | learned_risk_score_aligned_rho | learned_risk_score_risk_cov_80_improve_pct | model_disagreement_risk_aligned_rho | model_disagreement_risk_risk_cov_80_improve_pct |
| --- | --- | --- | --- | --- | --- | --- |
| KaggleCrossCell | 0.1208231819745679 | -0.5532018705208882 | 0.0313036310132466 | -4.456773353278907 | 0.3534350276855193 | 12.856780053226494 |
| Haber | 0.6106029118945587 | 5.099050536667971 | 0.008672516405563 | 1.2907924382281557 | -0.0219422162613999 | -1.6914971084512649 |
| Parekh | 0.4323276495117311 | 6.472721022629708 | -0.0996856352160196 | -6.062513682066115 | 0.8064382146140299 | 12.723001559770108 |

### 3.2 每个数据集当前最强单项信号

| dataset_name | score_name | score_type | n | direction_aligned_spearman | risk_cov_80_improve_pct | high_low_rmse_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Haber | context_similarity_score | confidence | 48 | 0.681731595308874 | 3.6358143497474167 | 0.0313305750600737 |
| KaggleCrossCell | model_disagreement_risk | risk | 46 | 0.3534350276855193 | 12.856780053226494 | 0.0531448277685609 |
| Parekh | model_disagreement_risk | risk | 60 | 0.8064382146140299 | 12.723001559770108 | 0.0383382076836481 |

## 4. 失败边界

- `KaggleCrossCell` 上 simple combined aligned rho = 0.121，80% coverage 改善为 -0.55%，不够稳。
- learned risk overall 未超过 model disagreement，说明现有轻量学习器暂时不是 E10 的主角。
- `perturbation_effect_stability` 在 KaggleCrossCell 缺失率 82.50%，在 Parekh 缺失率 56.67%，解释了组合分数在部分外部数据上不够稳定。

## 5. 图件

- `figures/F1_task_schematic.png`：任务定义
- `figures/F2_context_perturbation_matrix.png`：context × perturbation 覆盖
- `figures/F3_prediction_record_flow.png`：PredictionRecord 流程
- `figures/F4_confidence_vs_true_error_scatter.png`：分数与真实误差散点
- `figures/F5_risk_coverage_curve.png`：risk-coverage 曲线
- `figures/F7_per_dataset_spearman_comparison.png`：每数据集 Spearman 对比
- `figures/F10_transferability_ranking.png`：外部迁移排序

## 6. 下一步判断

1. E10 下一轮不急着扩成大模型训练，先固定一个清晰外部 split，比较 task-only、disagreement、magnitude、task+model combined。
2. 若目标是一区/CCF-A，E10 需要变成“跨数据源的 selective prediction / risk routing”证据，而不是只证明某个组合分数 Spearman 为正。
3. 对 KaggleCrossCell 需要单独查失败原因：context 相似性方向反了，stability 大量缺失，说明外部数据字段和主表协议不完全同构。

## 7. 复现

本次运行原始目录：

```text
runtime/e10_external_probe_kcc_haber_parekh_20260707
```

原始脚本：

```text
code/20260426_154505_perturb_transport_final_push/confidence_task/run_confidence_mvp_v2_1.py
```
