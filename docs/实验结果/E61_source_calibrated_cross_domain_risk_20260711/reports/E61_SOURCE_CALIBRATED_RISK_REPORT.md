# E61｜源域校准的跨数据集风险审计

## 设计

固定等权总分在 E59 中经常不如预测幅度。E61 不在目标域选择权重：每个源数据集先按任务切成 5 折，轮流留一折，得到源域内部“模拟未见任务”的特征和误差；用这些源域 OOF 行拟合 ridge calibrator。随后用全部源数据构建目标任务的四个输入特征，直接输出预测风险。目标真实 effect 只在最后计算 `error_combined_rmse`。

特征为：`log1p_source_support`、`nearest_context_similarity`、`prediction_disagreement_rmse`、`predicted_l2_combined`。系数在 `tables/E61_CALIBRATOR_COEFFICIENTS.csv` 中逐方向保存。

## 结果

| directional_pair | source_n_tasks | target_n_tasks | spearman_source_calibrated | spearman_predicted_magnitude | delta_calibrated_minus_magnitude | bootstrap_delta_ci95_low | bootstrap_delta_ci95_high | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lara_exvivo_celltype -> Lara_leukemia_celltype | 288 | 147 | 0.333 | -0.053 | 0.386 | 0.168 | 0.605 | 校准总分优于预测幅度 |
| Lara_exvivo_celltype -> Lara_invivo_celltype | 288 | 249 | 0.111 | 0.032 | 0.079 | -0.034 | 0.202 | 差异不明确 |
| TianActivation_batch -> TianInhibition_batch | 184 | 705 | 0.194 | 0.225 | -0.031 | -0.087 | 0.023 | 差异不明确 |
| KaggleCrossCell_celltype -> McFarland_cellline | 24 | 42 | 0.596 | 0.741 | -0.145 | -0.324 | -0.024 | 预测幅度优于校准总分 |
| Lara_invivo_celltype -> Lara_leukemia_celltype | 249 | 147 | 0.330 | 0.504 | -0.174 | -0.277 | -0.062 | 预测幅度优于校准总分 |
| Lara_invivo_celltype -> Lara_exvivo_celltype | 249 | 288 | 0.576 | 0.773 | -0.198 | -0.258 | -0.140 | 预测幅度优于校准总分 |
| KaggleCrossCell_celltype -> KaggleCrossPatient_donor | 24 | 30 | 0.581 | 0.812 | -0.231 | -0.551 | 0.053 | 差异不明确 |
| TianInhibition_batch -> TianActivation_batch | 705 | 184 | -0.027 | 0.320 | -0.346 | -0.532 | -0.155 | 预测幅度优于校准总分 |
| Lara_leukemia_celltype -> Lara_invivo_celltype | 147 | 249 | 0.458 | 0.865 | -0.407 | -0.508 | -0.304 | 预测幅度优于校准总分 |
| Lara_leukemia_celltype -> Lara_exvivo_celltype | 147 | 288 | 0.237 | 0.900 | -0.663 | -0.776 | -0.556 | 预测幅度优于校准总分 |

## 口径

这个实验只回答一件事：源域可见任务上学到的四项权重，换到目标数据集后，能否比单独的预测幅度更好地排序高误差任务。它不是用目标真值重新调参，也不把 oracle true magnitude 当作输入。若结果仍不稳定，就说明这四个特征的跨域可迁移性有限，应该如实作为方法边界。

## 文件

- 目标任务分数：`tables/E61_TARGET_TASK_SCORES.csv`
- 源域 OOF 校准行：`tables/E61_SOURCE_OOF_CALIBRATION_ROWS.csv`
- 各方向系数：`tables/E61_CALIBRATOR_COEFFICIENTS.csv`
- 汇总：`tables/E61_SUMMARY.csv`
- 图：`figures/F1_calibrated_vs_magnitude.svg`
