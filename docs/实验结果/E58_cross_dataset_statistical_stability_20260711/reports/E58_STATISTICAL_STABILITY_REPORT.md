# E58｜跨数据集风险排序：统计稳定性审计

## 这轮在回答什么

E55/E57 每个目标任务先用源数据集构造 `risk_cross_dataset`，随后才读取该任务真实效应并计算 `error_combined_rmse`。本轮固定这两个量，逐方向做 bootstrap 与置换检验，防止小任务数方向被过度解读。

- 输入任务行数：4,430
- 方向数：50
- bootstrap 次数：2000
- 置换次数：2000
- 统计图：`figures/F1_cross_dataset_risk_error_bootstrap_ci.svg`

## 可以放进主汇报的稳定正信号

| directional_pair | n_tasks | spearman_risk_vs_error | bootstrap_rho_ci95_low | bootstrap_rho_ci95_high | permutation_p_two_sided | top20_error_enrichment |
| --- | --- | --- | --- | --- | --- | --- |
| KaggleCrossCell_celltype -> KaggleCrossPatient_donor | 30 | 0.861 | 0.715 | 0.933 | 0.000 | 1.610 |
| KaggleCrossCell_celltype -> McFarland_cellline | 42 | 0.602 | 0.342 | 0.779 | 0.000 | 1.372 |
| Lara_leukemia_celltype -> Lara_exvivo_celltype | 288 | 0.547 | 0.460 | 0.622 | 0.000 | 1.888 |
| Lara_leukemia_celltype -> Lara_invivo_celltype | 249 | 0.538 | 0.434 | 0.626 | 0.000 | 1.862 |
| Lara_invivo_celltype -> Lara_exvivo_celltype | 288 | 0.384 | 0.280 | 0.480 | 0.000 | 1.526 |
| TianInhibition_batch -> Adamson_global | 76 | 0.297 | 0.058 | 0.476 | 0.007 | 1.212 |
| Lara_invivo_celltype -> Lara_leukemia_celltype | 147 | 0.205 | 0.018 | 0.382 | 0.014 | 1.399 |

这些方向同时满足：任务数至少 30、bootstrap 95% CI 下界大于 0、双侧置换 p < 0.05。

## 必须保留的边界

（无行）

## 探索性方向

| directional_pair | n_tasks | spearman_risk_vs_error | bootstrap_rho_ci95_low | bootstrap_rho_ci95_high | reporting_label |
| --- | --- | --- | --- | --- | --- |
| KaggleCrossCell_celltype -> crossPatient_patient | 10 | -0.153 | -0.794 | 0.707 | 探索性：样本量较小，需独立复现 |
| KaggleCrossPatient_celltype -> crossPatient_patient | 10 | -0.227 | -0.840 | 0.542 | 探索性：样本量较小，需独立复现 |
| McFarland_cellline -> crossPatient_patient | 10 | 0.454 | -0.319 | 0.873 | 探索性：样本量较小，需独立复现 |
| sciplex3_cellline -> crossPatient_patient | 10 | -0.785 | -0.981 | -0.231 | 探索性：样本量较小，需独立复现 |
| Dixit_13d_target -> Dixit_7d_target | 10 | 0.588 | -0.245 | 0.988 | 探索性：样本量较小，需独立复现 |
| Dixit_7d_target -> Dixit_13d_target | 10 | -0.127 | -0.824 | 0.640 | 探索性：样本量较小，需独立复现 |
| SciPlex2_cellline -> SciPlex4_cellline | 14 | -0.656 | -0.877 | -0.160 | 探索性：样本量较小，需独立复现 |
| sciplex3_small_cellline -> SciPlex4_cellline | 14 | -0.124 | -0.674 | 0.442 | 探索性：样本量较小，需独立复现 |
| sciplex3_cellline -> KaggleCrossCell_celltype | 24 | 0.225 | -0.187 | 0.589 | 探索性：样本量较小，需独立复现 |
| KaggleCrossPatient_celltype -> KaggleCrossCell_celltype | 24 | 0.577 | 0.202 | 0.811 | 探索性：样本量较小，需独立复现 |
| KaggleCrossPatient_donor -> KaggleCrossCell_celltype | 24 | 0.588 | 0.189 | 0.832 | 探索性：样本量较小，需独立复现 |
| crossPatient_patient -> KaggleCrossCell_celltype | 24 | 0.098 | -0.298 | 0.484 | 探索性：样本量较小，需独立复现 |
| McFarland_cellline -> KaggleCrossCell_celltype | 24 | -0.018 | -0.417 | 0.380 | 探索性：样本量较小，需独立复现 |
| McFarland_cellline -> sciplex3_cellline | 27 | -0.058 | -0.392 | 0.297 | 探索性：样本量较小，需独立复现 |
| KaggleCrossCell_celltype -> sciplex3_cellline | 27 | 0.297 | -0.153 | 0.655 | 探索性：样本量较小，需独立复现 |
| KaggleCrossPatient_celltype -> sciplex3_cellline | 27 | 0.303 | -0.041 | 0.580 | 探索性：样本量较小，需独立复现 |
| crossPatient_patient -> sciplex3_cellline | 27 | -0.448 | -0.764 | -0.029 | 探索性：样本量较小，需独立复现 |
| SciPlex2_cellline -> sciplex3_small_cellline | 27 | -0.349 | -0.661 | 0.021 | 探索性：样本量较小，需独立复现 |
| SciPlex4_cellline -> sciplex3_small_cellline | 27 | -0.303 | -0.661 | 0.140 | 探索性：样本量较小，需独立复现 |

## 汇报口径

可以直接说：分数与固定参考预测器的 `error_combined_rmse` 对照；打分输入不含目标真实效应。跨数据集结果存在稳定正方向，也存在不稳定和负方向，因此结论是风险排序受源—目标相似性、覆盖度和任务数影响，不能把一个方向的好结果外推到所有场景。
