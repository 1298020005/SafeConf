# E59｜跨数据集分数构成审计

## 问题

老师问到效应幅度时，必须分清两件事：`predicted_l2_combined` 是只由源数据构造的预测向量长度，可在打分时得到；`true_l2_diagnostic` 来自目标真实效应，只作 oracle 诊断，完全不进入 `risk_cross_dataset`。

跨数据集总分由四项相加：低历史支持、低控制状态相似度、两个源域参考预测器的分歧、预测效应幅度。本轮比较总分与单独预测幅度，检查组合项有没有实际贡献。

- 输入任务行：4,430
- 方向数：50
- bootstrap 次数：2000

## 任务数 ≥ 30 的比较

| directional_pair | n_tasks | rho_full | rho_predicted_magnitude | rho_without_predicted_magnitude | delta_full_minus_predicted_magnitude | bootstrap_delta_full_minus_predicted_magnitude_ci95_low | bootstrap_delta_full_minus_predicted_magnitude_ci95_high | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KaggleCrossCell_celltype -> KaggleCrossPatient_celltype | 38 | -0.163 | -0.446 | -0.041 | 0.283 | -0.024 | 0.598 | 总分与预测幅度差异不明确 |
| Lara_exvivo_celltype -> Lara_leukemia_celltype | 147 | 0.159 | -0.084 | 0.214 | 0.243 | 0.044 | 0.455 | 总分优于预测幅度：组合特征有额外贡献 |
| crossPatient_patient -> McFarland_cellline | 42 | 0.252 | 0.218 | 0.252 | 0.033 | -0.136 | 0.197 | 总分与预测幅度差异不明确 |
| KaggleCrossCell_celltype -> KaggleCrossPatient_donor | 30 | 0.861 | 0.828 | 0.770 | 0.033 | -0.103 | 0.171 | 总分与预测幅度差异不明确 |
| TianInhibition_batch -> Adamson_global | 76 | 0.297 | 0.297 | 0.103 | 0.000 | 0.000 | 0.000 | 总分与预测幅度差异不明确 |
| KaggleCrossCell_celltype -> McFarland_cellline | 42 | 0.602 | 0.603 | 0.550 | -0.001 | -0.213 | 0.240 | 总分与预测幅度差异不明确 |
| KaggleCrossPatient_celltype -> McFarland_cellline | 42 | 0.002 | 0.020 | 0.090 | -0.017 | -0.202 | 0.146 | 总分与预测幅度差异不明确 |
| Replogle_exp7_global -> Replogle_exp8_global | 61 | 0.128 | 0.148 | -0.111 | -0.019 | -0.382 | 0.361 | 总分与预测幅度差异不明确 |
| sciplex3_cellline -> McFarland_cellline | 42 | 0.000 | 0.055 | -0.042 | -0.055 | -0.288 | 0.173 | 总分与预测幅度差异不明确 |
| crossPatient_patient -> KaggleCrossPatient_celltype | 38 | 0.107 | 0.185 | 0.055 | -0.078 | -0.331 | 0.174 | 总分与预测幅度差异不明确 |
| Replogle_exp7_global -> Adamson_global | 76 | 0.080 | 0.180 | -0.171 | -0.100 | -0.397 | 0.153 | 总分与预测幅度差异不明确 |
| sciplex3_cellline -> KaggleCrossPatient_celltype | 38 | 0.099 | 0.202 | -0.138 | -0.103 | -0.372 | 0.182 | 总分与预测幅度差异不明确 |
| Lara_exvivo_celltype -> Lara_invivo_celltype | 249 | -0.059 | 0.084 | -0.109 | -0.143 | -0.247 | -0.036 | 预测幅度优于总分：组合项未带来增益 |
| Replogle_exp7_global -> Replogle_exp6_global | 69 | 0.071 | 0.246 | -0.241 | -0.175 | -0.540 | 0.157 | 总分与预测幅度差异不明确 |
| TianActivation_batch -> TianInhibition_batch | 705 | 0.025 | 0.225 | -0.048 | -0.200 | -0.287 | -0.108 | 预测幅度优于总分：组合项未带来增益 |
| Replogle_exp6_global -> Replogle_exp8_global | 61 | 0.146 | 0.354 | -0.353 | -0.208 | -0.588 | 0.000 | 总分与预测幅度差异不明确 |
| TianInhibition_batch -> TianActivation_batch | 184 | 0.108 | 0.320 | -0.044 | -0.212 | -0.356 | -0.059 | 预测幅度优于总分：组合项未带来增益 |
| Adamson_global -> Replogle_exp7_global | 104 | 0.120 | 0.348 | -0.148 | -0.228 | -0.523 | 0.000 | 总分与预测幅度差异不明确 |
| Adamson_global -> TianInhibition_batch | 705 | -0.057 | 0.185 | -0.121 | -0.242 | -0.324 | -0.147 | 预测幅度优于总分：组合项未带来增益 |
| Lara_invivo_celltype -> Lara_leukemia_celltype | 147 | 0.205 | 0.472 | 0.003 | -0.267 | -0.395 | -0.144 | 预测幅度优于总分：组合项未带来增益 |
| Replogle_exp6_global -> Replogle_exp7_global | 104 | 0.092 | 0.372 | -0.259 | -0.281 | -0.545 | 0.000 | 总分与预测幅度差异不明确 |
| Lara_leukemia_celltype -> Lara_exvivo_celltype | 288 | 0.547 | 0.886 | 0.283 | -0.338 | -0.418 | -0.260 | 预测幅度优于总分：组合项未带来增益 |
| Replogle_exp8_global -> Replogle_exp6_global | 69 | -0.123 | 0.238 | -0.239 | -0.362 | -0.698 | 0.000 | 总分与预测幅度差异不明确 |
| Lara_leukemia_celltype -> Lara_invivo_celltype | 249 | 0.538 | 0.902 | 0.305 | -0.364 | -0.461 | -0.281 | 预测幅度优于总分：组合项未带来增益 |
| Lara_invivo_celltype -> Lara_exvivo_celltype | 288 | 0.384 | 0.770 | -0.008 | -0.387 | -0.476 | -0.298 | 预测幅度优于总分：组合项未带来增益 |
| Replogle_exp8_global -> Replogle_exp7_global | 104 | -0.200 | 0.371 | -0.294 | -0.571 | -0.836 | -0.275 | 预测幅度优于总分：组合项未带来增益 |
| McFarland_cellline -> KaggleCrossPatient_celltype | 38 | -0.209 | nan | -0.209 | nan | nan | nan | 总分与预测幅度差异不明确 |

## 如何使用这张表

- 总分显著优于预测幅度的方向数：1。
- 预测幅度显著优于总分的方向数：9。
- 差异不明确的方向数：17。

这张表不支持把“预测幅度”说成 SafeConf 的独立贡献。若某个方向预测幅度更强，就按结果原样保留；若总分有额外贡献，才说明支持度、上下文和分歧提供了幅度以外的信息。

## 文件

- 图：`figures/F1_full_risk_vs_predicted_magnitude.svg`
- 各成分相关：`tables/E59_FEATURE_SCORE_SUMMARY.csv`
- 总分对幅度的 bootstrap 差值：`tables/E59_FULL_VS_MAGNITUDE_COMPARISON.csv`
