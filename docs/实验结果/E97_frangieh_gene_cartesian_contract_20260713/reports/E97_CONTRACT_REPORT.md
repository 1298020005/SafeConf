# E97｜Frangieh 遗传扰动三行矩阵冻结合同

Frangieh 原始数据包含 3 个细胞背景和 211 个单基因扰动。按每个“背景×扰动”至少 50 个细胞筛选后，得到完整的 3×189 矩阵。任务只根据标签、细胞数和哈希顺序冻结，没有读取表达矩阵、效应、预测或误差。

每个 fold 留出一个完整背景。30 个扰动同时作为整列未见基因，其余扰动在源背景可见；因此同一合同同时包含整行新背景、整列新扰动、背景与扰动双未见、随机缺失 pair。训练任务再冻结 25%、50%、75%、100% 四个嵌套子矩阵。

## 任务规模

| fold_id | setting | split | n_tasks | min_cells | median_cells | max_cells |
|---|---|---|---|---|---|---|
| frangieh_context_holdout_1_Co-culture | context_and_perturbation_unseen | test | 30 | 86 | 194.0 | 418 |
| frangieh_context_holdout_1_Co-culture | context_unseen_row | test | 159 | 58 | 203.0 | 475 |
| frangieh_context_holdout_1_Co-culture | perturbation_unseen_column | test | 60 | 57 | 174.5 | 325 |
| frangieh_context_holdout_1_Co-culture | random_missing_pair | test | 30 | 105 | 220.0 | 341 |
| frangieh_context_holdout_1_Co-culture | source_train_pair | train | 258 | 50 | 163.0 | 309 |
| frangieh_context_holdout_1_Co-culture | source_validation_pair | val | 30 | 69 | 177.0 | 311 |
| frangieh_context_holdout_2_Control | context_and_perturbation_unseen | test | 30 | 62 | 140.0 | 186 |
| frangieh_context_holdout_2_Control | context_unseen_row | test | 159 | 50 | 142.0 | 190 |
| frangieh_context_holdout_2_Control | perturbation_unseen_column | test | 60 | 83 | 225.0 | 396 |
| frangieh_context_holdout_2_Control | random_missing_pair | test | 30 | 65 | 233.0 | 289 |
| frangieh_context_holdout_2_Control | source_train_pair | train | 258 | 58 | 210.0 | 475 |
| frangieh_context_holdout_2_Control | source_validation_pair | val | 30 | 74 | 230.0 | 341 |
| frangieh_context_holdout_3_IFNγ | context_and_perturbation_unseen | test | 30 | 125 | 229.5 | 289 |
| frangieh_context_holdout_3_IFNγ | context_unseen_row | test | 159 | 78 | 227.0 | 341 |
| frangieh_context_holdout_3_IFNγ | perturbation_unseen_column | test | 60 | 73 | 171.5 | 418 |
| frangieh_context_holdout_3_IFNγ | random_missing_pair | test | 30 | 50 | 199.5 | 390 |
| frangieh_context_holdout_3_IFNγ | source_train_pair | train | 258 | 55 | 158.0 | 475 |
| frangieh_context_holdout_3_IFNγ | source_validation_pair | val | 30 | 61 | 166.5 | 396 |

## 训练子矩阵

| fold_id | heldout_context | train_fraction | n_train_pairs | n_train_contexts | n_train_perturbations |
|---|---|---|---|---|---|
| frangieh_context_holdout_1_Co-culture | Co-culture | 0.25 | 64 | 2 | 55 |
| frangieh_context_holdout_1_Co-culture | Co-culture | 0.5 | 129 | 2 | 98 |
| frangieh_context_holdout_1_Co-culture | Co-culture | 0.75 | 194 | 2 | 134 |
| frangieh_context_holdout_1_Co-culture | Co-culture | 1.0 | 258 | 2 | 154 |
| frangieh_context_holdout_2_Control | Control | 0.25 | 64 | 2 | 60 |
| frangieh_context_holdout_2_Control | Control | 0.5 | 128 | 2 | 100 |
| frangieh_context_holdout_2_Control | Control | 0.75 | 194 | 2 | 135 |
| frangieh_context_holdout_2_Control | Control | 1.0 | 258 | 2 | 152 |
| frangieh_context_holdout_3_IFNγ | IFNγ | 0.25 | 64 | 2 | 56 |
| frangieh_context_holdout_3_IFNγ | IFNγ | 0.5 | 128 | 2 | 100 |
| frangieh_context_holdout_3_IFNγ | IFNγ | 0.75 | 194 | 2 | 131 |
| frangieh_context_holdout_3_IFNγ | IFNγ | 1.0 | 258 | 2 | 155 |

E97 只完成实验合同，不把 reference predictor 当作正式双模型结果。后续预测器必须读取 `E97_TASK_MANIFEST.csv`，每个训练比例重新训练，并在预测落盘后才读取 test truth。
