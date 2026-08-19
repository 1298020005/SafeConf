# E81｜sciPlex3 子矩阵四象限正式合同

本轮只冻结任务和基因面板，不训练模型。拆分只读取 context、perturbation、每个任务的细胞数和固定随机种子；1000 基因面板只由 vehicle control 表达选择。目标扰动后的表达没有参与任务选择、基因选择或分组。

- 数据：`/home/yyf/data/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/sciplex3.h5ad`
- 原始形状：26,046 cells × 5,000 genes
- 合格任务：103，context=3，base drug=9，dose=4
- 冻结设置：9（3 个扰动覆盖度 × 3 个 held-out context）
- gene order hash：`sha256:e53bfe14230003dbba9396958fae945532976804da8c5950a489b1f269953618`

每个设置把任务分成一个训练子矩阵和四类测试任务：子矩阵内随机缺失 pair、新 context、新 base drug、context 与 base drug 同时未见。整列留出时，同一种药的所有剂量一起留出。后续 CPA/chemCPA 和基线只能读取 `role=train` 的 perturbed expression；测试真值封存到最终误差计算。

## 任务数

| manifest_id | heldout_context | new_context_new_perturbation | new_context_seen_perturbation | observed_submatrix_train | seen_context_new_perturbation | seen_context_seen_perturbation_pair_holdout |
|---|---|---|---|---|---|---|
| E81_r1_p25 | A549 | 28.0 | 8.0 | 12.0 | 51.0 | 4.0 |
| E81_r1_p50 | A549 | 20.0 | 16.0 | 26.0 | 35.0 | 6.0 |
| E81_r1_p75 | A549 | 8.0 | 28.0 | 44.0 | 12.0 | 11.0 |
| E81_r2_p25 | K562 | 26.0 | 8.0 | 11.0 | 54.0 | 4.0 |
| E81_r2_p50 | K562 | 18.0 | 16.0 | 25.0 | 38.0 | 6.0 |
| E81_r2_p75 | K562 | 8.0 | 26.0 | 42.0 | 16.0 | 11.0 |
| E81_r3_p25 | MCF7 | 25.0 | 8.0 | 12.0 | 54.0 | 4.0 |
| E81_r3_p50 | MCF7 | 19.0 | 14.0 | 24.0 | 40.0 | 6.0 |
| E81_r3_p75 | MCF7 | 8.0 | 25.0 | 43.0 | 16.0 | 11.0 |

## 文件

- `tables/E81_TASK_MATRIX.csv`
- `tables/E81_GENE_PANEL.csv`
- `tables/E81_SPLIT_MANIFEST.csv`
- `tables/E81_SPLIT_SUMMARY.csv`
