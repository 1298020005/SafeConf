# E56 跨数据集 source-size ablation

这一轮把老师说的“小矩阵/历史任务少”放到跨数据集 setting 里检查：目标数据集固定，源数据集只给一部分任务。

- pair 数：6
- 分数明细行数：3450

## 主表

| 方向 | source fraction | 源任务数均值 | 目标任务数 | ρ均值 | ρ标准差 | top20富集均值 | 平均支持数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sciplex3_cellline -> crossPatient_patient | 0.25 | 7.0 | 10 | -0.705 | 0.040 | 0.697 | 0.000 |
| sciplex3_cellline -> crossPatient_patient | 0.50 | 14.0 | 10 | -0.709 | 0.026 | 0.675 | 0.000 |
| sciplex3_cellline -> crossPatient_patient | 0.75 | 20.0 | 10 | -0.718 | 0.000 | 0.685 | 0.000 |
| sciplex3_cellline -> crossPatient_patient | 1.00 | 27.0 | 10 | -0.718 | nan | 0.680 | 0.000 |
| sciplex3_cellline -> KaggleCrossCell_celltype | 0.25 | 7.0 | 24 | 0.171 | 0.146 | 1.192 | 0.000 |
| sciplex3_cellline -> KaggleCrossCell_celltype | 0.50 | 14.0 | 24 | 0.122 | 0.096 | 1.195 | 0.000 |
| sciplex3_cellline -> KaggleCrossCell_celltype | 0.75 | 20.0 | 24 | 0.126 | 0.089 | 1.192 | 0.000 |
| sciplex3_cellline -> KaggleCrossCell_celltype | 1.00 | 27.0 | 24 | 0.154 | nan | 1.280 | 0.000 |
| KaggleCrossCell_celltype -> McFarland_cellline | 0.25 | 6.0 | 42 | 0.414 | 0.236 | 1.214 | 0.000 |
| KaggleCrossCell_celltype -> McFarland_cellline | 0.50 | 12.0 | 42 | 0.374 | 0.247 | 1.178 | 0.000 |
| KaggleCrossCell_celltype -> McFarland_cellline | 0.75 | 18.0 | 42 | 0.445 | 0.225 | 1.248 | 0.000 |
| KaggleCrossCell_celltype -> McFarland_cellline | 1.00 | 24.0 | 42 | 0.605 | nan | 1.365 | 0.000 |
| kangCrossCell_celltype -> kangCrossPatient_celltype | 0.25 | 3.0 | 8 | 0.652 | 0.340 | 1.265 | 3.000 |
| kangCrossCell_celltype -> kangCrossPatient_celltype | 0.50 | 4.0 | 8 | 0.708 | 0.265 | 1.405 | 4.000 |
| kangCrossCell_celltype -> kangCrossPatient_celltype | 0.75 | 6.0 | 8 | 0.696 | 0.124 | 1.374 | 6.000 |
| kangCrossCell_celltype -> kangCrossPatient_celltype | 1.00 | 8.0 | 8 | 0.976 | nan | 1.332 | 8.000 |
| KaggleCrossPatient_celltype -> KaggleCrossCell_celltype | 0.25 | 10.0 | 24 | 0.343 | 0.083 | 1.226 | 1.083 |
| KaggleCrossPatient_celltype -> KaggleCrossCell_celltype | 0.50 | 19.0 | 24 | 0.157 | 0.164 | 1.140 | 2.047 |
| KaggleCrossPatient_celltype -> KaggleCrossCell_celltype | 0.75 | 28.0 | 24 | 0.190 | 0.129 | 1.339 | 3.031 |
| KaggleCrossPatient_celltype -> KaggleCrossCell_celltype | 1.00 | 38.0 | 24 | 0.390 | nan | 1.213 | 4.125 |
| KaggleCrossCell_celltype -> KaggleCrossPatient_donor | 0.25 | 6.0 | 30 | 0.573 | 0.156 | 1.341 | 0.600 |
| KaggleCrossCell_celltype -> KaggleCrossPatient_donor | 0.50 | 12.0 | 30 | 0.670 | 0.109 | 1.340 | 1.200 |
| KaggleCrossCell_celltype -> KaggleCrossPatient_donor | 0.75 | 18.0 | 30 | 0.809 | 0.032 | 1.609 | 1.800 |
| KaggleCrossCell_celltype -> KaggleCrossPatient_donor | 1.00 | 24.0 | 30 | 0.833 | nan | 1.593 | 2.400 |

## 汇报口径

这张表用来回答：如果历史矩阵只给一小块，风险排序还能不能用。

看法很直接：同体系方向一般更稳；硬化学迁移对源任务数量和源/目标相似性更敏感。后面写论文时，可以把这部分作为“数据覆盖度影响”的补充实验。

## 文件

- 聚合表：`docs/实验结果/E56_cross_dataset_source_size_ablation_20260710/tables/E56_SOURCE_SIZE_AGG_SUMMARY.csv`
- repeat 表：`docs/实验结果/E56_cross_dataset_source_size_ablation_20260710/tables/E56_SOURCE_SIZE_REPEAT_SUMMARY.csv`
- 分数明细：`docs/实验结果/E56_cross_dataset_source_size_ablation_20260710/tables/E56_SOURCE_SIZE_SCORE_TABLE.csv`
- pair 状态：`docs/实验结果/E56_cross_dataset_source_size_ablation_20260710/tables/E56_SOURCE_SIZE_PAIR_STATUS.csv`
