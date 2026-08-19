# E55 跨数据集 transfer 审计

这一轮专门回答老师最后强调的 setting：一个数据集作为历史经验，另一个数据集作为目标场景。

分数输入只包含预测前能拿到的东西：源数据支持数、源/目标 control 状态相似度、两个源域参考预测器的分歧、预测效应大小。目标真值只用于最后计算误差。

## 1. 跑了什么

- 计划方向对：26
- 成功打分方向对：24
- 目标任务打分行数：634

数据分两类：

- same_system_cross_file：同一研究体系的跨文件迁移，例如 KaggleCrossCell 到 KaggleCrossPatient、Kang CrossCell 到 CrossPatient。
- hard_chemical_cross_dataset：不同化学扰动数据集互相迁移，例如 Kaggle、SciPlex3、McFarland、crossPatient。
- feasibility_boundary：只做可计算性检查。比如 sciplex3 和 TCDD 基因交集太少，不能硬做结论。

## 2. 主结果表

| 分组 | 方向 | 任务数 | 共同基因 | 共享扰动任务 | ρ(risk,error) | top20 错误富集 | 平均误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard_chemical_cross_dataset | KaggleCrossCell_celltype -> McFarland_cellline | 42 | 1000 | 0 | 0.602 | 1.365 | 0.200 |
| hard_chemical_cross_dataset | McFarland_cellline -> crossPatient_patient | 10 | 1000 | 0 | 0.454 | 1.203 | 0.139 |
| hard_chemical_cross_dataset | KaggleCrossPatient_celltype -> sciplex3_cellline | 27 | 566 | 0 | 0.303 | 1.067 | 0.089 |
| hard_chemical_cross_dataset | KaggleCrossCell_celltype -> sciplex3_cellline | 27 | 566 | 0 | 0.297 | 1.167 | 0.097 |
| hard_chemical_cross_dataset | crossPatient_patient -> McFarland_cellline | 42 | 1000 | 0 | 0.252 | 1.169 | 0.152 |
| hard_chemical_cross_dataset | sciplex3_cellline -> KaggleCrossCell_celltype | 24 | 566 | 0 | 0.225 | 1.286 | 0.117 |
| hard_chemical_cross_dataset | crossPatient_patient -> KaggleCrossPatient_celltype | 38 | 933 | 0 | 0.107 | 0.813 | 0.149 |
| hard_chemical_cross_dataset | sciplex3_cellline -> KaggleCrossPatient_celltype | 38 | 566 | 0 | 0.099 | 0.880 | 0.102 |
| hard_chemical_cross_dataset | crossPatient_patient -> KaggleCrossCell_celltype | 24 | 933 | 0 | 0.098 | 0.743 | 0.174 |
| hard_chemical_cross_dataset | KaggleCrossPatient_celltype -> McFarland_cellline | 42 | 1000 | 0 | 0.002 | 1.039 | 0.172 |
| hard_chemical_cross_dataset | sciplex3_cellline -> McFarland_cellline | 42 | 624 | 0 | 0.000 | 0.968 | 0.138 |
| hard_chemical_cross_dataset | McFarland_cellline -> KaggleCrossCell_celltype | 24 | 1000 | 0 | -0.018 | 0.639 | 0.180 |
| hard_chemical_cross_dataset | McFarland_cellline -> sciplex3_cellline | 27 | 624 | 0 | -0.058 | 0.939 | 0.120 |
| hard_chemical_cross_dataset | KaggleCrossCell_celltype -> crossPatient_patient | 10 | 933 | 0 | -0.153 | 0.854 | 0.162 |
| hard_chemical_cross_dataset | McFarland_cellline -> KaggleCrossPatient_celltype | 38 | 1000 | 0 | -0.209 | 0.694 | 0.155 |
| hard_chemical_cross_dataset | KaggleCrossPatient_celltype -> crossPatient_patient | 10 | 933 | 0 | -0.227 | 0.865 | 0.156 |
| hard_chemical_cross_dataset | crossPatient_patient -> sciplex3_cellline | 27 | 453 | 0 | -0.448 | 0.976 | 0.097 |
| hard_chemical_cross_dataset | sciplex3_cellline -> crossPatient_patient | 10 | 453 | 0 | -0.785 | 0.676 | 0.106 |
| same_system_cross_file | kangCrossCell_celltype -> kangCrossPatient_celltype | 8 | 1000 | 8 | 0.952 | 1.308 | 0.046 |
| same_system_cross_file | KaggleCrossCell_celltype -> KaggleCrossPatient_donor | 30 | 1000 | 30 | 0.861 | 1.610 | 0.036 |
| same_system_cross_file | kangCrossPatient_celltype -> kangCrossCell_celltype | 8 | 1000 | 8 | 0.833 | 1.339 | 0.045 |
| same_system_cross_file | KaggleCrossPatient_donor -> KaggleCrossCell_celltype | 24 | 1000 | 24 | 0.588 | 1.356 | 0.033 |
| same_system_cross_file | KaggleCrossPatient_celltype -> KaggleCrossCell_celltype | 24 | 1000 | 24 | 0.577 | 1.279 | 0.012 |
| same_system_cross_file | KaggleCrossCell_celltype -> KaggleCrossPatient_celltype | 38 | 1000 | 38 | -0.163 | 0.910 | 0.018 |

## 3. 怎么给老师解释

汇报时可以这样说：

> 老师上次提到跨数据集预测，我这次把它单独拆出来做了。源数据集只提供历史支持、control 状态和参考预测器，目标数据集的真实效应没有进入打分。最后再用目标真值算误差，看这个风险分数能不能把更容易错的任务排到前面。

如果 same_system_cross_file 为正，说明在相同研究体系内换一个文件/划分，风险排序还有迁移性。

hard_chemical_cross_dataset 如果很弱，就按边界处理：不同药物面板、不同细胞体系、共同基因较少时，源域经验未必能直接迁移。这部分写进限制和下一步，别硬吹。

## 4. 和老师要求逐条对应

| 老师的要求 | 当前对应证据 | 状态 |
|---|---|---|
| 分数到底和谁的误差相关 | E33 已审计：误差来自参考预测器；E55 继续用 source-only 预测器并记录 error_combined_rmse | 已补 |
| 输入不能偷看测试答案 | E55 的 risk_cross_dataset 不含 true_l2；true_l2 只标成 oracle diagnostic | 已补 |
| 小矩阵/低覆盖 | E34/E35 split smoke，E49-E52 正式化里已有低支持和留出版本 | 已有 |
| 整行/整列留出 | E34/E35、E49/E50/E52 覆盖 leave-context / leave-perturbation / dose-aware | 已有 |
| 一个数据集到另一个数据集 | 本轮 E55 | 新增 |
| 不同数据类型 | E40-E54 覆盖 chemical、gene combo、dose、regulatory、多模态审计；E55 重点补 chemical/immune 跨数据集 | 持续补 |

## 5. 文件

- 分数明细：`docs/实验结果/E55_cross_dataset_transfer_20260710/tables/E55_CROSS_DATASET_SCORE_TABLE.csv`
- 汇总表：`docs/实验结果/E55_cross_dataset_transfer_20260710/tables/E55_CROSS_DATASET_SUMMARY.csv`
- pair 状态：`docs/实验结果/E55_cross_dataset_transfer_20260710/tables/E55_PAIR_STATUS.csv`
- 数据任务状态：`docs/实验结果/E55_cross_dataset_transfer_20260710/tables/E55_DATASET_TASK_STATUS.csv`
- 运行状态：`docs/实验结果/E55_cross_dataset_transfer_20260710/RUN_STATUS.json`
