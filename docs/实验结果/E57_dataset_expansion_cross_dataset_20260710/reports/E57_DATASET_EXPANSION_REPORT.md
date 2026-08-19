# E57 数据集扩容：跨数据集审计

E55/E56 先把老师点名的跨数据集 setting 跑通。E57 继续加本地已下载的数据集，检查信号是否只来自 Kaggle/Kang 那几组。

- 计划方向对：26
- 成功打分方向对：26
- 目标任务打分行数：3796

新增覆盖：Lara 骨髓 CRISPR、Dixit TF 时间点、Tian CRISPRa/i、Replogle 小实验、Adamson/Replogle/Tian 跨研究、SciPlex2/4 化学扰动。

## 主结果

| 分组 | 方向 | 任务数 | 共同基因 | 共享扰动任务 | ρ(risk,error) | top20 错误富集 | 平均误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| dixit_timecourse | Dixit_13d_target -> Dixit_7d_target | 10 | 500 | 10 | 0.588 | 1.266 | 0.941 |
| dixit_timecourse | Dixit_7d_target -> Dixit_13d_target | 10 | 500 | 10 | -0.127 | 0.954 | 0.448 |
| hard_genetic_cross_study | TianInhibition_batch -> Adamson_global | 76 | 500 | 4 | 0.297 | 1.212 | 0.046 |
| hard_genetic_cross_study | Adamson_global -> Replogle_exp7_global | 104 | 500 | 6 | 0.120 | 1.005 | 0.041 |
| hard_genetic_cross_study | Replogle_exp7_global -> Adamson_global | 76 | 500 | 6 | 0.080 | 1.109 | 0.039 |
| hard_genetic_cross_study | Adamson_global -> TianInhibition_batch | 705 | 500 | 13 | -0.057 | 0.992 | 0.067 |
| lara_bone_marrow_cross_condition | Lara_leukemia_celltype -> Lara_exvivo_celltype | 288 | 500 | 288 | 0.547 | 1.888 | 1.908 |
| lara_bone_marrow_cross_condition | Lara_leukemia_celltype -> Lara_invivo_celltype | 249 | 500 | 249 | 0.538 | 1.862 | 3.277 |
| lara_bone_marrow_cross_condition | Lara_invivo_celltype -> Lara_exvivo_celltype | 288 | 500 | 267 | 0.384 | 1.526 | 1.144 |
| lara_bone_marrow_cross_condition | Lara_invivo_celltype -> Lara_leukemia_celltype | 147 | 500 | 113 | 0.205 | 1.398 | 2.563 |
| lara_bone_marrow_cross_condition | Lara_exvivo_celltype -> Lara_leukemia_celltype | 147 | 500 | 127 | 0.159 | 1.152 | 1.990 |
| lara_bone_marrow_cross_condition | Lara_exvivo_celltype -> Lara_invivo_celltype | 249 | 500 | 249 | -0.059 | 1.282 | 1.335 |
| replogle_small_cross_experiment | Replogle_exp6_global -> Replogle_exp8_global | 61 | 500 | 3 | 0.146 | 1.058 | 0.040 |
| replogle_small_cross_experiment | Replogle_exp7_global -> Replogle_exp8_global | 61 | 500 | 14 | 0.128 | 1.111 | 0.029 |
| replogle_small_cross_experiment | Replogle_exp6_global -> Replogle_exp7_global | 104 | 500 | 7 | 0.092 | 1.016 | 0.034 |
| replogle_small_cross_experiment | Replogle_exp7_global -> Replogle_exp6_global | 69 | 500 | 7 | 0.071 | 1.068 | 0.033 |
| replogle_small_cross_experiment | Replogle_exp8_global -> Replogle_exp6_global | 69 | 500 | 3 | -0.123 | 0.999 | 0.045 |
| replogle_small_cross_experiment | Replogle_exp8_global -> Replogle_exp7_global | 104 | 500 | 14 | -0.200 | 0.919 | 0.027 |
| sciplex_series_cross_dataset | sciplex3_small_cellline -> SciPlex4_cellline | 14 | 500 | 0 | -0.124 | 0.968 | 0.064 |
| sciplex_series_cross_dataset | SciPlex4_cellline -> sciplex3_small_cellline | 27 | 500 | 0 | -0.303 | 0.976 | 0.089 |
| sciplex_series_cross_dataset | SciPlex2_cellline -> sciplex3_small_cellline | 27 | 500 | 0 | -0.349 | 0.995 | 0.515 |
| sciplex_series_cross_dataset | SciPlex2_cellline -> SciPlex4_cellline | 14 | 500 | 0 | -0.656 | 0.986 | 0.511 |
| sciplex_series_cross_dataset | SciPlex4_cellline -> SciPlex2_cellline | 4 | 500 | 0 | nan | nan | 0.507 |
| sciplex_series_cross_dataset | sciplex3_small_cellline -> SciPlex2_cellline | 4 | 500 | 0 | nan | nan | 0.498 |
| tian_crispra_crispri | TianInhibition_batch -> TianActivation_batch | 184 | 500 | 42 | 0.108 | 1.091 | 0.050 |
| tian_crispra_crispri | TianActivation_batch -> TianInhibition_batch | 705 | 500 | 81 | 0.025 | 1.013 | 0.052 |

## 读法

这批结果主要用来增加数据覆盖面。强结果可以作为证据，弱结果和跳过项照样有用：它们说明哪些生物体系、共同基因、control 结构和任务数量会限制跨数据集打分。

适合进主汇报的新增结果：Lara 骨髓 CRISPR 三个条件互相迁移。它的任务数更大，且 leukemia -> exvivo / invivo 两个方向能把高错误任务排到前面。

Dixit 7d/13d 可作为时间点迁移补充，但目标任务只有 10 个。Tian、Replogle、Adamson 主要作为跨研究边界。SciPlex2/4 的 source/target 任务太少，只放补充表。

## 文件

- 汇总表：`docs/实验结果/E57_dataset_expansion_cross_dataset_20260710/tables/E57_DATASET_EXPANSION_SUMMARY.csv`
- 分数明细：`docs/实验结果/E57_dataset_expansion_cross_dataset_20260710/tables/E57_DATASET_EXPANSION_SCORE_TABLE.csv`
- pair 状态：`docs/实验结果/E57_dataset_expansion_cross_dataset_20260710/tables/E57_DATASET_EXPANSION_PAIR_STATUS.csv`
- 数据任务状态：`docs/实验结果/E57_dataset_expansion_cross_dataset_20260710/tables/E57_DATASET_EXPANSION_DATASET_STATUS.csv`
