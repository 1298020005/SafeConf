# E55 周老师要求执行记录：跨数据集与下载线

时间：2026-07-10

这份记录只服务一件事：把周老师最后那段话逐条落到实验和数据状态里，后面汇报时不靠印象说。

## 1. 老师最后真正要求了什么

| 老师关心点 | 我的理解 | 已落地内容 |
|---|---|---|
| risk score 到底在和谁的误差比 | 分数最后要和某个参考预测器的真实预测误差比较，不能只说“可信度” | E33 已做输入来源与误差来源审计；E55 每个任务记录 `error_combined_rmse` |
| 分数输入能不能提前得到 | 打分阶段不能使用目标真值，也不能把真实 effect magnitude 当输入 | E55 的 `risk_cross_dataset` 只用 source support、control similarity、predictor disagreement、predicted magnitude |
| random split 太容易 | 只留出随机 pair 不够，要看更难的矩阵结构 | E34/E35 已做 submatrix、整行、整列 split；E49-E52 做了 OpenProblems、sciplex3、Norman、TCDD 的正式化版本 |
| 一个数据集到另一个数据集 | 源数据集全部作为历史经验，目标数据集作为新场景 | 本轮新增 E55 cross-dataset transfer |
| 数据类型要更丰富 | gene、chemical、dose、组合扰动、regulatory、多模态都要看可计算性 | E40-E54 已补数据账本和多线 smoke/formal；E55 补 chemical/immune 跨数据集 |

## 2. E55 做了什么

目录：`docs/实验结果/E55_cross_dataset_transfer_20260710`

代码：`tools/scripts/run_e55_cross_dataset_transfer.py`

这次把目标拆成两类：

| 类型 | 目的 | 方向数 |
|---|---|---:|
| same_system_cross_file | 同一研究体系内换文件/换划分，看风险排序是否还能迁移 | 6 |
| hard_chemical_cross_dataset | 不同化学扰动数据集互相迁移，压力更大 | 18 |
| feasibility_boundary | 检查能不能算；不能算的直接记录 | 2 |

总结果：

| 指标 | 数值 |
|---|---:|
| 计划方向对 | 26 |
| 成功打分方向对 | 24 |
| 目标任务打分行数 | 634 |
| TCDD 与 sciplex3 共同基因 | 1 |

TCDD 和 sciplex3 方向被跳过。理由很简单：共同基因只有 1 个，不能拿来比较表达向量。这个边界要如实说。

## 3. 目前最好看的结果

看 `risk_cross_dataset` 和目标预测误差 `error_combined_rmse` 的 Spearman 相关。

| 方向 | 任务数 | ρ | top20 错误富集 |
|---|---:|---:|---:|
| kangCrossCell_celltype -> kangCrossPatient_celltype | 8 | 0.952 | 1.308 |
| KaggleCrossCell_celltype -> KaggleCrossPatient_donor | 30 | 0.861 | 1.610 |
| kangCrossPatient_celltype -> kangCrossCell_celltype | 8 | 0.833 | 1.339 |
| KaggleCrossPatient_donor -> KaggleCrossCell_celltype | 24 | 0.588 | 1.356 |
| KaggleCrossPatient_celltype -> KaggleCrossCell_celltype | 24 | 0.577 | 1.279 |
| KaggleCrossCell_celltype -> McFarland_cellline | 42 | 0.602 | 1.365 |

这几个结果说明：同一体系内跨文件，信号比较清楚；不同化学数据集之间也有少数方向能筛到高错误任务。

## 4. 也要保留的负面结果

| 方向 | ρ | 说明 |
|---|---:|---|
| KaggleCrossCell_celltype -> KaggleCrossPatient_celltype | -0.163 | 同 drug panel、同 cell-type 语义下，任务本身误差很小，risk 排序没拉开 |
| sciplex3_cellline -> crossPatient_patient | -0.785 | 目标只有 10 个任务，小样本，且药物面板完全不同 |
| crossPatient_patient -> sciplex3_cellline | -0.448 | source 只有 10 个任务，历史支持太薄 |
| sciplex3_cellline <-> TCDD_mouse_liver | 跳过 | 共同基因只有 1 个，不能比较表达向量 |

这些负面结果不能删。它们正好回答老师说的“取决于这些方法能不能在这些 setting 上算”。

## 5. 下载线状态

| 数据线 | 当前状态 | 说明 |
|---|---|---|
| Tahoe-100M raw | 继续下载中 | `aria2c` 进程还在跑；本地 data 约 93G；仍有 `.aria2` 残片 |
| OpenProblems NeurIPS 2023 | 核心 DGE 可用，剩余大文件网络失败 | `de_train.h5ad`、`de_test.h5ad` 已可用；OpenProblems S3 目前反复 TLS EOF，`moa_annotations.csv` 这种小文件也失败 |
| scPerturb / extra official h5ad | 已可用 | E42-E55 已直接使用 |

OpenProblems 续传已经尝试过；当前失败原因是 `openproblems-bio.s3.amazonaws.com` 的 TLS 连接中断。后续可以换时间重试，或者找镜像源。当前 E55 不依赖那些未完成的大文件。

## 6. 下一步实验安排

| 优先级 | 实验 | 为什么要做 | 产出 |
|---|---|---|---|
| P1 | E56 cross-dataset source-size ablation | 老师提到“小矩阵”，跨数据集也要看 source task 数从少到多时信号如何变化 | 已完成：`docs/实验结果/E56_cross_dataset_source_size_ablation_20260710` |
| P1 | E57 dataset expansion | 回答“数据集覆盖是否太少”，继续补本地已下载的 Lara、Dixit、Tian、Replogle、Adamson、SciPlex2/4 | 已完成：`docs/实验结果/E57_dataset_expansion_cross_dataset_20260710` |
| P1 | E58 bootstrap / permutation | E55/E57 里部分方向任务数少，需要 bootstrap CI 或置换检验 | 置信区间、稳定性图 |
| P2 | E59 submatrix/row/column formal polish | 把 E34/E35 的 smoke 结果升级成更正式的表和图 | 每个 dataset 的 coverage、row holdout、column holdout summary |
| P2 | E59 predictor replacement | 当前 E55 是轻量预测器，后面接 scGPT/GEARS/CPA 输出时要复查误差来源 | 同一 split 上的 predictor 对照表 |

当前最稳的汇报顺序：

1. 先回答老师：分数和哪个误差比、有没有偷看答案。
2. 再说矩阵难度：submatrix、整行、整列已经有结果。
3. 新增跨数据集 E55：同体系迁移整体较好，硬化学迁移有方向性，TCDD 因共同基因过少不强算。
4. E56 已补 source-size ablation：同体系方向比较稳，硬化学迁移明显依赖源任务数量和源/目标相似性。
5. E57 已补数据集扩容：Lara 骨髓 CRISPR 是最值得进主汇报的新增数据线；Dixit、Tian、Replogle、Adamson、SciPlex2/4 作为覆盖面和边界。
6. 下一步做 E58，把 E55/E57 的小样本方向补上置信区间或置换检验。
