# E197｜Systema 与 scPertEval 均值层评价

## 定位

本分析使用 E190/E192 已开封的旧结果，标签是 `POSTTRUTH_EXPLORATORY`。它补齐多指标评价，不是新的盲法确认。

现有预测只保存任务均值，因此 scPertEval 先按目标细胞数把 batch×gene 效应合并为每个 target gene 一个 centroid，再运行 pseudobulk/centroid 协议。没有运行 MMD、Energy、Sinkhorn、DE-AUPRC、DE-AUROC、DE-overlap、WMSE，也没有复制均值伪造预测细胞。

## 主要预测器与简单基线

| setting | predictor | mean_systema_inspired_transport_pearson_delta_all | mean_systema_inspired_transport_pearson_delta_abs_effect_top20_proxy | mean_systema_gene_centroid_accuracy_effect | mean_systema_gene_centroid_accuracy_post | mean_scperteval_mse | mean_scperteval_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E190_K562 | family_centroid | 0.3775 | 0.4591 | 0.6064 | 0.6212 | 0.0250 | 0.1415 |
| E190_K562 | matching_source_all_folds_mean | 0.3341 | 0.4510 | 0.7317 | 0.7553 | 0.0265 | 0.1087 |
| E190_K562 | matching_source_train_mean | 0.3390 | 0.4637 | 0.7285 | 0.7498 | 0.0264 | 0.1092 |
| E190_K562 | source_absolute_noncontrol_mean | 0.1486 | 0.2505 | 0.4991 | 0.5000 | 0.2548 | 0.4977 |
| E190_K562 | target_control_plus_source_mean_effect | NA | NA | 0.5000 | 0.5227 | 0.0310 | 0.5407 |
| E190_K562 | zero_effect | 0.0979 | -0.0506 | 0.5000 | 0.5222 | 0.0315 | 0.5407 |
| E192_RPE1 | family_centroid | 0.1562 | 0.2446 | 0.5119 | 0.5262 | 0.0876 | 0.1667 |
| E192_RPE1 | matching_source_all_folds_mean | 0.0732 | 0.2053 | 0.5262 | 0.5571 | 0.0925 | 0.1857 |
| E192_RPE1 | matching_source_train_mean | 0.0748 | 0.2028 | 0.5262 | 0.5571 | 0.0914 | 0.1810 |
| E192_RPE1 | source_absolute_noncontrol_mean | 0.1501 | 0.4752 | 0.5238 | 0.5000 | 0.3984 | 0.2810 |
| E192_RPE1 | target_control_plus_source_mean_effect | NA | NA | 0.5000 | 0.5214 | 0.0917 | 0.4952 |
| E192_RPE1 | zero_effect | 0.1832 | 0.1361 | 0.5000 | 0.5167 | 0.0894 | 0.4952 |

跨数据集主 Pearson-Δ 使用 `target control + source mean effect` 作为Systema-inspired 训练侧参考；另存 source train absolute reference 敏感性结果，不把前者称为 Systema 官方原式。基因 centroid accuracy 按目标细胞数重建每个基因的真实与预测质心，再使用 Systema 官方欧氏距离公式。主列先减去匹配 target control；post-state 列保留批次构成敏感性。MSE 与 rank 越低越好。

## 既有风险量对 Systema 全基因误差的排序

| setting | risk | spearman | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- |
| E190_K562 | source_effect_magnitude | -0.6515 | -0.8048 | -0.4243 |
| E192_RPE1 | diversity_lower_bound | -0.3208 | -0.7304 | 0.2317 |

表中点估计先在每个 target gene 内平均各 batch，再让 47/21 个基因等权；区间按 target gene bootstrap 5,000 次。task-weighted 结果另存于完整表。这些相关性只说明旧数据中的排序关系，不会把 E190/E192 的 ABSTAIN/PASS gate 重新判一次。

## Systema reference 与 top20 边界

Adamson train guide 数：E190_K562=54, E192_RPE1=26。每个 guide 的四个 train 伪重复先按细胞数合并，再对 guide 等权。`matching_source_train_mean` 只读 train；`matching_source_all_folds_mean` 读取既有 source all-fold asset，只作为更强敏感性基线。两者与全局 source mean 分开保存。

目标文件只有任务均值，没有逐细胞差异检验。因此 `abs-effect top20` 是按真实效应绝对值选出的事后代理，不称为官方 Systema Pearson-Δ20；source top20 完全由 Adamson 训练效应确定。

## 完整性

- formal gates：47/47 通过；
- scPertEval official source commit：`8709eb07a0e7d4ecf1c60c977f2018690a749975`；
- Systema official source commit：`aaf5b5353993b48b78543f2f93b3e18ca65df515`；
- 图均为白底 PNG/PDF；完整 task 指标、官方协议原始分数和 bootstrap 结果见 `tables/`。

## 解释边界

单个 pseudobulk centroid 不能恢复细胞内异质性或实验重复性。本结果不能写成完整 scPertEval population benchmark，也不能用来保证投稿录用。
