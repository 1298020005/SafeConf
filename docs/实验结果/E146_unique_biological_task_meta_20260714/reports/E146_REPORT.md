# E146｜唯一生物任务统计依赖再审计

E146 在 E140 结果已经解封之后执行。它回答重复 outer-fold 记录是否让区间过窄，不构成新的独立确认。

## 重复规模

| dataset | n_rows | n_folds | n_unique_perturbations | n_unique_context_tasks | mean_occurrences | median_occurrences | max_occurrences | n_tasks_repeated_across_folds | row_to_unique_context_task_ratio |
|---|---|---|---|---|---|---|---|---|---|
| Frangieh | 837 | 3 | 189 | 567 | 1.4762 | 1.0000 | 3 | 237 | 1.4762 |
| Lara_exvivo | 345 | 5 | 31 | 155 | 2.2258 | 2.0000 | 5 | 117 | 2.2258 |
| Liang | 612 | 9 | 18 | 162 | 3.7778 | 4.0000 | 7 | 151 | 3.7778 |
| Nadig_two_cellline | 256 | 2 | 96 | 192 | 1.3333 | 1.0000 | 2 | 64 | 1.3333 |
| Santinha | 255 | 5 | 23 | 115 | 2.2174 | 2.0000 | 4 | 92 | 2.2174 |
| Shifrut | 172 | 4 | 20 | 80 | 2.1500 | 2.0000 | 3 | 67 | 2.1500 |
| Tian_CRISPRi | 732 | 4 | 99 | 396 | 1.8485 | 2.0000 | 4 | 254 | 1.8485 |

3209 行记录对应 1667 个 `(dataset, context, perturbation)` context-task，以及 476 个 `(dataset, perturbation)` 簇。E140 fold-macro 主区间以 perturbation 整簇重抽，因此同一扰动的所有 context 与 outer-fold 记录同步出现；context-task 聚类和 pooled median 都只作敏感性。

## SafeConf 相对比较器：聚类 bootstrap

| dataset | estimand | metric | observed | ci95_low | ci95_high | p_gt_zero |
|---|---|---|---|---|---|---|
| Frangieh | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_disagreement | 0.1157 | 0.0383 | 0.1925 | 0.9980 |
| Frangieh | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_magnitude | 0.1049 | 0.0277 | 0.1834 | 0.9963 |
| Frangieh | pooled_context_task_median_sensitivity | delta_rho__safe_minus_disagreement | 0.1231 | 0.0346 | 0.2121 | 0.9970 |
| Frangieh | pooled_context_task_median_sensitivity | delta_rho__safe_minus_magnitude | 0.0752 | -0.0104 | 0.1638 | 0.9593 |
| Lara_exvivo | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_disagreement | 0.2110 | 0.0890 | 0.2985 | 0.9997 |
| Lara_exvivo | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_magnitude | 0.2396 | 0.1185 | 0.3427 | 1.0000 |
| Lara_exvivo | pooled_context_task_median_sensitivity | delta_rho__safe_minus_disagreement | -0.0236 | -0.1745 | 0.1217 | 0.3860 |
| Lara_exvivo | pooled_context_task_median_sensitivity | delta_rho__safe_minus_magnitude | 0.0254 | -0.1217 | 0.1718 | 0.6453 |
| Liang | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_disagreement | 0.1379 | -0.0073 | 0.2742 | 0.9670 |
| Liang | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_magnitude | 0.1382 | 0.0031 | 0.2857 | 0.9763 |
| Liang | pooled_context_task_median_sensitivity | delta_rho__safe_minus_disagreement | 0.2532 | 0.0721 | 0.4247 | 0.9983 |
| Liang | pooled_context_task_median_sensitivity | delta_rho__safe_minus_magnitude | 0.2327 | 0.0533 | 0.4110 | 0.9943 |
| Nadig_two_cellline | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_disagreement | 0.0007 | -0.0876 | 0.0880 | 0.5277 |
| Nadig_two_cellline | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_magnitude | -0.1716 | -0.2980 | -0.0468 | 0.0030 |
| Nadig_two_cellline | pooled_context_task_median_sensitivity | delta_rho__safe_minus_disagreement | 0.0215 | -0.1124 | 0.1565 | 0.6340 |
| Nadig_two_cellline | pooled_context_task_median_sensitivity | delta_rho__safe_minus_magnitude | -0.1375 | -0.2935 | 0.0286 | 0.0550 |
| Santinha | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_disagreement | 0.1914 | -0.0397 | 0.4182 | 0.9520 |
| Santinha | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_magnitude | 0.1539 | -0.0762 | 0.3216 | 0.9003 |
| Santinha | pooled_context_task_median_sensitivity | delta_rho__safe_minus_disagreement | -0.0930 | -0.2841 | 0.0967 | 0.1693 |
| Santinha | pooled_context_task_median_sensitivity | delta_rho__safe_minus_magnitude | -0.1285 | -0.3399 | 0.0783 | 0.1257 |
| Shifrut | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_disagreement | 0.1213 | -0.0129 | 0.2905 | 0.9613 |
| Shifrut | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_magnitude | -0.0360 | -0.2033 | 0.1443 | 0.3780 |
| Shifrut | pooled_context_task_median_sensitivity | delta_rho__safe_minus_disagreement | -0.0623 | -0.1655 | 0.0245 | 0.0847 |
| Shifrut | pooled_context_task_median_sensitivity | delta_rho__safe_minus_magnitude | -0.0392 | -0.1302 | 0.0536 | 0.1920 |
| Tian_CRISPRi | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_disagreement | 0.1521 | 0.0725 | 0.2312 | 1.0000 |
| Tian_CRISPRi | e140_fold_macro_perturbation_cluster | delta_rho__safe_minus_magnitude | 0.0675 | -0.0201 | 0.1550 | 0.9383 |
| Tian_CRISPRi | pooled_context_task_median_sensitivity | delta_rho__safe_minus_disagreement | 0.1338 | -0.0004 | 0.2641 | 0.9743 |
| Tian_CRISPRi | pooled_context_task_median_sensitivity | delta_rho__safe_minus_magnitude | 0.0784 | -0.0479 | 0.2081 | 0.8897 |

`e140_fold_macro_perturbation_cluster` 保持 E140 的原 fold-macro 点估计，并使用更严格的 perturbation-cluster 区间。`pooled_context_task_median_sensitivity` 把每个 context-task 跨 fold 取中位数，它明确是 pooled-median sensitivity，不是 E140 原估计量。

## 七研究随机效应：E140 fold-macro 与 pooled-median sensitivity 并列

| estimand | effect | k | pooled_z | ci95_low_z | ci95_high_z | tau2_reml | I2_percent | prediction_low_z | prediction_high_z |
|---|---|---|---|---|---|---|---|---|---|
| e140_fold_macro_perturbation_cluster | safeconf_minus_disagreement | 7 | 0.1292 | 0.0530 | 0.2053 | 0.0032 | 47.1988 | -0.0364 | 0.2947 |
| e140_fold_macro_perturbation_cluster | safeconf_minus_magnitude | 7 | 0.0749 | -0.0607 | 0.2104 | 0.0165 | 77.8467 | -0.2848 | 0.4345 |
| pooled_context_task_median_sensitivity | safeconf_minus_disagreement | 7 | 0.0520 | -0.0651 | 0.1690 | 0.0093 | 63.8441 | -0.2253 | 0.3292 |
| pooled_context_task_median_sensitivity | safeconf_minus_magnitude | 7 | 0.0200 | -0.0988 | 0.1387 | 0.0082 | 58.2616 | -0.2446 | 0.2846 |

| estimand | effect | pooled_rho_equivalent | ci95_low_rho_equivalent | ci95_high_rho_equivalent | prediction_low_rho_equivalent | prediction_high_rho_equivalent | tau2_reml | I2_percent |
|---|---|---|---|---|---|---|---|---|
| e140_fold_macro_perturbation_cluster | baseline_predicted_magnitude | 0.1383 | -0.0123 | 0.2828 | -0.2630 | 0.4988 | 0.0214 | 82.4377 |
| e140_fold_macro_perturbation_cluster | risk_model_disagreement | 0.0680 | -0.0439 | 0.1782 | -0.2099 | 0.3358 | 0.0099 | 71.8101 |
| e140_fold_macro_perturbation_cluster | safeconf_calibrated_pair_risk | 0.1995 | 0.1134 | 0.2826 | 0.0085 | 0.3764 | 0.0044 | 63.5839 |
| pooled_context_task_median_sensitivity | baseline_predicted_magnitude | 0.2541 | 0.1088 | 0.3887 | -0.1239 | 0.5677 | 0.0186 | 75.4821 |
| pooled_context_task_median_sensitivity | risk_model_disagreement | 0.2171 | 0.0597 | 0.3640 | -0.1869 | 0.5583 | 0.0211 | 75.8119 |
| pooled_context_task_median_sensitivity | safeconf_calibrated_pair_risk | 0.2690 | 0.1612 | 0.3705 | 0.0458 | 0.4667 | 0.0059 | 55.0101 |

两个 estimand 均使用 Fisher z、REML tau²、modified Knapp–Hartung 均值区间和研究预测区间。差值行位于 Fisher-z 尺度，不能当成原始 Δrho。

## LODO 的 Liang 依赖

| estimand | pooled_z | ci95_low_z | ci95_high_z |
|---|---|---|---|
| e140_fold_macro_perturbation_cluster | 0.0636 | -0.1015 | 0.2287 |
| pooled_context_task_median_sensitivity | -0.0036 | -0.1069 | 0.0998 |

删除 Liang 后，pooled-median sensitivity 的 SafeConf−magnitude 合并效应由全七研究的正值变为 -0.0036，发生符号反转。E140 fold-macro estimand 删除 Liang 后为 +0.0636。这一敏感性必须明写，不能只报告全数据平均。

## 方向风险：Nadig 单研究附录

E139 与 E140 Nadig 共 256 行一一对齐，absolute 端点最大差为 0。主方向 bootstrap 以 96 个 perturbation 整簇同步 HepG2、Jurkat；192 个 context-task pooled median 另作敏感性。方向结果没有并入七研究 absolute 元分析。

| estimand | score | n_independent_clusters | observed | ci95_low | ci95_high |
|---|---|---|---|---|---|
| nadig_direction_fold_macro_perturbation_cluster | directional_risk_frozen | 96 | 0.7535 | 0.6941 | 0.8021 |
| nadig_direction_fold_macro_perturbation_cluster | baseline_predicted_magnitude | 96 | 0.2913 | 0.1738 | 0.3936 |

## 解释边界

- E146 没有改变 SafeConf 分数、任务或端点。
- perturbation-cluster 同步同一扰动的所有 context 和 fold；context-task 与 pooled-median 结果仅用于敏感性。
- 七个研究仍然只有七个研究；prediction interval 比均值区间更接近未来新研究的不确定性。
- 删除 Liang 后 pooled-median SafeConf−magnitude 符号反转；完整 LODO 见 `tables/E146_LODO.csv`。
