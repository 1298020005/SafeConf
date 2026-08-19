# E153｜八研究正式扩展元分析

## 结论范围

E153 在 E152 已经解封之后，把 Replogle 的256个正式主任务加入 E140 七研究。它是 post-E152 expanded meta-analysis，不是新的预注册独立 gate。Replogle 自身的预注册方向 gate 仍由 E152 单独承担。

输入共 3465 行、8 个研究、34 个fold、1923 个context-task、604 个perturbation簇。E151 strict issues=0；E152 score-freeze、模型和任务哈希全部通过；Replogle恰有128个扰动×2个held-out contexts=256个唯一主任务。

## 数据与重复结构

| dataset | n_rows | n_folds | n_contexts | n_unique_perturbations | n_unique_context_tasks | mean_outer_fold_occurrences | median_outer_fold_occurrences | max_outer_fold_occurrences | row_to_unique_context_task_ratio |
|---|---|---|---|---|---|---|---|---|---|
| Frangieh | 837 | 3 | 3 | 189 | 567 | 1.4762 | 1.0000 | 3 | 1.4762 |
| Lara_exvivo | 345 | 5 | 5 | 31 | 155 | 2.2258 | 2.0000 | 5 | 2.2258 |
| Liang | 612 | 9 | 9 | 18 | 162 | 3.7778 | 4.0000 | 7 | 3.7778 |
| Nadig_two_cellline | 256 | 2 | 2 | 96 | 192 | 1.3333 | 1.0000 | 2 | 1.3333 |
| Replogle_two_cellline | 256 | 2 | 2 | 128 | 256 | 1.0000 | 1.0000 | 1 | 1.0000 |
| Santinha | 255 | 5 | 5 | 23 | 115 | 2.2174 | 2.0000 | 4 | 2.2174 |
| Shifrut | 172 | 4 | 4 | 20 | 80 | 2.1500 | 2.0000 | 3 | 2.1500 |
| Tian_CRISPRi | 732 | 4 | 4 | 99 | 396 | 1.8485 | 2.0000 | 4 | 1.8485 |

## Absolute-RMSE：研究内fold-macro

| dataset | safeconf_calibrated_pair_risk | risk_model_disagreement | baseline_predicted_magnitude | safe_minus_disagreement | safe_minus_magnitude |
|---|---|---|---|---|---|
| Frangieh | 0.2530 | 0.1373 | 0.1481 | 0.1157 | 0.1049 |
| Lara_exvivo | 0.3871 | 0.1761 | 0.1475 | 0.2110 | 0.2396 |
| Liang | 0.2124 | 0.0746 | 0.0742 | 0.1379 | 0.1382 |
| Nadig_two_cellline | 0.2310 | 0.2302 | 0.4026 | 0.0007 | -0.1716 |
| Replogle_two_cellline | 0.1726 | -0.0364 | 0.2143 | 0.2090 | -0.0417 |
| Santinha | 0.0646 | -0.1268 | -0.0893 | 0.1914 | 0.1539 |
| Shifrut | 0.1727 | 0.0513 | 0.2087 | 0.1213 | -0.0360 |
| Tian_CRISPRi | 0.1342 | -0.0179 | 0.0667 | 0.1521 | 0.0675 |

正值表示风险分数随误差升高。SafeConf减比较器为正表示SafeConf的错误排序更强。

## 每研究perturbation-cluster区间

| dataset | metric | observed | ci95_low | ci95_high | p_gt_zero |
|---|---|---|---|---|---|
| Frangieh | delta_rho__safe_minus_disagreement | 0.1157 | 0.0372 | 0.1908 | 0.9987 |
| Frangieh | delta_rho__safe_minus_magnitude | 0.1049 | 0.0265 | 0.1788 | 0.9953 |
| Lara_exvivo | delta_rho__safe_minus_disagreement | 0.2110 | 0.0876 | 0.2927 | 1.0000 |
| Lara_exvivo | delta_rho__safe_minus_magnitude | 0.2396 | 0.1161 | 0.3421 | 1.0000 |
| Liang | delta_rho__safe_minus_disagreement | 0.1379 | -0.0053 | 0.2796 | 0.9703 |
| Liang | delta_rho__safe_minus_magnitude | 0.1382 | 0.0027 | 0.2957 | 0.9777 |
| Nadig_two_cellline | delta_rho__safe_minus_disagreement | 0.0007 | -0.0848 | 0.0837 | 0.5267 |
| Nadig_two_cellline | delta_rho__safe_minus_magnitude | -0.1716 | -0.2943 | -0.0453 | 0.0043 |
| Replogle_two_cellline | delta_rho__safe_minus_disagreement | 0.2090 | 0.1046 | 0.3181 | 0.9997 |
| Replogle_two_cellline | delta_rho__safe_minus_magnitude | -0.0417 | -0.1868 | 0.1215 | 0.3083 |
| Santinha | delta_rho__safe_minus_disagreement | 0.1914 | -0.0526 | 0.4123 | 0.9420 |
| Santinha | delta_rho__safe_minus_magnitude | 0.1539 | -0.0939 | 0.3218 | 0.8993 |
| Shifrut | delta_rho__safe_minus_disagreement | 0.1213 | -0.0193 | 0.2828 | 0.9533 |
| Shifrut | delta_rho__safe_minus_magnitude | -0.0360 | -0.2026 | 0.1523 | 0.3697 |
| Tian_CRISPRi | delta_rho__safe_minus_disagreement | 0.1521 | 0.0741 | 0.2367 | 1.0000 |
| Tian_CRISPRi | delta_rho__safe_minus_magnitude | 0.0675 | -0.0195 | 0.1542 | 0.9403 |

## 八研究随机效应：主estimand

| effect | k | pooled_z | ci95_low_z | ci95_high_z | tau2_reml | I2_percent | prediction_low_z | prediction_high_z |
|---|---|---|---|---|---|---|---|---|
| safeconf_minus_disagreement | 8 | 0.1397 | 0.0704 | 0.2089 | 0.0033 | 49.8576 | -0.0190 | 0.2983 |
| safeconf_minus_magnitude | 8 | 0.0613 | -0.0582 | 0.1808 | 0.0153 | 76.4665 | -0.2657 | 0.3883 |

差值位于 Fisher-z 尺度，不能当成原始 Δrho。均值区间使用modified Knapp–Hartung；prediction interval用于表达未来研究可能落入的范围。

### Absolute分数本身

| effect | pooled_rho_equivalent | ci95_low_rho_equivalent | ci95_high_rho_equivalent | prediction_low_rho_equivalent | prediction_high_rho_equivalent | tau2_reml | I2_percent |
|---|---|---|---|---|---|---|---|
| baseline_predicted_magnitude | 0.1484 | 0.0210 | 0.2711 | -0.2057 | 0.4682 | 0.0185 | 80.6088 |
| risk_model_disagreement | 0.0535 | -0.0446 | 0.1507 | -0.1974 | 0.2979 | 0.0090 | 69.8403 |
| safeconf_calibrated_pair_risk | 0.1943 | 0.1244 | 0.2622 | 0.0410 | 0.3386 | 0.0031 | 57.9088 |

## LODO

| effect | removed_dataset | pooled_z | ci95_low_z | ci95_high_z | prediction_low_z | prediction_high_z |
|---|---|---|---|---|---|---|
| safeconf_minus_disagreement | Frangieh | 0.1444 | 0.0592 | 0.2296 | -0.0494 | 0.3383 |
| safeconf_minus_disagreement | Lara_exvivo | 0.1251 | 0.0521 | 0.1981 | -0.0296 | 0.2798 |
| safeconf_minus_disagreement | Liang | 0.1401 | 0.0604 | 0.2198 | -0.0440 | 0.3242 |
| safeconf_minus_disagreement | Nadig_two_cellline | 0.1619 | 0.1104 | 0.2134 | 0.1077 | 0.2160 |
| safeconf_minus_disagreement | Replogle_two_cellline | 0.1286 | 0.0518 | 0.2053 | -0.0397 | 0.2969 |
| safeconf_minus_disagreement | Santinha | 0.1371 | 0.0619 | 0.2123 | -0.0368 | 0.3110 |
| safeconf_minus_disagreement | Shifrut | 0.1419 | 0.0628 | 0.2210 | -0.0410 | 0.3248 |
| safeconf_minus_disagreement | Tian_CRISPRi | 0.1380 | 0.0530 | 0.2231 | -0.0555 | 0.3316 |
| safeconf_minus_magnitude | Frangieh | 0.0523 | -0.0924 | 0.1970 | -0.3305 | 0.4351 |
| safeconf_minus_magnitude | Lara_exvivo | 0.0309 | -0.0834 | 0.1452 | -0.2528 | 0.3146 |
| safeconf_minus_magnitude | Liang | 0.0497 | -0.0896 | 0.1889 | -0.3211 | 0.4204 |
| safeconf_minus_magnitude | Nadig_two_cellline | 0.1012 | 0.0053 | 0.1972 | -0.1178 | 0.3203 |
| safeconf_minus_magnitude | Replogle_two_cellline | 0.0749 | -0.0608 | 0.2106 | -0.2854 | 0.4352 |
| safeconf_minus_magnitude | Santinha | 0.0507 | -0.0855 | 0.1868 | -0.3150 | 0.4163 |
| safeconf_minus_magnitude | Shifrut | 0.0726 | -0.0633 | 0.2084 | -0.2901 | 0.4352 |
| safeconf_minus_magnitude | Tian_CRISPRi | 0.0593 | -0.0867 | 0.2053 | -0.3283 | 0.4470 |

主 estimand 的 LODO 未发生合并效应符号反转。LODO只能说明现有八研究中单个研究的影响，不能增加研究数量。

## Pooled-median sensitivity

| effect | k | pooled_z | ci95_low_z | ci95_high_z | tau2_reml | I2_percent | prediction_low_z | prediction_high_z |
|---|---|---|---|---|---|---|---|---|
| safeconf_minus_disagreement | 8 | -0.0271 | -0.2296 | 0.1754 | 0.0518 | 90.5993 | -0.6223 | 0.5681 |
| safeconf_minus_magnitude | 8 | -0.0379 | -0.1900 | 0.1141 | 0.0250 | 80.8175 | -0.4554 | 0.3795 |

该敏感性先把同一context-task跨fold取中位数，再做研究内pooled Spearman；它改变了E140的fold-macro estimand。bootstrap仍以perturbation为簇同步全部context，结果不能覆盖上面的主分析。

<!-- E153_POSTFREEZE_SCALE_AUDIT_START -->
### Pooled 敏感性的独立复核：Replogle 尺度混排

该敏感性中 SafeConf−disagreement（z=-0.0271）和 SafeConf−magnitude
（z=-0.0379）均为负，**不支持主分析结论**。删除 Replogle 后的结果为：

| effect | removed_dataset | pooled_z | ci95_low_z | ci95_high_z | prediction_low_z | prediction_high_z |
|---|---|---|---|---|---|---|
| safeconf_minus_disagreement | Replogle_two_cellline | 0.0464 | -0.0666 | 0.1594 | -0.2107 | 0.3035 |
| safeconf_minus_magnitude | Replogle_two_cellline | 0.0170 | -0.0916 | 0.1257 | -0.1883 | 0.2224 |

两项删除 Replogle 后均由负转正。Replogle 的逐细胞系诊断为：

| context | fold_id | n_tasks | mean_error | mean_calibrated_safeconf | within_fold_spearman | pooled_cross_context_spearman |
|---|---|---|---|---|---|---|
| K562 | Replogle_cellline_holdout_1_K562 | 128 | 0.0882 | 0.6541 | 0.1583 | -0.2669 |
| RPE1 | Replogle_cellline_holdout_2_RPE1 | 128 | 0.1369 | 0.1125 | 0.1870 | -0.2669 |

K562 与 RPE1 各自只出现在一个 fold。两折内部 SafeConf–误差相关均为正；但两个 fold 的
原始校准分数均值和误差均值方向相反，跨 context 直接排序后 rho=-0.2669。这是
Simpson 反转，不是字段错配。校准 SafeConf 属于 fold 特异尺度，不能把不同 fold 的原始值
直接当作同一量尺混排。

因此 pooled-median 结果应视作尺度混排诊断：它不能支持主结论，也不能覆盖或否定合同预定的
fold-macro 主分析。数值复核见 `tables/E153_REPLOGLE_SCALE_MIXING_AUDIT.csv`。
<!-- E153_POSTFREEZE_SCALE_AUDIT_END -->

## Directional-SafeConf：Nadig与Replogle

### 分研究fold-macro

| dataset | analysis | effect | n_independent_perturbation_clusters | fold_macro_spearman | ci95_low | ci95_high |
|---|---|---|---|---|---|---|
| Nadig_two_cellline | score_association | directional_risk_frozen | 96 | 0.7535 | 0.6894 | 0.8036 |
| Nadig_two_cellline | score_association | baseline_predicted_magnitude | 96 | 0.2913 | 0.1753 | 0.3859 |
| Nadig_two_cellline | directional_minus_comparator | directional_minus_baseline_predicted_magnitude | 96 | 0.4622 | 0.3691 | 0.5655 |
| Replogle_two_cellline | score_association | directional_risk_frozen | 128 | 0.1905 | 0.0682 | 0.3084 |
| Replogle_two_cellline | score_association | baseline_predicted_magnitude | 128 | 0.2081 | 0.0966 | 0.3172 |
| Replogle_two_cellline | directional_minus_comparator | directional_minus_baseline_predicted_magnitude | 128 | -0.0175 | -0.1635 | 0.1334 |

### 固定两研究描述性合并

| analysis | effect | k_studies | equal_study_mean_spearman | minimum_study_spearman | maximum_study_spearman | fixed_two_study_bootstrap_ci95_low | fixed_two_study_bootstrap_ci95_high |
|---|---|---|---|---|---|---|---|
| score_association | directional_risk_frozen | 2 | 0.4720 | 0.1905 | 0.7535 | 0.4024 | 0.5355 |
| score_association | baseline_predicted_magnitude | 2 | 0.2497 | 0.2081 | 0.2913 | 0.1695 | 0.3167 |
| directional_minus_comparator | directional_minus_baseline_predicted_magnitude | 2 | 0.2223 | -0.0175 | 0.4622 | 0.1357 | 0.3143 |

这里的区间只条件于Nadig和Replogle这两个已观察研究。k=2不进行REML、Knapp–Hartung、I²或prediction interval，也不声称已经获得跨研究稳定保证。Replogle两个细胞系来自同一研究且目标control可见，其证据范围仍是control-observed跨细胞系复制。

## 审计边界

- E153没有重新训练、重新打分、换任务或换端点。
- E152 frozen score SHA-256：`sha256:611ac06b5630b33dd3e2f16e62f5a1c5bd903d1cb895ce69ee97e56578011b42`；E135 frozen model SHA-256：`sha256:77caf3b7b46071ced9577a8bd5289ce4c7bf5899c329ab37e835c41bda07d4b3`；两者均验证通过。E151 strict issue count：`0`。
- 八研究平均效应、异质性和LODO不能保证未来研究、期刊录用或湿实验机制验证。
