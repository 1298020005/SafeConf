# E194 注册家族构成与治理压力测试报告

状态：**PASS**

证据标签：`POSTTRUTH_FAMILY_GOVERNANCE_STRESS`。E194 复用已打开的 E190/E192
真值，不能算新的独立确认。

## 运行范围

- 数据：E190 K562 692 个任务；
  E192 RPE1 175 个任务；
- 几何：absolute RMSE、cosine、Pearson；
- 主 family：预真值冻结的 3 个 scGPT + 3 个 GEARS；
- 共评估 310 个
  `dataset×geometry×scenario`，逐任务记录
  134385 行。

## 确认性实现检查

- family lower-bound violations：0；
- worst-member lower-bound violations：0；
- 最大平方恒等式残差：6.661e-16；
- governance / C4 不变量检查：
  492/492 通过。

这些检查证明代码对每个声明 family 正确实现了证书；它们不证明任意 family 都有
同样好的经验排序。

`diversity_error_spearman` 以自身 family RMS 为结果，其中含有确定性的平方结构
耦合。表中同时给出固定 A0 family RMS 与 A0 centroid error，跨 family 解释以
固定结果列为准。

## absolute RMSE 关键场景

| dataset | target_family_id | mean_family_rms_error | mean_diversity_lower_bound | median_diversity_sq_over_family_sq | diversity_error_spearman | diversity_a0_family_error_spearman | diversity_a0_centroid_error_spearman | oracle_normalized_utility_at_20pct | a0_family_oracle_normalized_utility_at_20pct | relative_diversity_change_vs_a0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E190_K562 | A0_primary_balanced_3x3 | 0.24919 | 0.07331 | 0.07804 | 0.42388 | 0.42388 | 0.29341 | 0.44303 | 0.44303 | 0.00000 |
| E190_K562 | A1_scgpt_seed_only | 0.25826 | 0.02444 | 0.00971 | -0.07619 | -0.08054 | -0.08621 | -0.04016 | -0.04400 | -0.66661 |
| E190_K562 | A2_gears_seed_only | 0.23850 | 0.02256 | 0.00836 | 0.03149 | 0.15614 | 0.05632 | 0.08426 | 0.21936 | -0.69231 |
| E190_K562 | A4_architecture_centroids | 0.24800 | 0.06893 | 0.06709 | 0.42654 | 0.42921 | 0.29881 | 0.43853 | 0.44301 | -0.05978 |
| E190_K562 | B3_overweight_scgpt_flat | 0.25236 | 0.06977 | 0.07091 | 0.45763 | 0.42588 | 0.29547 | 0.47583 | 0.44303 | -0.04820 |
| E190_K562 | B3_overweight_scgpt_governed | 0.24919 | 0.07331 | 0.07804 | 0.42388 | 0.42388 | 0.29341 | 0.44303 | 0.44303 | 0.00000 |
| E190_K562 | C2_add_source_portfolio | 0.24758 | 0.07274 | 0.07869 | 0.39937 | 0.42095 | 0.28992 | 0.41753 | 0.44303 | -0.00769 |
| E190_K562 | C4_symmetric_attack_lambda4 | 0.37593 | 0.28293 | 0.54815 | 0.91913 | 0.42887 | 0.29851 | 0.96068 | 0.44301 | 2.85948 |
| E192_RPE1 | A0_primary_balanced_3x3 | 0.30327 | 0.04731 | 0.02802 | 0.29976 | 0.29976 | 0.28938 | 0.69577 | 0.69577 | 0.00000 |
| E192_RPE1 | A1_scgpt_seed_only | 0.29773 | 0.00752 | 0.00076 | 0.33208 | 0.32822 | 0.33023 | 0.26786 | 0.26241 | -0.84095 |
| E192_RPE1 | A2_gears_seed_only | 0.30862 | 0.01259 | 0.00192 | 0.17849 | 0.18521 | 0.18483 | 0.28935 | 0.27589 | -0.73393 |
| E192_RPE1 | A4_architecture_centroids | 0.30308 | 0.04613 | 0.02669 | 0.29300 | 0.29363 | 0.28324 | 0.68198 | 0.68204 | -0.02489 |
| E192_RPE1 | B3_overweight_scgpt_flat | 0.30144 | 0.04454 | 0.02518 | 0.29729 | 0.30137 | 0.29099 | 0.69787 | 0.69857 | -0.05841 |
| E192_RPE1 | B3_overweight_scgpt_governed | 0.30327 | 0.04731 | 0.02802 | 0.29976 | 0.29976 | 0.28938 | 0.69577 | 0.69577 | 0.00000 |
| E192_RPE1 | C2_add_source_portfolio | 0.30402 | 0.04705 | 0.02776 | 0.29959 | 0.29941 | 0.28904 | 0.69577 | 0.69577 | -0.00533 |
| E192_RPE1 | C4_symmetric_attack_lambda4 | 0.35648 | 0.18900 | 0.31520 | 0.42511 | 0.29429 | 0.28390 | 0.77704 | 0.68204 | 2.99542 |

## 基因整簇 bootstrap

| dataset | target_family_id | outcome | metric | point | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| E190_K562 | A0_primary_balanced_3x3 | own_family_rms | diversity_error_spearman | 0.42388 | 0.14011 | 0.63027 |
| E190_K562 | A0_primary_balanced_3x3 | own_family_rms | oracle_normalized_utility | 0.44303 | 0.02664 | 0.79322 |
| E190_K562 | A0_primary_balanced_3x3 | fixed_a0_family_rms | diversity_error_spearman | 0.42388 | 0.14004 | 0.62967 |
| E190_K562 | A0_primary_balanced_3x3 | fixed_a0_family_rms | oracle_normalized_utility | 0.44303 | 0.04557 | 0.80534 |
| E190_K562 | A0_primary_balanced_3x3 | fixed_a0_centroid_error | diversity_error_spearman | 0.29341 | -0.01366 | 0.52670 |
| E190_K562 | A0_primary_balanced_3x3 | fixed_a0_centroid_error | oracle_normalized_utility | 0.29674 | -0.12638 | 0.71684 |
| E190_K562 | A1_scgpt_seed_only | own_family_rms | diversity_error_spearman | -0.07619 | -0.17383 | -0.00683 |
| E190_K562 | A1_scgpt_seed_only | own_family_rms | oracle_normalized_utility | -0.04016 | -0.15861 | 0.07227 |
| E190_K562 | A1_scgpt_seed_only | fixed_a0_family_rms | diversity_error_spearman | -0.08054 | -0.17755 | -0.00891 |
| E190_K562 | A1_scgpt_seed_only | fixed_a0_family_rms | oracle_normalized_utility | -0.04400 | -0.16592 | 0.07057 |
| E190_K562 | A1_scgpt_seed_only | fixed_a0_centroid_error | diversity_error_spearman | -0.08621 | -0.18025 | -0.01291 |
| E190_K562 | A1_scgpt_seed_only | fixed_a0_centroid_error | oracle_normalized_utility | -0.04531 | -0.16642 | 0.07498 |
| E190_K562 | A2_gears_seed_only | own_family_rms | diversity_error_spearman | 0.03149 | -0.21297 | 0.30095 |
| E190_K562 | A2_gears_seed_only | own_family_rms | oracle_normalized_utility | 0.08426 | -0.31858 | 0.54356 |
| E190_K562 | A2_gears_seed_only | fixed_a0_family_rms | diversity_error_spearman | 0.15614 | -0.16542 | 0.45427 |
| E190_K562 | A2_gears_seed_only | fixed_a0_family_rms | oracle_normalized_utility | 0.21936 | -0.32288 | 0.75787 |
| E190_K562 | A2_gears_seed_only | fixed_a0_centroid_error | diversity_error_spearman | 0.05632 | -0.25627 | 0.35739 |
| E190_K562 | A2_gears_seed_only | fixed_a0_centroid_error | oracle_normalized_utility | 0.09387 | -0.37875 | 0.66644 |
| E190_K562 | A4_architecture_centroids | own_family_rms | diversity_error_spearman | 0.42654 | 0.15121 | 0.63712 |
| E190_K562 | A4_architecture_centroids | own_family_rms | oracle_normalized_utility | 0.43853 | 0.02928 | 0.80748 |
| E190_K562 | A4_architecture_centroids | fixed_a0_family_rms | diversity_error_spearman | 0.42921 | 0.15849 | 0.64046 |
| E190_K562 | A4_architecture_centroids | fixed_a0_family_rms | oracle_normalized_utility | 0.44301 | 0.02622 | 0.80718 |
| E190_K562 | A4_architecture_centroids | fixed_a0_centroid_error | diversity_error_spearman | 0.29881 | 0.00727 | 0.53452 |
| E190_K562 | A4_architecture_centroids | fixed_a0_centroid_error | oracle_normalized_utility | 0.29678 | -0.13372 | 0.71922 |
| E192_RPE1 | A0_primary_balanced_3x3 | own_family_rms | diversity_error_spearman | 0.29976 | -0.05201 | 0.57660 |
| E192_RPE1 | A0_primary_balanced_3x3 | own_family_rms | oracle_normalized_utility | 0.69577 | 0.11261 | 0.87430 |
| E192_RPE1 | A0_primary_balanced_3x3 | fixed_a0_family_rms | diversity_error_spearman | 0.29976 | -0.06182 | 0.58613 |
| E192_RPE1 | A0_primary_balanced_3x3 | fixed_a0_family_rms | oracle_normalized_utility | 0.69577 | 0.11881 | 0.87352 |
| E192_RPE1 | A0_primary_balanced_3x3 | fixed_a0_centroid_error | diversity_error_spearman | 0.28938 | -0.06225 | 0.57169 |
| E192_RPE1 | A0_primary_balanced_3x3 | fixed_a0_centroid_error | oracle_normalized_utility | 0.68806 | 0.11849 | 0.87011 |
| E192_RPE1 | A1_scgpt_seed_only | own_family_rms | diversity_error_spearman | 0.33208 | 0.10419 | 0.54097 |
| E192_RPE1 | A1_scgpt_seed_only | own_family_rms | oracle_normalized_utility | 0.26786 | 0.06231 | 0.59189 |
| E192_RPE1 | A1_scgpt_seed_only | fixed_a0_family_rms | diversity_error_spearman | 0.32822 | 0.10544 | 0.54173 |
| E192_RPE1 | A1_scgpt_seed_only | fixed_a0_family_rms | oracle_normalized_utility | 0.26241 | 0.05628 | 0.57488 |
| E192_RPE1 | A1_scgpt_seed_only | fixed_a0_centroid_error | diversity_error_spearman | 0.33023 | 0.10556 | 0.53554 |
| E192_RPE1 | A1_scgpt_seed_only | fixed_a0_centroid_error | oracle_normalized_utility | 0.26495 | 0.05772 | 0.57403 |
| E192_RPE1 | A2_gears_seed_only | own_family_rms | diversity_error_spearman | 0.17849 | -0.28737 | 0.55105 |
| E192_RPE1 | A2_gears_seed_only | own_family_rms | oracle_normalized_utility | 0.28935 | -0.25276 | 0.80368 |
| E192_RPE1 | A2_gears_seed_only | fixed_a0_family_rms | diversity_error_spearman | 0.18521 | -0.29803 | 0.55641 |
| E192_RPE1 | A2_gears_seed_only | fixed_a0_family_rms | oracle_normalized_utility | 0.27589 | -0.26252 | 0.79203 |
| E192_RPE1 | A2_gears_seed_only | fixed_a0_centroid_error | diversity_error_spearman | 0.18483 | -0.28520 | 0.56314 |
| E192_RPE1 | A2_gears_seed_only | fixed_a0_centroid_error | oracle_normalized_utility | 0.27215 | -0.26245 | 0.78547 |
| E192_RPE1 | A4_architecture_centroids | own_family_rms | diversity_error_spearman | 0.29300 | -0.06325 | 0.57268 |
| E192_RPE1 | A4_architecture_centroids | own_family_rms | oracle_normalized_utility | 0.68198 | 0.11407 | 0.87352 |
| E192_RPE1 | A4_architecture_centroids | fixed_a0_family_rms | diversity_error_spearman | 0.29363 | -0.07183 | 0.56974 |
| E192_RPE1 | A4_architecture_centroids | fixed_a0_family_rms | oracle_normalized_utility | 0.68204 | 0.11688 | 0.87493 |
| E192_RPE1 | A4_architecture_centroids | fixed_a0_centroid_error | diversity_error_spearman | 0.28324 | -0.06812 | 0.56789 |
| E192_RPE1 | A4_architecture_centroids | fixed_a0_centroid_error | oracle_normalized_utility | 0.67419 | 0.11517 | 0.87265 |

## 结果边界

1. A0、A1、A2、A4 是不同预测对象。某个子 family 的高相关不能替代 A0；
2. governed duplicate 场景恢复 A0，说明 lineage/架构权重合同能阻止复制成员增加
   话语权；
3. flat duplication 与 leave-one-out 量化 family 构成敏感性，必须如实保留；
4. absolute RMSE 的 C4 在 A0 质心误差不变时放大 diversity，直接否定“成员
   越多、分歧越大，证据越强”的写法；方向几何不创建离开球面的伪预测；
5. zero/source/synthetic 成员改变了 target family，只能作为负控或 portfolio
   分析，不能进入主证书。

完整组合范围见 `tables/E194_GROUP_RANGE_SUMMARY.csv`，逐任务证据见压缩 CSV。
