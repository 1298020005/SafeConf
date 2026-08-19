# E84｜CPA-RDKit 化学四象限正式复核

E84 在 E81 的 8 个未查看 manifest 上固定运行 CPA 0.8.8 + RDKit Morgan embedding。E83 开发时查看过的 `E81_r1_p75` 永久排除。8 个正式运行统一使用 20 epochs、相同网络、log10-dose、相同细胞抽样上限和相同评价合同。

- manifest：8
- manifest-task：629
- strict PredictionRecord：1258
- strict issues：0
- pair-mean 三角下界：629/629
- pair-max 三角下界：629/629

## 分象限描述

| quadrant | n_manifests | mean_spearman_disagreement | disagreement_manifest_bootstrap_ci95_low | disagreement_manifest_bootstrap_ci95_high | mean_spearman_magnitude | magnitude_manifest_bootstrap_ci95_low | magnitude_manifest_bootstrap_ci95_high | mean_delta_disagreement_minus_magnitude | delta_manifest_bootstrap_ci95_low | delta_manifest_bootstrap_ci95_high | positive_disagreement_manifests | positive_delta_manifests |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| new_context_new_perturbation | 8 | 0.769 | 0.612 | 0.901 | 0.75 | 0.603 | 0.879 | 0.019 | -0.01 | 0.05 | 8 | 4 |
| new_context_seen_perturbation | 8 | 0.779 | 0.705 | 0.849 | 0.843 | 0.786 | 0.897 | -0.063 | -0.098 | -0.036 | 8 | 0 |
| seen_context_new_perturbation | 8 | 0.336 | 0.145 | 0.509 | 0.566 | 0.518 | 0.613 | -0.229 | -0.383 | -0.08 | 7 | 2 |
| seen_context_seen_perturbation_pair_holdout | 8 | 0.738 | 0.569 | 0.85 | 0.631 | 0.414 | 0.81 | 0.106 | 0.016 | 0.207 | 8 | 5 |

分歧在四个象限的平均相关都为正。相对 predicted magnitude 的增量只在随机缺失 pair 中稳定为正；新 context 和新药两个难象限由 magnitude 稳定占优，双未见的增量区间跨 0。这个结果支持“分歧能排序部分 pair risk”，不支持“它在所有难 setting 中优于简单幅度基线”。

区间通过 manifest 重采样得到，只表示同一 sciPlex3 数据内不同冻结 split 的敏感性，不能写成 8 个独立数据集的外部置信区间。所有负 delta 和失败象限保留。
