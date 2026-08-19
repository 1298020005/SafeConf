# E196｜CPA 原生潜空间支持距离审计

E196 严格加载 E94 开发适配器和 E84 八个 formal manifest 的已有 CPA 权重，
不重训模型，也不修改任何 reference set。这里的 CPA 原生分数是目标
`cell line + dose-weighted drug` 潜向量到训练条件潜向量的最近距离，不是预测方差、
校准概率或误差下界。

- 证据标签：`POSTTRUTH_DIRECT_COMPETITOR_AUDIT`
- formal manifest：8
- formal manifest-task：629
- 开发适配器任务：59
- reference set：无阈值显式 train、官方 `>30` 敏感性、perturbed-train-only
- invariant failures：0

## 同一 CPA outcome 的主结果

| score_name | metric | n_manifests_estimable | manifest_equal_macro | manifest_bootstrap_ci95_low | manifest_bootstrap_ci95_high |
| --- | --- | --- | --- | --- | --- |
| CPA cosine support distance | spearman | 8 | 0.3899 | 0.3169 | 0.4648 |
| CPA Euclidean support distance | spearman | 8 | 0.3455 | 0.2814 | 0.4214 |
| CPA predicted magnitude | spearman | 8 | 0.5906 | 0.4478 | 0.7158 |
| CPA cosine support distance | normalized_aurc_50_100 | 8 | 0.9666 | 0.9572 | 0.9767 |
| CPA cosine support distance | oracle_normalized_utility@0.20 | 8 | 0.4277 | 0.2923 | 0.5660 |
| CPA Euclidean support distance | normalized_aurc_50_100 | 8 | 0.9726 | 0.9645 | 0.9797 |
| CPA Euclidean support distance | oracle_normalized_utility@0.20 | 8 | 0.3654 | 0.2342 | 0.5192 |
| CPA predicted magnitude | normalized_aurc_50_100 | 8 | 0.9515 | 0.9352 | 0.9673 |
| CPA predicted magnitude | oracle_normalized_utility@0.20 | 8 | 0.5816 | 0.4157 | 0.7276 |

## support distance 相对 magnitude 的配对 ΔSpearman

正值表示 support distance 的相关更高。表中并列给出 manifest-resampling
描述性区间，以及保持同一“manifest 内计算、manifest 间等权”估计量的
biological-task cluster 描述性区间。八个 manifest 共享任务且不是 iid 抽样，
10,000 只是 Monte Carlo 重采样次数，不增加有效样本量。

| score_name | point_delta | manifest_interval_low | manifest_interval_high | n_manifests_estimable | task_cluster_interval_low | task_cluster_interval_high | n_unique_task_clusters | point_matches_manifest_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native_cosine_distance | -0.2007 | -0.3194 | -0.0617 | 8 | -0.3579 | -0.0467 | 103.0000 | True |
| native_euclidean_distance | -0.2451 | -0.4064 | -0.0765 | 8 | -0.3987 | -0.0920 | 103.0000 | True |

## 参考条件与 control 最近邻

| manifest_id | n_reference_all | n_reference_gt30 | n_reference_perturbed | n_control_nearest_cosine_all | n_control_nearest_euclidean_all |
| --- | --- | --- | --- | --- | --- |
| E81_r1_p75 | 38 | 38 | 35 | 38 | 38 |
| E81_r1_p25 | 11 | 11 | 8 | 57 | 56 |
| E81_r1_p50 | 24 | 24 | 21 | 41 | 41 |
| E81_r2_p25 | 10 | 9 | 7 | 65 | 66 |
| E81_r2_p50 | 23 | 23 | 20 | 38 | 37 |
| E81_r2_p75 | 37 | 36 | 34 | 43 | 44 |
| E81_r3_p25 | 11 | 11 | 8 | 64 | 64 |
| E81_r3_p50 | 22 | 22 | 19 | 54 | 53 |
| E81_r3_p75 | 37 | 37 | 34 | 40 | 40 |

E94 只做模型重建、任务连接和预测复现检查，不进入 formal 宏平均。所有四个
Cartesian quadrant、负结果、constant-score 标记和 reference-set 敏感性均保留。
CPA–ridge disagreement 对 pair-mean RMSE 作为不同 predictor family 的补充，
没有与 CPA 自身 RMSE 的 head-to-head 结果混写。
