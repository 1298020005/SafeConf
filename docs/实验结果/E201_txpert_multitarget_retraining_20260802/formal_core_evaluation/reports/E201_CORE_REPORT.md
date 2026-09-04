# E201 四目标四种子核心评价

- 执行完整性：**PASS**。
- family-error 代数证书：**FAIL**。
- 预测前经验路由：**SUPPORTED**。
- 相对 predicted magnitude 的增量：**SUPPORTED**。

## 证书复核

1,808 个主任务和 200 个敏感性任务均已重算。恒等式最大绝对残差为 `3.6107837e-10`，family RMS 低于 disagreement 的任务数为 `0`。证书只说明四种子分歧是 family RMS error 的确定性下界，不替代经验排序结果。

## 主关联

| predictor | estimate | ci95_lower | ci95_upper |
| --- | --- | --- | --- |
| safeconf_e201_risk | 0.4082 | 0.3506 | 0.4621 |
| predicted_magnitude | 0.6189 | 0.5747 | 0.6579 |
| family_disagreement | 0.2402 | 0.1812 | 0.2974 |

## 控制 predicted magnitude

| predictor | covariate | estimate | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- |
| safeconf_e201_risk | predicted_magnitude | 0.2503 | 0.2021 | 0.2980 |

## 固定 20% 复核预算

| predictor | n_selected | high_error_capture | error_lift | oracle_normalized_utility | utility_ci95_lower | utility_ci95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| safeconf_e201_risk | 362 | 0.3343 | 1.1925 | 0.3200 | 0.2456 | 0.3929 |
| predicted_magnitude | 362 | 0.4558 | 1.3576 | 0.5943 | 0.5386 | 0.6525 |

## SafeConf 相对 magnitude 的配对增量

| measure | estimate | ci95_lower | ci95_upper |
| --- | --- | --- | --- |
| delta_oracle_normalized_utility | -0.2743 | -0.3576 | -0.2008 |
| delta_spearman | -0.2106 | -0.2651 | -0.1574 |

## family centroid 与简单基线

| baseline | predictor_mean_error | baseline_mean_error | task_win_rate | mean_delta | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| official_general_baseline | 0.0611 | 0.0637 | 0.9121 | -0.0026 | -0.0028 | -0.0024 |
| batch_matched_control | 0.0611 | 0.0696 | 0.7561 | -0.0085 | -0.0093 | -0.0076 |
| source_transfer | 0.0611 | 0.0640 | 0.7235 | -0.0029 | -0.0032 | -0.0026 |

## 四个 target

| scope | n_tasks | estimate | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- |
| K562 | 566 | 0.4803 | 0.4069 | 0.5467 |
| RPE1 | 416 | 0.2868 | 0.1927 | 0.3740 |
| hepg2 | 405 | 0.5633 | 0.4858 | 0.6305 |
| jurkat | 421 | 0.4453 | 0.3681 | 0.5209 |

| target | n_tasks | family_centroid_rmse_mean | family_rms_error_mean | worst_seed_error_mean | control_error_mean | official_general_baseline_error_mean | source_transfer_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K562 | 566 | 0.0540 | 0.0543 | 0.0560 | 0.0523 | 0.0584 | 0.0593 |
| RPE1 | 416 | 0.0788 | 0.0790 | 0.0795 | 0.0959 | 0.0801 | 0.0786 |
| hepg2 | 405 | 0.0576 | 0.0577 | 0.0583 | 0.0697 | 0.0598 | 0.0590 |
| jurkat | 421 | 0.0564 | 0.0566 | 0.0571 | 0.0666 | 0.0583 | 0.0608 |

## 正式门

| gate | passed | observed | criterion | gate_type |
| --- | --- | --- | --- | --- |
| input_integrity | True | pretruth_max=3.4973438e-09;family_mean_max=4.7683716e-07;official_general_equivalence=2.7865171e-06 | 2,008 aligned tasks; pretruth residual <5e-6 | execution |
| family_error_certificate | False | identity_max=3.6107837e-10;lower_bound_violations=0 | identity residual <=1e-10; zero lower-bound violations | deterministic_certificate |
| empirical_routing | True | rho_lower=0.35063625;utility_lower=0.24558702 | both pooled 95% cluster-bootstrap lower bounds >0 | scientific |
| incremental_vs_magnitude | True | partial_lower=0.20206962;delta_utility_lower=-0.35759519 | either paired 95% cluster-bootstrap lower bound >0 | scientific |
| target_reporting_complete | True | K562,RPE1,hepg2,jurkat | all four targets reported without directional filtering | reporting |

## 解释边界

E201 检查公开 TxPert STRING-GAT 在四个跨细胞背景目标上的四种子重训练family。任何未通过的科学门均保留为 NOT_SUPPORTED；不会因 pooled 或某个target 的方向不利而删除任务，也不会把代数证书写成对所有模型、扰动模态或数据集的经验有效性。scPertEval 五端点在独立补充程序中运行。
