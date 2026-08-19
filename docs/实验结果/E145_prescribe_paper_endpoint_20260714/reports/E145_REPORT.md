# E145｜PRESCRIBE 论文终点口径纠正

## 结论

PRESCRIBE 原论文用预测与真实扰动效应的 Pearson 相关作为置信度校准的默认准确度终点。E145 按这一口径重新分析已有 E95 Norman P1/P2 formal 输出。组合置信度与 Pearson 准确度的 Spearman 为：P1 `0.0496`，P2 `0.5843`，双面板等权宏平均 `0.3170`（任务 bootstrap 95% CI `0.0324` 至 `0.5669`）。组合置信度在两个面板同向，且双面板宏平均区间高于 0。

同一终点上，predicted magnitude 的双面板宏 Spearman 为 `0.3104`。组合置信度相对 magnitude 的 Δρ 为 `0.0065`（95% CI `-0.0317` 至 `0.0534`）。组合置信度相对 magnitude 的增量区间未高于 0，不能宣称独立排序增益。

组合置信度与 magnitude 的面板内 Spearman 分别为 P1 `0.9965`、P2 `0.9939`。因此，E96 中“PRESCRIBE 在当前 setting 完全没有可靠性信号”的表述需要收窄；E145 能回答的是论文定义的方向准确度信号，不能据此宣称它稳定超过幅度基线。

## 论文口径核对

论文式（5）定义 `pseudo E-distance = 2 × normalized posterior evidence − normalized predictive entropy`。官方测试代码对每个扰动取 `epistemic_conf` 和 `aleatoric_conf` 的细胞均值，再计算 `2 × epistemic + aleatoric`。论文第 4.2 节把置信度校准默认定义为置信度与预测—真实 log-fold-change Pearson 准确度之间的相关。E145 的预测效应和真实效应均由同一 log-normalized 表达空间减去同一 control 均值得到。

来源：<https://papers.nips.cc/paper_files/paper/2025/file/d6383e7643415842b48a5077a1b09c98-Paper-Conference.pdf>。

## Pearson 主终点

| scope | score | spearman_rho | bootstrap_ci95_low | bootstrap_ci95_high |
|---|---|---|---|---|
| Norman_P1 | epistemic_confidence | 0.0478 | -0.3946 | 0.4769 |
| Norman_P1 | aleatoric_confidence | 0.0878 | -0.3471 | 0.5312 |
| Norman_P1 | combined_confidence | 0.0496 | -0.3918 | 0.4775 |
| Norman_P1 | predicted_magnitude | 0.047 | -0.3962 | 0.4799 |
| Norman_P2 | epistemic_confidence | 0.5757 | 0.1914 | 0.8322 |
| Norman_P2 | aleatoric_confidence | -0.0739 | -0.505 | 0.3491 |
| Norman_P2 | combined_confidence | 0.5843 | 0.207 | 0.8446 |
| Norman_P2 | predicted_magnitude | 0.5739 | 0.2133 | 0.8243 |
| two_panel_macro | epistemic_confidence | 0.3117 | 0.0194 | 0.5688 |
| two_panel_macro | aleatoric_confidence | 0.007 | -0.3004 | 0.3157 |
| two_panel_macro | combined_confidence | 0.317 | 0.0324 | 0.5669 |
| two_panel_macro | predicted_magnitude | 0.3104 | 0.024 | 0.5593 |

## 相对 predicted magnitude 的增量

| score | raw_delta_rho | raw_bootstrap_ci95_low | raw_bootstrap_ci95_high |
|---|---|---|---|
| epistemic_confidence | 0.0013 | -0.042 | 0.048 |
| aleatoric_confidence | -0.3035 | -0.5271 | -0.0768 |
| combined_confidence | 0.0065 | -0.0317 | 0.0534 |

## 与 predicted magnitude 的排序重合

| scope | score | spearman_rho | bootstrap_ci95_low | bootstrap_ci95_high |
|---|---|---|---|---|
| Norman_P1 | epistemic_confidence | 0.9939 | 0.9631 | 1.0 |
| Norman_P2 | epistemic_confidence | 0.9948 | 0.9625 | 1.0 |
| two_panel_macro | epistemic_confidence | 0.9943 | 0.9725 | 0.9991 |
| Norman_P1 | aleatoric_confidence | 0.9765 | 0.9073 | 0.9938 |
| Norman_P2 | aleatoric_confidence | 0.1478 | -0.3928 | 0.6212 |
| two_panel_macro | aleatoric_confidence | 0.5622 | 0.2863 | 0.7908 |
| Norman_P1 | combined_confidence | 0.9965 | 0.9737 | 1.0 |
| Norman_P2 | combined_confidence | 0.9939 | 0.9596 | 1.0 |
| two_panel_macro | combined_confidence | 0.9952 | 0.9746 | 0.9996 |

## 论文式 5%/10% 过滤的回顾性比较

下表是置信度过滤与 magnitude 过滤在同一 coverage 下的保留集平均 Pearson 差值；正值有利于置信度。

| score | coverage | raw_delta_retained_mean | raw_bootstrap_ci95_low | raw_bootstrap_ci95_high |
|---|---|---|---|---|
| epistemic_confidence | 0.95 | 0.0 | -0.0021 | 0.0 |
| epistemic_confidence | 0.9 | 0.0 | -0.0024 | 0.0 |
| aleatoric_confidence | 0.95 | -0.0159 | -0.0383 | -0.0086 |
| aleatoric_confidence | 0.9 | -0.0399 | -0.0529 | -0.01 |
| combined_confidence | 0.95 | 0.0 | -0.0021 | 0.0 |
| combined_confidence | 0.9 | 0.0 | -0.0022 | 0.0 |

## 边界

- E145 使用已查看真实结果的 P1/P2 数据，只是 post-unblinding metric correction，不是独立确认。
- 过滤曲线在相同测试任务上排序并评价，只能描述回顾性选择性表现。
- cosine、RMSE 和所有逐任务数据保存在 `tables/`，没有因结果删除任务。
- E145 不把 SafeConf 与 PRESCRIBE 混成同预测器的直接比较，也不改动 E96 已报告的 RMSE 结果。
