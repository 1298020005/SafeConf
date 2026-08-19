# E96｜PRESCRIBE 原生不确定性双面板对照

Norman P1、P2 各包含 24 个事先冻结且互不重叠的单基因测试任务。PRESCRIBE 的认知不确定性、数据不确定性和论文组合分数只与 PRESCRIBE 自身误差比较；任务平均表达谱 RMSE 是主目标，逐细胞逐基因 RMSE 是补充目标。预测幅度使用同一 PRESCRIBE 输出和训练侧可得的 control 均值。

## 双面板主结果

| score | spearman | bootstrap_ci95_low | bootstrap_ci95_high |
|---|---|---|---|
| PRESCRIBE epistemic | -0.053 | -0.36 | 0.258 |
| PRESCRIBE aleatoric | -0.165 | -0.441 | 0.128 |
| PRESCRIBE combined | -0.056 | -0.364 | 0.261 |
| PRESCRIBE magnitude | 0.059 | -0.245 | 0.372 |

## 相对自身 magnitude 基线

| score | target | delta_rho_vs_magnitude | bootstrap_ci95_low | bootstrap_ci95_high |
|---|---|---|---|---|
| PRESCRIBE epistemic | task_mean_profile_rmse | -0.113 | -0.723 | 0.505 |
| PRESCRIBE aleatoric | task_mean_profile_rmse | -0.224 | -0.756 | 0.318 |
| PRESCRIBE combined | task_mean_profile_rmse | -0.115 | -0.731 | 0.495 |

两套面板上没有一种 PRESCRIBE 原生不确定性与自身任务误差形成稳定正相关，三种分数相对自身 magnitude 的 Δρ 也均为负且区间跨 0。选择性路由同样没有稳定收益：P2 的 aleatoric 分数在拒绝 20% 任务后，剩余误差反而增加约 10.4%。这说明原生不确定性在当前未见单基因 setting 中不能直接当作可靠质检分数。

`E96_SIDE_BY_SIDE_DIFFERENT_TARGETS.csv` 同时列出相同任务上的 GEARS–scGPT 分歧结果。两类方法对应不同预测器和不同误差，只作并列展示，不把相关系数混成一次直接胜负检验。拒绝 10%、20%、30% 的结果和 50%–100% coverage 曲线分别保存在 `E96_ROUTING_METRICS.csv` 与 `E96_RISK_COVERAGE.csv`。

E96 关闭了“缺少直接不确定性竞品”的工程缺口，但不能据此宣称 SafeConf 全面优于 PRESCRIBE。当前能够比较的是各自分数对各自误差的排序能力；SafeConf 的可写优势仍限定在异构预测器的 post-hoc pair-risk 下界和不改造原预测模型，不能扩写成单模型概率校准。
