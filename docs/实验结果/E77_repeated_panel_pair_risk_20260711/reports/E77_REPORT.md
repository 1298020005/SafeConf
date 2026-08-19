# E77｜模型家族分歧的 pair-risk 理论证书

设两个模型在同一任务上的预测效应为 $p_1,p_2$，真实效应为 $y$，$d$ 为 RMSE 距离。三角不等式直接给出：

$$\max\{d(p_1,y),d(p_2,y)\}\geq \frac{1}{2}d(p_1,p_2),$$

$$\frac{d(p_1,y)+d(p_2,y)}{2}\geq \frac{1}{2}d(p_1,p_2).$$

这个下界只用两个预测向量。它证明高分歧时至少一个模型存在相应规模的错误，也明确了它对应的是 pair mean/max risk，不能据此判断 GEARS 或 scGPT 谁错。

平方误差下还有精确恒等式：

$$\frac{\mathrm{MSE}(p_1,y)+\mathrm{MSE}(p_2,y)}{2}=\mathrm{MSE}\left(\frac{p_1+p_2}{2},y\right)+\frac{1}{4}\mathrm{MSE}(p_1,p_2).$$

144 个真实任务中，三角下界违反数为 0；平方误差恒等式最大绝对数值误差为 1.475e-17。

## 三数据集×两套不重叠面板的分层关联

| score | target | 平均分层 ρ | task-bootstrap 95% CI | 分层置换 p |
|---|---|---:|---:|---:|
| model_disagreement | GEARS_RMSE | 0.459 | [0.306, 0.599] | 0.0001 |
| model_disagreement | scGPT_RMSE | 0.420 | [0.273, 0.557] | 0.0001 |
| model_disagreement | pair_mean_RMSE | 0.489 | [0.348, 0.617] | 0.0001 |
| model_disagreement | pair_max_RMSE | 0.552 | [0.417, 0.670] | 0.0001 |
| magnitude_mean_rank | GEARS_RMSE | 0.340 | [0.186, 0.477] | 0.0001 |
| magnitude_mean_rank | scGPT_RMSE | 0.238 | [0.079, 0.388] | 0.0019 |
| magnitude_mean_rank | pair_mean_RMSE | 0.307 | [0.149, 0.452] | 0.0001 |
| magnitude_mean_rank | pair_max_RMSE | 0.341 | [0.187, 0.487] | 0.0001 |
| magnitude_max_rank | GEARS_RMSE | 0.341 | [0.188, 0.477] | 0.0002 |
| magnitude_max_rank | scGPT_RMSE | 0.233 | [0.089, 0.379] | 0.0025 |
| magnitude_max_rank | pair_mean_RMSE | 0.322 | [0.165, 0.459] | 0.0001 |
| magnitude_max_rank | pair_max_RMSE | 0.355 | [0.203, 0.492] | 0.0002 |

## 分歧相对固定 magnitude 聚合

| target | baseline | Δρ | bootstrap 95% CI |
|---|---|---:|---:|
| pair_mean_RMSE | magnitude_mean_rank | 0.182 | [0.062, 0.310] |
| pair_mean_RMSE | magnitude_max_rank | 0.167 | [0.050, 0.290] |
| pair_max_RMSE | magnitude_mean_rank | 0.212 | [0.087, 0.342] |
| pair_max_RMSE | magnitude_max_rank | 0.197 | [0.075, 0.320] |

## 使用边界

分歧证书是下界，不是误差上界，也不是概率校准。低分歧时两个模型仍可能一起犯错；高分歧时能够确定 pair-level 风险，但不能仅凭分歧定位错误模型。SafeConf 的学习部分只负责把这个证书与任务新颖性、支持度校准到具体筛选预算。
