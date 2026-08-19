# E111｜SafeConf 实际在筛哪类错误

E108 的主目标仍是两模型平均 RMSE。E111 沿用 E98 已经列出的模型别误差与最坏误差作为次要终点，解释信号来源，不回调风险分数。

scGPT 平均 RMSE 为 0.0526，GEARS 为 0.0566。

## 3-fold 宏平均

| target | score | ρ | RC80 改善 |
|---|---|---:|---:|
| error_gears_rmse | baseline_predicted_magnitude | 0.254 | 3.24% |
| error_gears_rmse | risk_model_disagreement | 0.248 | 3.30% |
| error_gears_rmse | safeconf_calibrated_pair_risk | 0.397 | 2.30% |
| error_gears_rmse | safeconf_frozen_pair_risk | 0.385 | 2.74% |
| error_scgpt_rmse | baseline_predicted_magnitude | 0.001 | 1.44% |
| error_scgpt_rmse | risk_model_disagreement | -0.014 | 1.42% |
| error_scgpt_rmse | safeconf_calibrated_pair_risk | 0.071 | 0.84% |
| error_scgpt_rmse | safeconf_frozen_pair_risk | 0.057 | 0.87% |
| error_two_predictor_max_rmse | baseline_predicted_magnitude | 0.243 | 3.35% |
| error_two_predictor_max_rmse | risk_model_disagreement | 0.234 | 3.40% |
| error_two_predictor_max_rmse | safeconf_calibrated_pair_risk | 0.364 | 2.39% |
| error_two_predictor_max_rmse | safeconf_frozen_pair_risk | 0.357 | 2.82% |
| error_two_predictor_mean_rmse | baseline_predicted_magnitude | 0.148 | 2.37% |
| error_two_predictor_mean_rmse | risk_model_disagreement | 0.137 | 2.39% |
| error_two_predictor_mean_rmse | safeconf_calibrated_pair_risk | 0.253 | 1.60% |
| error_two_predictor_mean_rmse | safeconf_frozen_pair_risk | 0.242 | 1.84% |

## 解释

SafeConf 对 GEARS 误差和双模型最坏误差的排序明显强于对 scGPT 误差。分歧在这里更像“较弱预测器偏离较强预测器”的路由信号，而不是与模型无关的通用置信度。论文应把适用对象写成任务/预测器组合，并报告模型别结果。

## 聚类 bootstrap

| target | comparator | Δρ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---:|
| error_two_predictor_mean_rmse | safeconf_frozen_pair_risk | 0.011 | [-0.028, 0.055] | 0.682 |
| error_two_predictor_mean_rmse | risk_model_disagreement | 0.116 | [-0.029, 0.278] | 0.931 |
| error_two_predictor_mean_rmse | baseline_predicted_magnitude | 0.105 | [-0.042, 0.247] | 0.906 |
| error_two_predictor_max_rmse | safeconf_frozen_pair_risk | 0.007 | [-0.032, 0.051] | 0.623 |
| error_two_predictor_max_rmse | risk_model_disagreement | 0.130 | [-0.003, 0.277] | 0.971 |
| error_two_predictor_max_rmse | baseline_predicted_magnitude | 0.120 | [-0.012, 0.248] | 0.959 |
| error_gears_rmse | safeconf_frozen_pair_risk | 0.012 | [-0.029, 0.060] | 0.690 |
| error_gears_rmse | risk_model_disagreement | 0.149 | [0.014, 0.293] | 0.988 |
| error_gears_rmse | baseline_predicted_magnitude | 0.143 | [0.007, 0.275] | 0.981 |
| error_scgpt_rmse | safeconf_frozen_pair_risk | 0.013 | [-0.025, 0.053] | 0.763 |
| error_scgpt_rmse | risk_model_disagreement | 0.085 | [-0.068, 0.271] | 0.814 |
| error_scgpt_rmse | baseline_predicted_magnitude | 0.070 | [-0.082, 0.243] | 0.776 |
