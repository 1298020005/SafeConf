# E69｜真实双模型风险校准器跨数据集迁移

Adamson→Norman 与 Norman→Adamson 两个方向都只在 source tasks 上选择 Ridge alpha、计算特征均值/方差和拟合系数。target truth 只在冻结预测之后计算 Spearman、MAE、top20 enrichment 与 risk–coverage。这里迁移的是风险校准器，不声称 perturbation predictor 本身已经 A→B 迁移。

## 目标域排序结果

| source→target | error | risk model | target ρ | target MAE | top20 enrichment |
|---|---|---|---:|---:|---:|
| Adamson→Norman | gears_ensemble_rmse | magnitude | -0.377 | 0.0271 | 1.100 |
| Adamson→Norman | gears_ensemble_rmse | magnitude_plus_disagreement | 0.708 | 0.0224 | 1.284 |
| Adamson→Norman | scgpt_finetuned_rmse | magnitude | 0.340 | 0.0369 | 1.139 |
| Adamson→Norman | scgpt_finetuned_rmse | magnitude_plus_disagreement | 0.355 | 0.0394 | 1.139 |
| Adamson→Norman | task_mean_rmse | magnitude | 0.434 | 0.0239 | 1.191 |
| Adamson→Norman | task_mean_rmse | magnitude_plus_disagreement | 0.547 | 0.0235 | 1.191 |
| Adamson→Norman | task_max_rmse | magnitude | 0.403 | 0.0297 | 1.132 |
| Adamson→Norman | task_max_rmse | magnitude_plus_disagreement | 0.543 | 0.0288 | 1.456 |
| Norman→Adamson | gears_ensemble_rmse | magnitude | 0.076 | 0.0515 | 0.937 |
| Norman→Adamson | gears_ensemble_rmse | magnitude_plus_disagreement | 0.335 | 0.0445 | 1.377 |
| Norman→Adamson | scgpt_finetuned_rmse | magnitude | 0.371 | 0.0392 | 1.229 |
| Norman→Adamson | scgpt_finetuned_rmse | magnitude_plus_disagreement | 0.557 | 0.0382 | 1.453 |
| Norman→Adamson | task_mean_rmse | magnitude | 0.284 | 0.0404 | 1.146 |
| Norman→Adamson | task_mean_rmse | magnitude_plus_disagreement | 0.471 | 0.0393 | 1.418 |
| Norman→Adamson | task_max_rmse | magnitude | 0.369 | 0.0412 | 1.121 |
| Norman→Adamson | task_max_rmse | magnitude_plus_disagreement | 0.590 | 0.0371 | 1.481 |

## 加入模型分歧相对幅度基线的增量

| source→target | error | Δρ | bootstrap 95% CI | 稳定超过幅度 |
|---|---|---:|---:|---|
| Adamson→Norman | gears_ensemble_rmse | 1.085 | [0.489, 1.598] | 是 |
| Adamson→Norman | scgpt_finetuned_rmse | 0.015 | [-0.116, 0.167] | 否 |
| Adamson→Norman | task_mean_rmse | 0.113 | [-0.026, 0.289] | 否 |
| Adamson→Norman | task_max_rmse | 0.139 | [-0.061, 0.377] | 否 |
| Norman→Adamson | gears_ensemble_rmse | 0.259 | [-0.393, 0.813] | 否 |
| Norman→Adamson | scgpt_finetuned_rmse | 0.186 | [-0.029, 0.473] | 否 |
| Norman→Adamson | task_mean_rmse | 0.187 | [-0.049, 0.483] | 否 |
| Norman→Adamson | task_max_rmse | 0.221 | [-0.171, 0.678] | 否 |

## 解释边界

1. 只有两个 source→target 方向、每个目标域 24 个任务，区间必须优先于点估计。
2. 若 combined 没有稳定超过 magnitude，不能把 E65/E67 的同域相关性写成跨域独立增益。
3. predictor 本身仍是各自在本数据集训练；完整回答老师的跨数据集问题还要让 predictor A 训练后直接在 B 输出。
