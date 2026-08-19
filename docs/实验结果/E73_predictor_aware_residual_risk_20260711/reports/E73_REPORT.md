# E73｜Predictor-aware residual risk：三数据集冻结外推

每轮留出一个完整数据集。来源数据先拟合 magnitude-only 风险排序，再用跨数据集 out-of-fold 残差训练模型分歧校正器。目标数据的误差、真实表达和效应不参与特征、标准化、选参或拟合。

## 目标域结果

| target | error | method | ρ | top20 enrichment |
|---|---|---|---:|---:|
| Adamson | gears_ensemble_rmse | magnitude_only | 0.107 | 1.192 |
| Adamson | gears_ensemble_rmse | disagreement_only | 0.335 | 1.377 |
| Adamson | gears_ensemble_rmse | fixed_rank_average | 0.241 | 1.097 |
| Adamson | gears_ensemble_rmse | predictor_aware_residual | 0.318 | 1.097 |
| Adamson | scgpt_finetuned_rmse | magnitude_only | 0.281 | 1.141 |
| Adamson | scgpt_finetuned_rmse | disagreement_only | 0.566 | 1.453 |
| Adamson | scgpt_finetuned_rmse | fixed_rank_average | 0.424 | 1.147 |
| Adamson | scgpt_finetuned_rmse | predictor_aware_residual | 0.555 | 1.453 |
| Adamson | task_mean_rmse | magnitude_only | 0.148 | 1.160 |
| Adamson | task_mean_rmse | disagreement_only | 0.494 | 1.418 |
| Adamson | task_mean_rmse | fixed_rank_average | 0.364 | 1.124 |
| Adamson | task_mean_rmse | predictor_aware_residual | 0.495 | 1.418 |
| Adamson | task_max_rmse | magnitude_only | 0.096 | 1.121 |
| Adamson | task_max_rmse | disagreement_only | 0.600 | 1.481 |
| Adamson | task_max_rmse | fixed_rank_average | 0.461 | 1.142 |
| Adamson | task_max_rmse | predictor_aware_residual | 0.601 | 1.481 |
| Norman | gears_ensemble_rmse | magnitude_only | 0.420 | 1.240 |
| Norman | gears_ensemble_rmse | disagreement_only | 0.708 | 1.284 |
| Norman | gears_ensemble_rmse | fixed_rank_average | 0.565 | 1.287 |
| Norman | gears_ensemble_rmse | predictor_aware_residual | 0.617 | 1.238 |
| Norman | scgpt_finetuned_rmse | magnitude_only | 0.340 | 1.139 |
| Norman | scgpt_finetuned_rmse | disagreement_only | 0.298 | 1.166 |
| Norman | scgpt_finetuned_rmse | fixed_rank_average | 0.278 | 1.185 |
| Norman | scgpt_finetuned_rmse | predictor_aware_residual | 0.329 | 1.270 |
| Norman | task_mean_rmse | magnitude_only | 0.383 | 1.222 |
| Norman | task_mean_rmse | disagreement_only | 0.589 | 1.227 |
| Norman | task_mean_rmse | fixed_rank_average | 0.495 | 1.238 |
| Norman | task_mean_rmse | predictor_aware_residual | 0.584 | 1.191 |
| Norman | task_max_rmse | magnitude_only | 0.374 | 1.279 |
| Norman | task_max_rmse | disagreement_only | 0.596 | 1.320 |
| Norman | task_max_rmse | fixed_rank_average | 0.460 | 1.323 |
| Norman | task_max_rmse | predictor_aware_residual | 0.561 | 1.278 |
| Frangieh | gears_ensemble_rmse | magnitude_only | 0.238 | 0.967 |
| Frangieh | gears_ensemble_rmse | disagreement_only | 0.326 | 1.027 |
| Frangieh | gears_ensemble_rmse | fixed_rank_average | 0.333 | 1.047 |
| Frangieh | gears_ensemble_rmse | predictor_aware_residual | 0.294 | 0.984 |
| Frangieh | scgpt_finetuned_rmse | magnitude_only | -0.017 | 0.909 |
| Frangieh | scgpt_finetuned_rmse | disagreement_only | 0.303 | 0.959 |
| Frangieh | scgpt_finetuned_rmse | fixed_rank_average | 0.196 | 0.980 |
| Frangieh | scgpt_finetuned_rmse | predictor_aware_residual | 0.135 | 0.920 |
| Frangieh | task_mean_rmse | magnitude_only | 0.011 | 0.916 |
| Frangieh | task_mean_rmse | disagreement_only | 0.349 | 0.994 |
| Frangieh | task_mean_rmse | fixed_rank_average | 0.292 | 1.014 |
| Frangieh | task_mean_rmse | predictor_aware_residual | 0.198 | 0.968 |
| Frangieh | task_max_rmse | magnitude_only | -0.014 | 0.918 |
| Frangieh | task_max_rmse | disagreement_only | 0.370 | 1.009 |
| Frangieh | task_max_rmse | fixed_rank_average | 0.338 | 1.028 |
| Frangieh | task_max_rmse | predictor_aware_residual | 0.036 | 0.918 |

## 相对 magnitude-only 的增量

| target | error | Δρ | bootstrap 95% CI | 稳定为正 |
|---|---|---:|---:|---|
| Adamson | gears_ensemble_rmse | 0.211 | [-0.235, 0.635] | 否 |
| Adamson | scgpt_finetuned_rmse | 0.274 | [-0.093, 0.672] | 否 |
| Adamson | task_max_rmse | 0.505 | [0.037, 0.951] | 是 |
| Adamson | task_mean_rmse | 0.347 | [-0.118, 0.763] | 否 |
| Frangieh | gears_ensemble_rmse | 0.056 | [-0.224, 0.378] | 否 |
| Frangieh | scgpt_finetuned_rmse | 0.152 | [-0.067, 0.390] | 否 |
| Frangieh | task_max_rmse | 0.050 | [-0.107, 0.194] | 否 |
| Frangieh | task_mean_rmse | 0.187 | [-0.206, 0.582] | 否 |
| Norman | gears_ensemble_rmse | 0.197 | [-0.044, 0.498] | 否 |
| Norman | scgpt_finetuned_rmse | -0.011 | [-0.299, 0.283] | 否 |
| Norman | task_max_rmse | 0.187 | [-0.052, 0.443] | 否 |
| Norman | task_mean_rmse | 0.202 | [-0.069, 0.507] | 否 |

## 解释

这个实验的主判断是三份完整目标数据集上的增量是否方向一致，以及区间是否支持它。单个数据集的正结果不作为方法成立的证据。batch 内特征排序需要拿到待筛选任务的一批预测，但不读取目标真值。
