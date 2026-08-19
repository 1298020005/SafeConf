# 已有结果审计

生成时间：2026-05-21 16:43:37

## 1. 结果文件总体情况

完整结果文件列名清单：`docs/result_file_inventory.csv`

常见结果文件类型统计：

| name | count |
| --- | --- |
| GPU_DEEP_SELECTED_DATASETS.csv | 16 |
| GPU_DEEP_STATUS.json | 14 |
| GPU_DEEP_AUDIT.csv | 14 |
| GPU_DEEP_TASK_METRICS_INCREMENTAL.csv | 14 |
| GPU_DEEP_TASK_METRICS.csv | 14 |
| GPU_DEEP_SUMMARY.csv | 13 |
| SAFETY_EXTERNAL_SELECTED.csv | 11 |
| SAFETY_STATUS.json | 11 |
| SAFETY_MAIN_SELECTED.csv | 11 |
| NetworkSafeTransPT_VS_V2.csv | 11 |
| NetworkSafeTransPT_VS_V0.csv | 11 |
| SafeTransPT_no_pathway_VS_V0.csv | 11 |
| SafeTransPT_no_pathway_VS_V2.csv | 11 |
| V2_VS_V2.csv | 11 |
| V2_VS_V0.csv | 11 |
| SafeTransPT_no_abstain_VS_V2.csv | 11 |
| SAFETY_AUDIT.csv | 11 |
| RISK_COVERAGE_INCREMENTAL.csv | 11 |
| RISK_COVERAGE.csv | 11 |
| SAFETY_SUMMARY.csv | 11 |
| SafeTransPT_no_abstain_VS_V0.csv | 11 |
| SafeTransPT_VS_V2.csv | 11 |
| SafeTransPT_VS_V0.csv | 11 |
| SAFE_UNSAFE_CONTRAST_INCREMENTAL.csv | 11 |
| SAFE_UNSAFE_CONTRAST.csv | 11 |
| SAFETY_TASK_METRICS_INCREMENTAL.csv | 11 |
| SAFETY_TASK_METRICS.csv | 11 |
| PolicySafeTransPT_VS_V2.csv | 10 |
| PolicySafeTransPT_VS_V0.csv | 10 |
| Q1_READINESS_REPORT.json | 8 |
| PolicySafeTransPT_VS_ContextSimBaseline.csv | 5 |
| ContextSimBaseline_VS_V0.csv | 5 |
| ContextSimBaseline_VS_V2.csv | 5 |
| GPU_EXTERNAL_SELECTED_cuda0.csv | 2 |
| GPU_MAIN_SELECTED_cuda0.csv | 2 |
| Q1_READINESS_REPORT_DeepCalibratedSafeTransport.json | 2 |
| GPU_DEEP_AUDIT_INCREMENTAL.csv | 1 |
| Q1_READINESS_REPORT_TopRankGraftV2.json | 1 |
| Q1_READINESS_REPORT_EffectBlendV2.json | 1 |
| TRANSPORT_GATE_TASKS.csv | 1 |

## 2. 重点结果目录

| results_dir | exists | label | primary_model | SAFETY_TASK_METRICS.csv_rows | RISK_COVERAGE.csv_rows | SAFE_UNSAFE_CONTRAST.csv_rows | GPU_DEEP_TASK_METRICS.csv_rows | gaps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/46_q1_cpu_push_20260520/results | True | NOT_READY | PolicySafeTransPT | 5760.0 | 4428.0 | 738.0 | nan | PolicySafeTransPT must win V0 on >=75% main held-out settings (effect metrics). / Must beat V2 on >=55% held-out perturbation settings. /... |
| /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/51_policy_calibrated_q1_20260520/results | True | NOT_READY | PolicySafeTransPT | 24560.0 | 4824.0 | 804.0 | nan | PolicySafeTransPT must win V0 on >=75% main held-out settings (effect metrics). / Must beat V2 on >=55% held-out perturbation settings. /... |
| /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/52_gpu_policy_fix_main_20260520/results | True | nan | nan | nan | nan | nan | 6650.0 | nan |
| /home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/53_gpu_policy_fix_external_20260520/results | True | nan | nan | nan | nan | nan | 20965.0 | nan |

## 3. 关键结果文件列名

| file | columns | has_predictor_name | has_context | has_perturbation | has_true_error | has_predicted_risk | has_confidence | has_unsafe_flag | has_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 46_q1_cpu_push_20260520/results/SAFETY_TASK_METRICS.csv | pearson, spearman, rmse, top20_overlap, deg_precision_top50, program_shift_consistency, phase, dataset, split_type, heldout, seed, n_trai... | True | False | False | True | True | True | True | True |
| 46_q1_cpu_push_20260520/results/RISK_COVERAGE.csv | coverage, abstention_rate, mean_confidence, rmse, top20_overlap, deg_precision_top50, n_kept, n_total, phase, dataset, split_type, heldou... | True | False | False | True | False | False | False | True |
| 46_q1_cpu_push_20260520/results/SAFE_UNSAFE_CONTRAST.csv | phase, dataset, split_type, heldout, seed, n_train, n_tasks, model, n_safe, n_unsafe, status, unsafe_minus_safe_rmse, safe_minus_unsafe_t... | True | False | False | False | False | False | False | False |
| 46_q1_cpu_push_20260520/results/SAFETY_SUMMARY.csv | phase, dataset, split_type, model, n_runs, n_tasks_mean, pearson_mean, pearson_std, spearman_mean, spearman_std, rmse_mean, rmse_std, top... | True | False | False | False | False | False | False | True |
| 46_q1_cpu_push_20260520/results/Q1_READINESS_REPORT.json | json keys: results_dir, label, primary_model, checks, gaps, n_summary_rows, models_present | nan | nan | nan | nan | nan | nan | nan | nan |
| 51_policy_calibrated_q1_20260520/results/SAFETY_TASK_METRICS.csv | pearson, spearman, rmse, top20_overlap, deg_precision_top50, program_shift_consistency, phase, dataset, split_type, heldout, seed, n_trai... | True | False | False | True | True | True | True | True |
| 51_policy_calibrated_q1_20260520/results/RISK_COVERAGE.csv | coverage, abstention_rate, mean_confidence, rmse, top20_overlap, deg_precision_top50, n_kept, n_total, phase, dataset, split_type, heldou... | True | False | False | True | False | False | False | True |
| 51_policy_calibrated_q1_20260520/results/SAFE_UNSAFE_CONTRAST.csv | phase, dataset, split_type, heldout, seed, n_train, n_tasks, model, n_safe, n_unsafe, status, unsafe_minus_safe_rmse, safe_minus_unsafe_t... | True | False | False | False | False | False | False | False |
| 51_policy_calibrated_q1_20260520/results/SAFETY_SUMMARY.csv | phase, dataset, split_type, model, n_runs, n_tasks_mean, pearson_mean, pearson_std, spearman_mean, spearman_std, rmse_mean, rmse_std, top... | True | False | False | False | False | False | False | True |
| 51_policy_calibrated_q1_20260520/results/Q1_READINESS_REPORT.json | json keys: results_dir, label, primary_model, checks, gaps, n_summary_rows, models_present | nan | nan | nan | nan | nan | nan | nan | nan |
| 52_gpu_policy_fix_main_20260520/results/GPU_DEEP_TASK_METRICS.csv | pearson, spearman, rmse, top20_overlap, deg_precision_top50, program_shift_consistency, phase, dataset, split_type, heldout, seed, n_trai... | True | False | False | True | False | False | False | True |
| 52_gpu_policy_fix_main_20260520/results/GPU_DEEP_SUMMARY.csv | phase, dataset, split_type, model, n_runs, n_tasks_mean, pearson_mean, pearson_std, spearman_mean, spearman_std, rmse_mean, rmse_std, top... | True | False | False | False | False | False | False | True |
| 53_gpu_policy_fix_external_20260520/results/GPU_DEEP_TASK_METRICS.csv | pearson, spearman, rmse, top20_overlap, deg_precision_top50, program_shift_consistency, phase, dataset, split_type, heldout, seed, n_trai... | True | False | False | True | False | False | False | True |
| 53_gpu_policy_fix_external_20260520/results/GPU_DEEP_SUMMARY.csv | phase, dataset, split_type, model, n_runs, n_tasks_mean, pearson_mean, pearson_std, spearman_mean, spearman_std, rmse_mean, rmse_std, top... | True | False | False | False | False | False | False | True |

## 4. 是否包含 confidence scoring 需要的字段

- `predictor_name`：没有看到这个列名，但大多数结果用 `model` 表示模型/预测器名称。
- `context`：在 task-level result 中没有直接看到 `context` 列；只有 `split_type` 和 `heldout`。如果要做正式 PredictionRecord，需要补。
- `perturbation`：task-level result 中没有直接看到 `perturbation` 列；held-out perturbation split 的 `heldout` 可以间接表示被留出的 perturbation，但不是每个 task 的完整字段。
- `true_error`：没有看到 `true_error` 这个列名，但 `rmse` 就是每个 task 的真实误差代理。
- `predicted_risk`：没有统一列名。`confidence` 存在时可以派生 `predicted_risk = 1 - confidence`，但方向要重新验证。
- `confidence`：`SAFETY_TASK_METRICS.csv` 中有。
- `unsafe_flag`：`SAFETY_TASK_METRICS.csv` 和 `SAFE_UNSAFE_CONTRAST.csv` 中有。
- `RMSE`：以小写 `rmse` 或 summary 中 `rmse_mean` 存在。

## 5. 最新 CPU safety 结果快照

以下基于 `/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/51_policy_calibrated_q1_20260520/results`：

```json
SAFETY_TASK_METRICS.csv: 24560 rows, columns=['pearson', 'spearman', 'rmse', 'top20_overlap', 'deg_precision_top50', 'program_shift_consistency', 'phase', 'dataset', 'split_type', 'heldout', 'seed', 'n_train', 'n_tasks', 'model', 'task_id', 'confidence', 'unsafe_flag']
{
  "model": "PolicySafeTransPT",
  "n_task_rows": 3070,
  "full_rmse": 0.22158690225636338,
  "has_confidence": true,
  "risk_error_pearson_1_minus_confidence": -0.12550178540355486,
  "risk_error_spearman_1_minus_confidence": -0.13417596301819967,
  "confidence_error_pearson": 0.12550178540355483,
  "confidence_error_spearman": 0.13417596301819967,
  "n_safe": 2170,
  "n_unsafe": 900,
  "safe_rmse": 0.2173065236808243,
  "unsafe_rmse": 0.23190737059960762,
  "unsafe_minus_safe_rmse": 0.014600846918783328,
  "risk_full_rmse": 0.08439849728042285,
  "risk_80cov_rmse": 0.0792352359660859,
  "risk_80cov_gain": 0.061177171166703015
}
```

解释：

- 当前已有结果能做一版“confidence 与 RMSE 是否相关”的初步分析，因为有 `confidence` 和 `rmse`。
- 但是它还不是理想的 PredictionRecord，因为缺少每个 task 的 `context`、`perturbation`、`true_effect`/`predicted_effect` 向量或向量索引。
- `risk_coverage.csv` 已经有：coverage、abstention_rate、mean_confidence、rmse、top20、DEG precision。
- `SAFE_UNSAFE_CONTRAST.csv` 已有 safe vs unsafe RMSE 对比。
- 目前 `predicted_risk = 1 - confidence` 与 RMSE 的相关性需要谨慎解释：如果相关性为负，说明“risk 越大 error 越大”这个定义还没站稳；但 risk-coverage 在 80% coverage 是否下降也要同时看。

## 6. 能不能直接支持新 confidence scoring task

可以支持“第一版初步审计”：

- 用已有 `SAFETY_TASK_METRICS.csv` 计算 `confidence` vs `rmse` 相关性。
- 用已有 `RISK_COVERAGE.csv` 看高 confidence 样本保留后 RMSE 是否下降。
- 用已有 `SAFE_UNSAFE_CONTRAST.csv` 看 unsafe 样本是否真的更难预测。

还不能支持“定稿级 confidence scoring paper 表格”：

- 缺完整 `PredictionRecord`。
- 缺多种 confidence baseline 的公平对比，例如 random、support count、context similarity、perturbation stability、expert disagreement。
- 缺 held-out context-perturbation pair split。
- 缺 calibration 指标，如 error quantile calibration / ECE 风格分箱。
