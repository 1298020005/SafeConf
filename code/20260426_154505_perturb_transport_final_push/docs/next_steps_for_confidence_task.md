# 下一步可执行计划：confidence scoring task

生成时间：2026-05-21 16:43:37

## Step 1：先把已有结果复盘成 confidence 证据表（1 小时内，CPU）

输入：`/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/51_policy_calibrated_q1_20260520/results/SAFETY_TASK_METRICS.csv`、`RISK_COVERAGE.csv`、`SAFE_UNSAFE_CONTRAST.csv`。

输出：`confidence_existing_result_summary.csv`、`risk_error_correlation.csv`、`safe_unsafe_rmse_summary.csv`、`risk_coverage_summary.csv`。

要算：`confidence` vs `rmse` Pearson/Spearman；`predicted_risk = 1 - confidence` vs `rmse`；high/medium/low confidence 三组 RMSE；80% coverage 下 RMSE 是否下降。

需要 GPU：不需要。需要下载数据：不需要。

## Step 2：生成真正的 PredictionRecord（半天内，CPU）

建议新增脚本位置：`03_code/build_prediction_records.py`。

关键列：`dataset`、`split_type`、`heldout`、`seed`、`task_id`、`context`、`perturbation`、`predictor_name`、`rmse`、`top20_overlap`、`deg_precision_top50`、`program_shift_consistency`、`confidence_method`、`confidence_score`、`predicted_risk`、`unsafe_flag`、`support_count`、`context_similarity`、`perturbation_stability`、`expert_disagreement`。

需要 GPU：不需要。需要下载数据：不需要。

## Step 3：补 held-out context-perturbation pair split（半天，CPU）

建议修改位置：`03_code/build_context_splits.py`。

新增函数：`feasible_pair_splits()`；`materialize_pair_split()` 或扩展 `materialize_split()`。

设计：test 留出 `(context, perturbation)` pair；train 不能出现同 pair；train 最好仍有这个 context 的其他 perturbation，以及这个 perturbation 的其他 context。

需要 GPU：不需要。需要下载数据：不需要。

## Step 4：实现 confidence baseline 对比（半天，CPU）

建议新增脚本位置：`03_code/evaluate_confidence_scoring.py`。

baseline：random confidence、support_count、context_similarity、perturbation_stability、expert_disagreement、PolicySafeTransPT confidence。

指标：Spearman(`predicted_risk`, `rmse`)、high-low RMSE gap、risk-coverage AUC、80% coverage RMSE gain、unsafe group 是否 RMSE 更高。

## Step 5：出导师讨论图（1 小时内，CPU）

图：context × perturbation matrix、confidence vs true error scatter、risk-coverage curve、safe vs unsafe RMSE、high/medium/low confidence RMSE、feature/baseline comparison。

## Step 6：第二阶段再考虑深度模型（1-2 天，可用 GPU）

前提：Step 1-5 证明 confidence task 有信号。如果没有信号，不建议训练深度模型。

## 最小路线

1. 先用已有结果算 confidence-error correlation。
2. 再用 KaggleCrossCell 生成 PredictionRecord。
3. 加 pair split。
4. 跑 V0 + ContextSimBaseline。
5. 比较几种 confidence score 谁更能预测 error。
