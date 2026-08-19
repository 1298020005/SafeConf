# 项目目录和代码审计

生成时间：2026-05-21 16:43:37

## 1. 项目根目录

`/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push`

主代码目录：`/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/03_code`

## 2. 当前实验目录

```text
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/00_meta
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/01_asset_audit
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/02_data
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/03_code
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/03_code/__pycache__
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/03_code/configs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/04_logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/27_safetrans_gate_net_20260518
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/27_safetrans_gate_net_20260518/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/27_safetrans_gate_net_20260518/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/28_active_q1_push_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/28_active_q1_push_20260519/full_cpu
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/28_active_q1_push_20260519/gpu0_deep_context
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/28_active_q1_push_20260519/gpu1_deep_external
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/28_active_q1_push_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/29_policy_router_refresh_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/29_policy_router_refresh_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/29_policy_router_refresh_20260519/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/30_policy_router_wide_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/30_policy_router_wide_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/30_policy_router_wide_20260519/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/31_policy_full_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/31_policy_full_20260519/06_full_runs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/31_policy_full_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/32_policy_router_soft_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/32_policy_router_soft_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/32_policy_router_soft_20260519/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/33_policy_router_hybrid_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/33_policy_router_hybrid_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/33_policy_router_hybrid_20260519/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/34_policy_tian_ext_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/34_policy_tian_ext_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/34_policy_tian_ext_20260519/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/35_gpu_deep_gpu0_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/35_gpu_deep_gpu0_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/35_gpu_deep_gpu0_20260519/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/36_gpu_deep_gpu1_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/36_gpu_deep_gpu1_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/36_gpu_deep_gpu1_20260519/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/37_gpu_deep_smoke_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/37_gpu_deep_smoke_20260519/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/37_gpu_deep_smoke_20260519/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/37_gpu_deep_smoke_fix_20260519
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/37_gpu_deep_smoke_fix_20260519/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/38_gpu_calibrated_smoke_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/38_gpu_calibrated_smoke_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/39_gpu_calibrated_main_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/39_gpu_calibrated_main_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/39_gpu_calibrated_main_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/40_gpu_calibrated_tian_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/40_gpu_calibrated_tian_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/40_gpu_calibrated_tian_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/41_effect_objective_smoke_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/41_effect_objective_smoke_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/42_effect_blend_smoke_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/42_effect_blend_smoke_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/43_gpu_effect_objective_main_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/43_gpu_effect_objective_main_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/43_gpu_effect_objective_main_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/44_gpu_effect_objective_tian_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/44_gpu_effect_objective_tian_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/44_gpu_effect_objective_tian_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/45_gpu_effect_queue_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/45_gpu_effect_queue_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/46_q1_cpu_push_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/46_q1_cpu_push_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/46_q1_cpu_push_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/46_top_rank_graft_smoke_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/46_top_rank_graft_smoke_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/47_top_rank_graft_smoke_fast_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/47_top_rank_graft_smoke_fast_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/48_gpu_graft_tian_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/48_gpu_graft_tian_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/48_gpu_graft_tian_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/49_policy_calibrated_smoke_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/49_policy_calibrated_smoke_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/49_policy_calibrated_smoke_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/50_policy_calibrated_smoke2_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/50_policy_calibrated_smoke2_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/50_policy_calibrated_smoke2_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/51_policy_calibrated_q1_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/51_policy_calibrated_q1_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/51_policy_calibrated_q1_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/52_gpu_policy_fix_main_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/52_gpu_policy_fix_main_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/52_gpu_policy_fix_main_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/53_gpu_policy_fix_external_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/53_gpu_policy_fix_external_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/53_gpu_policy_fix_external_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/54_policy_calibrated_smoke3_20260520
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/54_policy_calibrated_smoke3_20260520/logs
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/54_policy_calibrated_smoke3_20260520/results
/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/docs
```

## 3. 核心代码模块清单

完整清单见：`docs/code_inventory.csv`

重点模块：

| file | role | keyword_hits |
| --- | --- | --- |
| 03_code/build_context_splits.py | 从 h5ad 构造 context + perturbation 的 effect task，并提供 leave_context / heldout_perturbation split。 |  |
| 03_code/encoders.py | hash/pathway prior/nearest prior smoothing 等轻量特征。 |  |
| 03_code/evaluate_q1_readiness.py | 把结果目录打分为 NOT_READY/Q2/Q1 等 readiness 标签。 | SafeTransPT, PolicySafeTransPT, risk_coverage, RMSE, rmse |
| 03_code/evaluators.py | Pearson/Spearman/RMSE/top20/DEG/program consistency 指标。 | rmse |
| 03_code/risk_coverage.py | 按 confidence 做 risk-coverage 曲线和单 task error。 | confidence, risk_coverage, rmse |
| 03_code/run_deep_gpu_transport.py | GPU 深度模型 runner。 | V0StrongBaseline, confidence, rmse |
| 03_code/run_gpu_deepsafetrans.py | GPU 深度模型 runner。 | V0StrongBaseline, rmse |
| 03_code/run_safety_abstention_evidence.py | CPU 主证据 runner：读数据、构造 split、跑 V0/V2/Policy/Safe，并输出 safety 结果表。 | V0StrongBaseline, ContextSimilarityBaseline, SafeTransPT, PolicySafeTransPT, transportability_score, unsafe_flag, confidence, risk_covera... |
| 03_code/safetrans_models.py | SafeTransPT / PolicySafeTransPT 等 safety、router、confidence、unsafe 相关模型。 | V0StrongBaseline, ContextSimilarityBaseline, SafeTransPT, PolicySafeTransPT, transportability_score, unsafe_flag, confidence, rmse |
| 03_code/transport_models.py | V0、V1、V2、ContextSimBaseline、V3 等 effect predictor/baseline。 | V0StrongBaseline, ContextSimilarityBaseline, transportability_score, unsafe_flag |

## 4. 关键函数/变量行号

| file | symbol_or_logic | line |
| --- | --- | --- |
| build_context_splits.py | CONTROL_STRINGS | 12 |
| build_context_splits.py | CONTEXT_CANDIDATES | 13 |
| build_context_splits.py | PERT_CANDIDATES | 14 |
| build_context_splits.py | build_effect_tasks() | 76 |
| build_context_splits.py | context_col / pert_col 推断 | 79-80 |
| build_context_splits.py | 表达矩阵读取 logNor 或 X | 85 |
| build_context_splits.py | 按 context + perturbation 分组 | 93 |
| build_context_splits.py | control_mean = 同 context control 平均 | 101-103 |
| build_context_splits.py | effect = perturbed_mean - control_mean | 104-114 |
| build_context_splits.py | feasible_splits() | 133 |
| build_context_splits.py | materialize_split() | 152 |
| transport_models.py | V0StrongBaseline | 37 |
| transport_models.py | V0 fit: by_pert/by_context/global_mean | 42-52 |
| transport_models.py | V0 predict: by_pert + 0.15 by_context | 55-59 |
| transport_models.py | V2GraphPriorTransport | 120 |
| transport_models.py | ContextSimilarityBaseline | 175 |
| transport_models.py | ContextSim confidence/unsafe | 194-245 |
| safetrans_models.py | PolicySafeTransPT | 584 |
| safetrans_models.py | Policy router features | 725-814 |
| safetrans_models.py | utility_reg_ / error_reg_ | 900-902 |
| safetrans_models.py | _calibrated_confidence() | 1011-1069 |
| safetrans_models.py | predict_details 输出 confidence/unsafe/features | 1120-1205 |
| risk_coverage.py | task_errors true_error proxy | 9-20 |
| risk_coverage.py | risk_coverage_curve() | 23-52 |
| run_safety_abstention_evidence.py | 模型列表 | 149-162 |
| run_safety_abstention_evidence.py | 写 confidence / unsafe_flag | 164-180 |
| run_safety_abstention_evidence.py | 输出 CSV | 266-275 |
| evaluate_q1_readiness.py | risk_coverage_gain | 89-102 |
| evaluate_q1_readiness.py | unsafe_contrast_ok | 105-111 |
| evaluate_q1_readiness.py | Q2/Q1 判定逻辑 | 167-195 |

## 5. 当前是否已有 confidence scoring 需要的组件

| 能力/字段 | 当前状态 | 证据 |
| --- | --- | --- |
| context × perturbation task | 已有 | build_context_splits.py:76-130，task 字段含 context / perturbation / effect / control_mean |
| true_effect | 已有但字段名叫 effect | build_context_splits.py:107-114，effect = perturbed_mean - control_mean |
| predicted_effect | 已有但没有统一落盘字段 | predict 返回矩阵；SAFETY_TASK_METRICS 只保存指标，不保存向量 |
| true_error | 已有但字段名多为 rmse | risk_coverage.py:9-20；SAFETY_TASK_METRICS.csv 中 rmse 是单 task error |
| V0 prediction | 已有 | transport_models.py:37-62 |
| ContextSim prediction | 已有 | transport_models.py:175-245 |
| risk score / confidence score | 已有 | safetrans_models.py:1011-1069；输出 transportability_score；runner 落盘为 confidence |
| unsafe flag | 已有 | safetrans_models.py:1171-1178；run_safety_abstention_evidence.py:85-88 |
| risk-coverage | 已有 | risk_coverage.py:23-52；RISK_COVERAGE.csv |
| safe vs unsafe RMSE | 已有 | run_safety_abstention_evidence.py:100-116；SAFE_UNSAFE_CONTRAST.csv |
| Q1 readiness | 已有 | evaluate_q1_readiness.py:114-222 |
| held-out context-perturbation pair split | 没找到 | 当前 build_context_splits.py 只有 leave_context 和 heldout_perturbation；未见 pair-holdout split 实现 |
| 统一 PredictionRecord | 部分已有 | 有 task metrics，但缺 context、perturbation、predicted_effect 向量、true_effect 向量等字段 |

## 6. 对新任务可复用的部分

- `build_context_splits.py` 可复用：已经把原始 h5ad 转成 task，每个 task 是一格 `context × perturbation`。
- `transport_models.py` 可复用：V0 和 ContextSimBaseline 是很适合 confidence task 的 predictor/baseline。
- `risk_coverage.py` 可复用：已经实现按 confidence 从高到低保留样本，看 RMSE 是否下降。
- `run_safety_abstention_evidence.py` 可复用：已有 CPU runner 能输出 task-level RMSE、confidence、unsafe_flag。
- `evaluate_q1_readiness.py` 可复用但不应作为新任务唯一目标：它是 publication-readiness gate，不是 confidence task 的核心 evaluator。

## 7. 新任务还需要补的最小部分

- 需要一个统一 `PredictionRecord` 表。现在结果表有 task-level metrics，但缺少 `context`、`perturbation`、`true_effect`/`predicted_effect` 向量或向量文件索引。
- 需要明确 risk/confidence baseline：random confidence、support_count、context similarity、perturbation stability、expert disagreement。
- 需要新增 held-out context-perturbation pair split。当前没有找到这个 split，只找到 `leave_context` 和 `heldout_perturbation`。
- 需要把 `confidence` 和真实 `rmse` 的 Spearman/Pearson、高低 confidence 组 RMSE、risk-coverage 作为第一层主指标。
