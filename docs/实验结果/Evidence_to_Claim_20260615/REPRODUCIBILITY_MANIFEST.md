# Reproducibility Manifest

日期：2026-06-16
阶段：Phase 4c + Phase 5a-0 + Phase 5a-1 + Phase 5a-2

## 1. 这一包是什么

这是一包只读整理产物，不包含新实验。Phase 5a-0 追加了
bad-prediction retrieval / risk-coverage 的 figure-ready tables 和 Fig 5 草图，
用于展示 prediction triage 的实际检出价值。Phase 5a-1 追加了
证据绑定版 Methods 初稿和 Claude 审核请求。Phase 5a-2 追加了
证据绑定版 Results 初稿和 Claude 审核请求。

包含：

- `SAFE_CONF_EVIDENCE_TO_CLAIM_MATRIX.md`
- `figure_ready_tables/`
- `figures/`
- `plot_scripts/`
- `PHASE5A1_METHODS_DRAFT.md`
- `PHASE5A1_METHODS_REVIEW_REQUEST.md`
- `PHASE5A2_RESULTS_DRAFT.md`
- `PHASE5A2_RESULTS_REVIEW_REQUEST.md`

用途：

- 给 Claude 审核 claim 是否越界；
- 给后续正文和图表提供固定输入；
- 记录每张图来自哪些冻结结果。
- 给 Methods/Results 写作提供 evidence-bound draft 入口。

## 2. 数据来源

所有图表都只来自已冻结的结果表，不重新跑大实验。

### 主数据源

- `docs/实验结果/Task_risk_audit_20260611/tables/A1_task_vs_predictor_variance_summary.csv`
- `docs/实验结果/Task_risk_audit_20260611/tables/A1_paired_predictor_error_table.csv`
- `docs/实验结果/Formal_main_20260604/corrected_v3_drop_blank_1000_20260609/tables/FORMAL_MAIN_TABLE.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E2_magnitude_residual/E2_MAGNITUDE_RESIDUAL_SUMMARY.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E3_negative_controls/E3_EMPIRICAL_PVALUES.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E3_negative_controls/E3_MISSINGNESS_PAIRED_DELTA.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E4_model_stability/E4_SEED_SUMMARY.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E1_group_ablation/E1_GROUP_ABLATION_DELTAS.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E1_group_ablation/E1_GROUP_GATE.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E1_group_ablation/E1_LODO_LOPO_GATE.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_SUMMARY.json`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_POSTHOC_SAMPLE_SIZE_SUMMARY.json`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_PER_METHOD.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_POSTHOC_SAMPLE_SIZE_PER_METHOD.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_CONTROLS.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_SHUFFLED_NULL.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_PERFEATURE.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_SENSITIVITY_DEG.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_SCIPLEX3_PER_METHOD.csv`
- `docs/实验结果/Task_risk_audit_20260611/tables/B1_bad_prediction_retrieval.csv`
- `docs/实验结果/Formal_main_20260604/paper_figures/tables/PAPER_RISK_COVERAGE_CURVES.csv`

### 版本边界

- 7 主表使用 `corrected_v3_drop_blank_1000_20260609`
- E8b 使用 `e8a3594` 对应的冻结输出
- 本阶段不改 frozen v0.2
- 本阶段不重跑大实验

## 3. 生成脚本

### Figure-ready tables

- `docs/实验结果/Evidence_to_Claim_20260615/plot_scripts/build_figure_ready_tables.py`

作用：

- 汇总 Fig 1-5 和 supplement 的绘图专用 CSV；
- 追加 `source_file` 和 `source_commit`；
- 不做新分析，只做整理。

### Draft figures

- `docs/实验结果/Evidence_to_Claim_20260615/plot_scripts/plot_phase4c_figures.py`

作用：

- 读取 figure-ready tables；
- 生成 Fig 1-5 草图；
- 输出 PNG 和 SVG；
- 用于 Claude 只读审阅信息层级。

## 4. Figure 输出

### Fig 1

文件：

- `figures/FIG1_task_risk_motivation.png`
- `figures/FIG1_task_risk_motivation.svg`

来源：

- `FIG1_A1_VARIANCE_DECOMPOSITION.csv`
- `FIG1_A1_ERROR_SCATTER.csv`

### Fig 2

文件：

- `figures/FIG2_formal_main_forest.png`
- `figures/FIG2_formal_main_forest.svg`

来源：

- `FIG2_FORMAL_MAIN_FOREST.csv`

### Fig 3

文件：

- `figures/FIG3_magnitude_residual_and_learned.png`
- `figures/FIG3_magnitude_residual_and_learned.svg`

来源：

- `FIG3_E2_MAGNITUDE_RESIDUAL.csv`
- `FIG3_LOPO_LEARNED_PANEL.csv`

### Fig 4

文件：

- `figures/FIG4_e8b_external_benchmark.png`
- `figures/FIG4_e8b_external_benchmark.svg`

来源：

- `FIG4_E8B_EXTERNAL_BENCHMARK.csv`
- `FIG4_E8B_PARTIAL_PER_METHOD.csv`
- `FIG4_E8B_CONTROLS.csv`
- `FIG4_E8B_SHUFFLED_NULL.csv`

### Fig 5

文件：

- `figures/FIG5_cost_effectiveness.png`
- `figures/FIG5_cost_effectiveness.svg`
- `figures/SFIG5_cost_effectiveness_thresholds.png`
- `figures/SFIG5_cost_effectiveness_thresholds.svg`

来源：

- `FIG5_COST_EFFECTIVENESS.csv`
- `FIG5_COST_EFFECTIVENESS_MACRO_TOP10.csv`
- `FIG5_COST_EFFECTIVENESS_HEATMAP.csv`

口径：

- 宏平均行来自 `__macro_mean__`，定义为 7 个真实数据集的 macro mean；
- top 10% 风险阈值下，Frozen v0.2 enrichment = 3.35x，Magnitude-only = 3.30x；
- 主文 Fig 5 Panel B 只画 top 10% per-dataset heatmap；
- 完整 top 5% / 10% / 20% per-dataset heatmap 放入 supplement figure；
- Per-dataset risk = 5.36x 是 within-dataset 上界参考，不是 frozen protocol；
- Oracle 使用非部署式真实效应诊断，仅作参考上界。

## 5. Figure-ready tables

所有表都在：

`docs/实验结果/Evidence_to_Claim_20260615/figure_ready_tables/`

主要表：

- `FIG1_A1_VARIANCE_DECOMPOSITION.csv`
- `FIG1_A1_ERROR_SCATTER.csv`
- `FIG2_FORMAL_MAIN_FOREST.csv`
- `FIG3_E2_MAGNITUDE_RESIDUAL.csv`
- `FIG3_LOPO_LEARNED_PANEL.csv`
- `FIG4_E8B_EXTERNAL_BENCHMARK.csv`
- `FIG4_E8B_PARTIAL_PER_METHOD.csv`
- `FIG4_E8B_CONTROLS.csv`
- `FIG4_E8B_SHUFFLED_NULL.csv`
- `FIG5_COST_EFFECTIVENESS.csv`
- `FIG5_COST_EFFECTIVENESS_MACRO_TOP10.csv`
- `FIG5_COST_EFFECTIVENESS_HEATMAP.csv`
- `SFIG_E1_GROUP_ABLATION_HEATMAP.csv`
- `SFIG_E3_NEGATIVE_CONTROLS.csv`
- `SFIG_E3_MISSINGNESS_ONLY_DELTA.csv`
- `SFIG_E8B_PERFEATURE.csv`
- `SFIG_E8B_SENSITIVITY.csv`
- `SFIG_E8B_SCIPLEX3_SENSITIVITY.csv`
- `SFIG_RISK_COVERAGE.csv`
- `TABLE_DATASET_SUMMARY.csv`
- `SOURCE_FILES_USED.csv`

## 6. 已知口径

- E2 是 learned residual / magnitude calibration extension，不是 frozen v0.2 本身。
- E8b 是 external benchmark method-error association，不是 full external validation。
- Fig 4 的 sample-size adjustment 是 post hoc diagnostic，不是 preregistered gate。
- Fig 3B 的 learned panel 使用 seed min/max 稳定性，不伪装成 bootstrap CI。
- Fig 1 的 A1 只基于 V0/ContextSim retrieval predictors on 7 main datasets。
- Fig 5 的 `Frozen v0.2` 与 `Magnitude-only` 在 top 10% enrichment 上几乎持平；
  不应写成 frozen 全面优于 magnitude，也不应写成二者 per-dataset 表现相同。
- Fig 5 的正确口径是 macro-averaged enrichment comparable, with complementary
  per-dataset strengths。
- Fig 5 的 `Per-dataset risk` 是 within-dataset training/reference upper bound；
  不应写成 deployable frozen protocol。
- Phase 5a-1 Methods draft 是当前证据线的 Methods-only 草稿；
  不应与旧 v0.3 audit-contract manuscript 混用为同一稿件版本。
- Phase 5a-2 Results draft 是 evidence-bound Results 草稿；
  不应把其中的 cautious boundary notes 删除后直接用作投稿正文。

## 7. 当前状态

- matrix 已按 Claude 修正；
- figure-ready tables 已生成；
- Fig 1-5 草图已生成；
- Phase 5a-0 已追加 Fig 5 cost-effectiveness / triage 数据整理；
- Phase 5a-1 已追加 evidence-bound Methods 初稿；
- Phase 5a-2 已追加 evidence-bound Results 2.1-2.7 初稿；
- 下一步是把 Results draft 交给 Claude 只读复核，确认后进入 Discussion/Introduction。
