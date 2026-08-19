# SafeConf Evidence-to-Claim Matrix

日期：2026-06-15
阶段：Phase 4c-1
状态：提交 Claude 审核前版本
当前 commit：`e8a3594 exp: E8b scPerturBench aggregate-error association`

## 0. 使用规则

这个文件不是论文正文，也不是新实验结果。它的作用是把每个可写入论文的 claim 和已有证据一一绑定，防止后续写作和画图时过度外推。

Claude 审核通过本文件后，Codex 才进入 Phase 4c-2/3/4：

- figure-ready tables；
- plot draft scripts；
- reproducibility manifest。

本阶段不做新实验，不修改 frozen v0.2，不修改已有结果文件。

## 1. Claim 总览

| Claim ID | 核心结论 | 论文角色 | 主图建议 | 当前状态 |
|---|---|---|---|---|
| F | Task difficulty dominates predictor choice | Motivation + Discussion 地基 | Fig 1 / S-Fig 1 | 可用，需谨慎解释 |
| A | Frozen SafeConf v0.2 在 7 主表中能排序 prediction risk | 主结果 | Fig 2 | 可用，McFarland 是失败边界 |
| B | SafeConf 信号不只是 effect magnitude | 主文关键防御 | Fig 3A | 可用，E2 是最强证据 |
| C | learned task-risk model 能捕获 frozen v0.2 错过的风险信号 | 方法扩展/边界解释 | Fig 3B / Table | 可用，但不能反向修改 frozen 协议 |
| D | 外部公开 benchmark 上存在 method-error association | 外部关联证据 | Fig 4 | 可用，必须带 sample-size 和 metric caveat |
| E | Failure boundary 和限制是明确的 | Discussion + Supplement | Fig 2 标记 / S-Table | 可用，不能淡化 |

## 2. Claim F：Task Difficulty Dominates Predictor Choice

### 允许写法

SafeConf 的动机来自一个经验观察：在当前 benchmark 中，同一个 task 在不同简单预测器下的误差高度相关，误差方差主要由 task identity 而不是 predictor identity 解释。因此，评估 task-level risk 有实际意义。

### 支撑证据

来源：

- `docs/实验结果/Task_risk_audit_20260611/tables/A1_task_vs_predictor_variance_summary.csv`
- `docs/实验结果/Task_risk_audit_20260611/reports/A1_task_risk_interpretation.md`

关键数字：

- overall task variance fraction：0.936267；
- overall predictor variance fraction：0.001060；
- overall residual fraction：0.062673；
- overall Spearman(V0 error, ContextSim error)：0.972663；
- 7/7 数据集均标记为 `task_difficulty_dominant`。

### 不能写法

- 不能写成“所有模型误差都由 task 决定”。
- 不能写成“深度学习模型也一定遵循同样方差结构”。
- 不能写成“predictor choice 不重要”。这里比较的是 V0/ContextSim 这类 retrieval-style predictors。

### 图表定位

- Fig 1 motivation panel 或 S-Fig 1；
- 可以在 Introduction/Results 开头作为 SafeConf 合理性的背景；
- 不建议占用一张完整主图，除非 Claude 认为论文需要更强 motivation。

## 3. Claim A：Frozen SafeConf v0.2 Sorts Prediction Risk Across Main Datasets

### 允许写法

冻结的 SafeConf v0.2 协议在 7 个正式主表数据集中有 6/7 个数据集的 magnitude-controlled partial rho 为正；McFarland 是明确失败边界。

### 支撑证据

来源：

- `docs/实验结果/Formal_main_20260604/corrected_v3_drop_blank_1000_20260609/tables/FORMAL_MAIN_TABLE.csv`
- `docs/实验结果/Formal_main_20260604/corrected_v3_drop_blank_1000_20260609/reports/CORRECTED_7MAIN_DECISION.md`

关键数字：

| dataset | aligned rho | partial rho | partial 95% CI | magnitude-only rho | 备注 |
|---|---:|---:|---:|---:|---|
| Cui | 0.445 | 0.328 | [0.293, 0.362] | 0.736 | positive |
| Frangieh | 0.583 | 0.474 | [0.430, 0.510] | 0.797 | positive |
| Lara ex vivo | 0.563 | 0.443 | [0.376, 0.506] | 0.486 | positive |
| Lara in vivo | 0.394 | 0.357 | [0.290, 0.424] | 0.634 | positive |
| McFarland | -0.086 | -0.061 | [-0.100, -0.023] | 0.795 | failure boundary |
| Santinha | 0.152 | 0.212 | [0.129, 0.297] | 0.840 | weak positive |
| Srivatsan sciplex3 | 0.428 | 0.629 | [0.595, 0.660] | 0.740 | positive |

### 不能写法

- 不能写成“7/7 成功”。
- 不能写成“7/7 LODO all positive”。历史 LODO 表中 McFarland frozen 为负，learned LODO 在 Frangieh/Srivatsan 上也有负结果。
- 不能写成“SafeConf 优于 magnitude-only”。Fig 2 中 magnitude-only rho 通常更高；独立贡献需要 Claim B/E2 支撑。
- 不能把 McFarland 从 frozen v0.2 failure boundary 中移除。
- 不能把 Santinha 写成强阳性；它是 weak positive CRISPR evidence。

### 图表定位

Fig 2：七主表 forest plot。

Claude 指定规格：

- 每个数据集一行；
- 三个点：aligned rho、partial rho、magnitude-only rho；
- partial rho 画 bootstrap CI；
- McFarland 特殊红色标记；
- 目标信息：partial rho 通常低于 magnitude-only，但多数仍为正。

## 4. Claim B：SafeConf Signal Is Not Merely Effect Magnitude

### 允许写法

在 fold-safe E2 magnitude-residual calibration 中，所有 7 个数据集的 learned residual risk 在控制 magnitude 后仍为正；combined predicted error 相比 magnitude-only 排序进一步降低 AURC。这个实验是回答 magnitude bias 的主证据。

### 支撑证据

来源：

- `docs/实验结果/E1_E4_preregistered_20260614/E2_magnitude_residual/E2_MAGNITUDE_RESIDUAL_SUMMARY.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E1_E4_GATE_REPORT.md`

关键数字（isotonic calibration）：

| dataset | residual partial rho | 95% CI | AURC improvement: magnitude minus combined | 95% CI |
|---|---:|---:|---:|---:|
| Cui | 0.600 | [0.555, 0.644] | 0.045 | [0.036, 0.055] |
| Frangieh | 0.232 | [0.132, 0.327] | 0.000823 | [0.000630, 0.001019] |
| Lara ex vivo | 0.638 | [0.556, 0.700] | 1.132 | [0.780, 1.468] |
| Lara in vivo | 0.615 | [0.519, 0.688] | 2.433 | [1.468, 3.193] |
| McFarland | 0.226 | [0.173, 0.281] | 0.307 | [0.252, 0.374] |
| Santinha | 0.291 | [0.175, 0.409] | 0.088 | [0.058, 0.117] |
| Srivatsan sciplex3 | 0.154 | [0.065, 0.241] | 0.002741 | [0.001842, 0.003644] |

Gate result：7/7 residual partial-rho CI lower > 0；7/7 AURC improvement CI lower > 0。两项均超过预注册 4/7 threshold。

### 不能写法

- 不能把 E2 写成 frozen v0.2 本身的结果；E2 是 learned residual / magnitude calibration extension。
- 不能写成“magnitude 不重要”。实际 magnitude 是强 baseline。
- 不能说“完全消除了 confounding”。应写成“在 magnitude 建模或控制后仍有剩余信号”。

### 图表定位

Fig 3 Panel A：

- 七数据集 AURC improvement bar chart；
- `combined predicted error = magnitude_expected_error + learned_predicted_residual`；
- 使用 `magnitude minus combined`，数值大于 0 表示 combined AURC 更低、更好；
- 带 95% CI。

## 5. Claim C：Learned Task-Risk Model Captures Signals Missed by Frozen v0.2

### 允许写法

Fold-safe learned LOPO 结果显示，全 14 特征 HistGBT 在 PertMean 第三预测器上 7/7 数据集 partial rho 为正。McFarland 在 frozen v0.2 中失败，但 learned risk model 在同一数据集上获得正 partial rho，说明该数据集中存在 frozen 简单公式未捕获的风险信号。

### 支撑证据

来源：

- `docs/实验结果/E1_E4_preregistered_20260614/E4_model_stability/E4_SEED_SUMMARY.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E4_model_stability/E4_CONFIG_SUMMARY.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E1_group_ablation/E1_GROUP_GATE.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E1_group_ablation/E1_LODO_LOPO_GATE.csv`

关键数字：

- E4 seed summary：7/7 datasets positive across 10 seeds；
- McFarland learned LOPO partial rho：0.330542；
- frozen McFarland partial rho：-0.060924；
- configuration sensitivity：5/6 或 6/6 configs positive per dataset，具体按 `E4_CONFIG_SUMMARY.csv`；
- E1 LOPO group gate：
  - context：5/7；
  - support：3/7；
  - prediction_output：3/7；
  - disagreement：2/7；
  - OOD：1/7；
  - historical：1/7。
- E1 LODO×LOPO group gate：
  - context：3/7；
  - support：1/7；
  - prediction_output：4/7；
  - disagreement：4/7；
  - OOD：3/7；
  - historical：3/7。

### 不能写法

- 不能写成“McFarland 被 frozen v0.2 挽救”。
- 不能用 learned model 结果回头修改 frozen v0.2 的成功率。
- 不能声称 learned model 完全不依赖预测器输出；full model 含 prediction magnitude/output features。
- 不能写“model_disagreement 是唯一或最稳定信号”。更准确写法是：不同风险源的数据集依赖很强，disagreement 在 LODO×LOPO 下有一定跨数据集贡献，但不是唯一主导因素。

### 图表定位

Fig 3 Panel B 或 Table：

- McFarland frozen vs learned partial rho 对比；
- frozen partial rho = -0.061；
- learned LOPO partial rho = 0.331；
- 目的是解释 failure boundary 中仍存在可学习风险信号。

Supplement：

- E1 group ablation heatmap；
- E4 model stability/config sensitivity。

## 6. Claim D：External Benchmark Method-Error Association Exists

### 允许写法

在 Frangieh shared biological dataset 上，SafeConf frozen v0.2 perturbation-level risk 与 scPerturBench 官方 high-dimensional MSE benchmark 中多种方法的 per-perturbation error 正相关。该结果是 external benchmark method-error association，而不是完整逐向量 SafeConf external validation。

### 支撑证据

来源：

- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_SUMMARY.json`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_PER_METHOD.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_POSTHOC_SAMPLE_SIZE_SUMMARY.json`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_SENSITIVITY_DEG.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_SCIPLEX3_PER_METHOD.csv`

关键数字：

- Frangieh primary：74 perturbations，15 methods；
- primary metric：MSE, DEG=5000；
- median Spearman：0.584；
- perturbation bootstrap 95% CI：[0.393, 0.726]；
- positive methods：14/15；
- shuffled-risk null median：0.007；
- shuffled null 95% range：[-0.232, 0.231]；
- empirical one-sided p：1/201；
- sample-size baseline median rho：0.764；
- risk vs sample-size risk rho：0.524；
- post hoc controlling log(Nstimulated)：median partial rho = 0.335，95% CI [0.047, 0.538]，15/15 methods positive；
- method-level distribution：
  - 12/15 methods cluster at rho in [0.55, 0.62]；
  - baseMLP ≈ 0；
  - scFoundation ≈ 0.04；
  - CPA ≈ 0.21；
  - this bimodal pattern suggests the association is not uniform across all architectures；
- single-feature decomposition：
  - disagreement-only：median rho = 0.642，95% CI [0.471, 0.769]；
  - support-only：median rho = 0.124，95% CI [-0.090, 0.337]，CI crosses zero；
  - context-only：undefined，because context score is constant after perturbation-level aggregation；
- sensitivity：
  - DEG20 MSE median rho = 0.009；
  - DEG50 MSE median rho = 0.084；
  - DEG100 MSE median rho = 0.212；
  - pearson_distance DEG5000 median rho = -0.427；
  - sciplex3 pooled median rho = 0.384。

### 不能写法

- 不能写成“完整外部验证”。
- 不能写成“验证了 27 种架构”。
- 不能写成“跨所有 metrics 都有效”。
- 不能忽略 sample-size baseline；Nstimulated baseline 比 SafeConf frozen risk 更强。
- 不能把 post hoc sample-size partial 当作预注册 gate；它是诊断性分析。

### 图表定位

Fig 4：

- Panel A：15 methods rho bar plot，叠加 shuffled null 95% 灰色带；
- Panel B：raw rho vs sample-size-adjusted partial rho 成对点图；
- 图注明确：
  - primary gate 使用 raw E8b pre-registered analysis；
  - sample-size adjustment 是 post hoc diagnostic；
  - claim 限定 high-dimensional MSE。

Supplement：

- sciplex3 sensitivity；
- DEG/metric sensitivity；
- alias audit。

## 7. Claim E：Failure Boundaries and Limitations Are Explicit

### 允许写法

SafeConf 的失败边界和限制包括：McFarland 在 frozen v0.2 下失败；GEARS 与主表存在 context/gene-space mismatch；Tahoe 目前是 sampled smoke validation；E8b 是 aggregate benchmark association；scPerturBench 没有公开逐任务 expression vectors，不能做 full vector-level external scoring。

### 支撑证据

来源：

- McFarland：
  - `docs/实验结果/Formal_main_20260604/diagnostics/reports/McFarland_failure_diagnosis.md`
  - `docs/实验结果/Formal_main_20260604/corrected_v3_drop_blank_1000_20260609/tables/FORMAL_MAIN_TABLE.csv`
- GEARS：
  - `docs/实验结果/Formal_main_20260604/gears_alignment_audit_20260606/reports/GEARS_ALIGNMENT_AUDIT.md`
- Tahoe：
  - `docs/实验结果/Formal_main_20260604/tahoe_sampled_formal_v1/RUN_STATUS.json`
  - `docs/实验结果/Formal_main_20260604/tahoe_sampled_formal_v1/tables/TAHOE_FORMAL_EVAL_SUMMARY_SMOKE.csv`
- E8:
  - `docs/实验结果/E1_E4_preregistered_20260614/E8_SC_PERTURBENCH_RESOURCE_AUDIT.md`
  - `docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FINAL_REPORT.md`

关键数字：

- McFarland frozen partial rho = -0.061；
- McFarland frozen aligned rho = -0.086；
- Tahoe sampled smoke：100/1026 shards，4,132 test records，partial rho = 0.293；
- Tahoe total PredictionRecords = 20,742；
- E8b Frangieh gate PASS but sample-size baseline rho = 0.764；
- E8b pearson_distance DEG5000 median rho = -0.427。

### 不能写法

- 不能把 Tahoe 写成 full formal validation。
- 不能把 GEARS 写成已完成主表对齐或第三架构正式验证。
- 不能说 McFarland 是数据错误导致；目前应写作 frozen v0.2 的 failure boundary，并讨论 drug/dose/time 混杂。
- 不能把 E8b 说成 full SafeConf scoring over external model predictions。

### 图表定位

- McFarland：Fig 2 特殊标记；S-Table 或 supplement diagnostic；
- Tahoe：Supplement scalability/smoke；
- GEARS：Supplement feasibility/limitation；
- E8b caveat：Fig 4 + Discussion。

## 8. Supporting Validity Audits

这些不构成新的主 claim，但用于审稿防御。

### E1：Group ablation

来源：

- `docs/实验结果/E1_E4_preregistered_20260614/E1_group_ablation/E1_GROUP_GATE.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E1_group_ablation/E1_LODO_LOPO_GATE.csv`

用途：

- 支持“多个风险源共同贡献，且贡献具有数据集依赖性”；
- 不支持“单一特征稳定主导”。

### E3：Negative controls

来源：

- `docs/实验结果/E1_E4_preregistered_20260614/E3_negative_controls/E3_EMPIRICAL_PVALUES.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E3_negative_controls/E3_MISSINGNESS_PAIRED_DELTA.csv`

用途：

- shuffled target / shuffled features 负对照排除随机伪信号；
- missingness-only 0/7 显著，排除单纯缺失模式作弊；
- 当前 permutation=200，最小经验 p=1/201，不宜写过细 p 值。

### E4：Stability

来源：

- `docs/实验结果/E1_E4_preregistered_20260614/E4_model_stability/E4_SEED_SUMMARY.csv`
- `docs/实验结果/E1_E4_preregistered_20260614/E4_model_stability/E4_CONFIG_SUMMARY.csv`

用途：

- 10 seeds 结果一致，主要反映当前 HistGBT 设置近似确定；
- configuration sensitivity 比 seed sensitivity 更有信息量；
- 可写作 supplement robustness，不建议作为主文核心。

## 9. 主图与 Claim 对应关系

| 图 | 对应 claim | 主要信息 | 必须显示的 caveat |
|---|---|---|---|
| Fig 1 | F + method overview | task-risk motivation + SafeConf workflow | A1 uses V0/ContextSim retrieval predictors on 7 main datasets. Not demonstrated for deep-learning predictors or external datasets. |
| Fig 2 | A + E | 7 main datasets forest plot | McFarland failure；magnitude-only 通常强 |
| Fig 3A | B | E2 AURC improvement over magnitude-only | E2 是 learned extension，不是 frozen-only |
| Fig 3B | C + E | McFarland frozen vs learned | learned 不能修改 frozen v0.2 结论 |
| Fig 4A | D | E8b per-method rho + shuffled null | aggregate association, not full external validation |
| Fig 4B | D | raw vs Nstimulated-adjusted partial rho | sample-size adjustment 是 post hoc |

## 10. Claude 需要审核的具体问题

请 Claude 在进入 Phase 4c-2 前确认：

1. Claim F 的位置：放 Fig 1 motivation panel，还是只放 supplement？
   - Claude decision：放 Fig 1 motivation panel，建议 3-panel layout：workflow + V0/ContextSim error scatter + variance decomposition bar。
2. Fig 3 是否接受 “E2 Panel A + McFarland frozen/learned Panel B” 的组合？
   - Claude decision：Panel A 保留；Panel B 改为 7-dataset frozen vs learned partial-rho scatter，McFarland 高亮。
3. Fig 4 Panel B 是否用 post hoc sample-size-adjusted partial rho，还是只在图注/正文报告？
   - Claude decision：画在图里，paired dot plot；同时标注 Nstimulated baseline rho = 0.764。
4. Claim B 中是否允许正文使用“beyond magnitude-only”这个表述？Codex 建议允许，但必须限定为 E2 learned extension。
   - Claude decision：允许，但主语必须是 E2 learned residual-risk model；不能写 frozen v0.2 surpasses magnitude。
5. 是否同意 Santinha 在主表中仅写 weak positive CRISPR evidence？
   - Claude decision：同意。

Claude 确认后，Codex 再进入：

- `figure_ready_tables/`
- `plot_scripts/`
- `REPRODUCIBILITY_MANIFEST.md`

## 11. Phase 4c-1 结论

当前证据已经足够支撑一条谨慎但完整的论文主线：

1. task difficulty 在当前 benchmark 中主导预测误差；
2. frozen SafeConf v0.2 在多数正式数据集上能排序风险；
3. magnitude 是强混杂/强 baseline，但 E2 显示 residual signal 和 combined AURC gain；
4. learned task-risk model 能捕获 frozen 公式漏掉的 McFarland 风险信号；
5. scPerturBench Frangieh 上存在 external benchmark method-error association；
6. 失败边界和限制必须主动写清。

不建议在 Claude 审核本矩阵前继续新增实验或绘制正式图。
