# Adamson｜GEARS 固定任务三 seed 正式审计

## 这轮直接回答老师的问题

这里比较的是 **GEARS 三 seed ensemble 对 Adamson 真实 held-out effect 的 RMSE**。每个任务先由三个独立训练的 GEARS 模型给出预测，再从预测向量得到分歧和预测幅度；读取真实效应只发生在最后计算 RMSE 时。

- 固定未见单基因任务：24
- GEARS PredictionRecord：72
- 训练 seeds：11, 22, 33
- strict 合同问题：0
- 训练支持度：所有 held-out gene 都是 0，因此不是本 setting 的可排序特征。

## 任务级结果

| task | GEARS ensemble RMSE | seed disagreement | predicted magnitude | seed RMSE SD |
|---|---:|---:|---:|---:|
| HSPA5+ctrl | 0.0789 | 0.0178 | 2.9452 | 0.0054 |
| CAD+ctrl | 0.0776 | 0.0120 | 1.7537 | 0.0015 |
| SEC61A1+ctrl | 0.0701 | 0.0122 | 1.2185 | 0.0018 |
| MTHFD1+ctrl | 0.0672 | 0.0135 | 1.6785 | 0.0004 |
| PPWD1+ctrl | 0.0575 | 0.0131 | 1.7247 | 0.0015 |
| ASCC3+ctrl | 0.0514 | 0.0138 | 1.5042 | 0.0010 |
| HSD17B12+ctrl | 0.0440 | 0.0136 | 1.7320 | 0.0017 |
| SRPRB+ctrl | 0.0429 | 0.0139 | 1.3373 | 0.0014 |
| COPB1+ctrl | 0.0427 | 0.0166 | 2.2232 | 0.0024 |
| MRGBP+ctrl | 0.0415 | 0.0147 | 1.6706 | 0.0025 |
| SPCS2+ctrl | 0.0391 | 0.0187 | 2.6535 | 0.0037 |
| DDOST+ctrl | 0.0338 | 0.0107 | 1.4575 | 0.0018 |
| SCYL1+ctrl | 0.0334 | 0.0118 | 1.3616 | 0.0025 |
| SRP72+ctrl | 0.0324 | 0.0136 | 1.4481 | 0.0017 |
| SOCS1+ctrl | 0.0313 | 0.0136 | 1.6701 | 0.0033 |
| TMED2+ctrl | 0.0301 | 0.0139 | 1.4958 | 0.0012 |
| SEC61B+ctrl | 0.0288 | 0.0114 | 1.1837 | 0.0015 |
| P4HB+ctrl | 0.0283 | 0.0174 | 1.5419 | 0.0054 |
| DDRGK1+ctrl | 0.0267 | 0.0158 | 1.7424 | 0.0044 |
| TTI2+ctrl | 0.0260 | 0.0120 | 1.3577 | 0.0035 |
| TMED10+ctrl | 0.0260 | 0.0146 | 1.4220 | 0.0006 |
| AMIGO3+ctrl | 0.0243 | 0.0150 | 1.7574 | 0.0037 |
| GMPPB+ctrl | 0.0241 | 0.0122 | 1.4406 | 0.0020 |
| UFM1+ctrl | 0.0228 | 0.0151 | 1.4018 | 0.0027 |

## 分数对 GEARS 实际误差的关联

| score | 可部署 | ρ | bootstrap 95% CI | top20 高误差富集 |
|---|---|---:|---:|---:|
| risk_ensemble_disagreement | 是 | -0.116 | [-0.541, 0.283] | 1.055 |
| risk_predicted_magnitude | 是 | 0.332 | [-0.140, 0.715] | 1.285 |
| true_l2_diagnostic | 否（oracle） | 0.947 | — | nan |

## 扰动特异性核查

在 24 个冻结测试任务中，GEARS ensemble 的 panel-level centroid accuracy 为 0.083。计算方式：每个预测 effect 与所有测试任务的真实 effect centroid 都比较余弦相似度，看最近的 centroid 是否是它自己的扰动。它参考 Systema 的扰动特异性思想，但这里是固定面板诊断，不把它称为完整 Systema 复现。

## 边界

1. 这是一个真正模型输出的固定任务实验，但只覆盖 GEARS 和 Adamson；不能代替 GEARS、scGPT、CPA 的统一对照。
2. 所有测试基因都在训练中完全未见，支持度为常数 0；这个设置用于检验未见扰动与 seed 分歧，不能用来评价 support feature。
3. `true_l2_diagnostic` 只保留用于核查，不能作为部署时的风险分数。

## 文件

- 任务表：`tables/E60_TASK_RISK_TABLE.csv`
- 分数表：`tables/E60_RISK_ERROR_SUMMARY.csv`
- 扰动特异性表：`tables/E60_PERTURBATION_SPECIFIC_EVAL.csv`
- 图：`figures/F1_gears_disagreement_vs_error.svg`
- 原始每 seed 输出：`raw_gears/seed_*/adamson/seed_*/`
