# Norman｜GEARS 固定任务三 seed 正式审计

## 这轮直接回答老师的问题

这里比较的是 **GEARS 三 seed ensemble 对 Norman 真实 held-out effect 的 RMSE**。每个任务先由三个独立训练的 GEARS 模型给出预测，再从预测向量得到分歧和预测幅度；读取真实效应只发生在最后计算 RMSE 时。

- 固定未见单基因任务：24
- GEARS PredictionRecord：72
- 训练 seeds：11, 22, 33
- strict 合同问题：0
- 训练支持度：所有 held-out gene 都是 0，因此不是本 setting 的可排序特征。

## 任务级结果

| task | GEARS ensemble RMSE | seed disagreement | predicted magnitude | seed RMSE SD |
|---|---:|---:|---:|---:|
| FEV+ctrl | 0.0967 | 0.0061 | 7.3576 | 0.0023 |
| PRDM1+ctrl | 0.0865 | 0.0044 | 2.0454 | 0.0009 |
| CLDN6+ctrl | 0.0790 | 0.0049 | 6.3258 | 0.0018 |
| SPI1+ctrl | 0.0767 | 0.0082 | 5.8570 | 0.0022 |
| IKZF3+ctrl | 0.0750 | 0.0091 | 8.3751 | 0.0031 |
| FOSB+ctrl | 0.0745 | 0.0043 | 6.6316 | 0.0030 |
| IER5L+ctrl | 0.0708 | 0.0134 | 6.4146 | 0.0088 |
| SNAI1+ctrl | 0.0689 | 0.0071 | 6.1016 | 0.0033 |
| JUN+ctrl | 0.0673 | 0.0064 | 5.4368 | 0.0022 |
| MIDN+ctrl | 0.0558 | 0.0042 | 2.3407 | 0.0008 |
| PTPN13+ctrl | 0.0557 | 0.0032 | 3.5193 | 0.0004 |
| MAPK1+ctrl | 0.0550 | 0.0055 | 3.7084 | 0.0011 |
| FOXA1+ctrl | 0.0547 | 0.0031 | 4.7106 | 0.0016 |
| CKS1B+ctrl | 0.0519 | 0.0035 | 2.0960 | 0.0004 |
| MEIS1+ctrl | 0.0499 | 0.0052 | 4.1961 | 0.0034 |
| OSR2+ctrl | 0.0466 | 0.0071 | 3.4106 | 0.0015 |
| S1PR2+ctrl | 0.0446 | 0.0167 | 3.3258 | 0.0050 |
| PTPN12+ctrl | 0.0399 | 0.0040 | 3.9409 | 0.0016 |
| STIL+ctrl | 0.0394 | 0.0065 | 2.6969 | 0.0026 |
| FOXA3+ctrl | 0.0390 | 0.0041 | 3.9287 | 0.0010 |
| TSC22D1+ctrl | 0.0376 | 0.0049 | 2.7139 | 0.0020 |
| POU3F2+ctrl | 0.0367 | 0.0033 | 2.4303 | 0.0001 |
| IGDCC3+ctrl | 0.0359 | 0.0048 | 3.2345 | 0.0006 |
| BCL2L11+ctrl | 0.0294 | 0.0036 | 1.9017 | 0.0015 |

## 分数对 GEARS 实际误差的关联

| score | 可部署 | ρ | bootstrap 95% CI | top20 高误差富集 |
|---|---|---:|---:|---:|
| risk_ensemble_disagreement | 是 | 0.331 | [-0.027, 0.625] | 1.179 |
| risk_predicted_magnitude | 是 | 0.611 | [0.177, 0.896] | 1.390 |
| true_l2_diagnostic | 否（oracle） | 0.713 | — | nan |

## 扰动特异性核查

在 24 个冻结测试任务中，GEARS ensemble 的 panel-level centroid accuracy 为 0.125。计算方式：每个预测 effect 与所有测试任务的真实 effect centroid 都比较余弦相似度，看最近的 centroid 是否是它自己的扰动。它参考 Systema 的扰动特异性思想，但这里是固定面板诊断，不把它称为完整 Systema 复现。

## 边界

1. 这是一个真正模型输出的固定任务实验，但只覆盖 GEARS 和 Norman；不能代替 GEARS、scGPT、CPA 的统一对照。
2. 所有测试基因都在训练中完全未见，支持度为常数 0；这个设置用于检验未见扰动与 seed 分歧，不能用来评价 support feature。
3. `true_l2_diagnostic` 只保留用于核查，不能作为部署时的风险分数。

## 文件

- 任务表：`tables/E60_TASK_RISK_TABLE.csv`
- 分数表：`tables/E60_RISK_ERROR_SUMMARY.csv`
- 扰动特异性表：`tables/E60_PERTURBATION_SPECIFIC_EVAL.csv`
- 图：`figures/F1_gears_disagreement_vs_error.svg`
- 原始每 seed 输出：`raw_gears/seed_*/norman/seed_*/`
