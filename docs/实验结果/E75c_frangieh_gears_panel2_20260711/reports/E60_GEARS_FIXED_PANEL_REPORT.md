# Frangieh｜GEARS 固定任务三 seed 正式审计

## 这轮直接回答老师的问题

这里比较的是 **GEARS 三 seed ensemble 对 Frangieh 真实 held-out effect 的 RMSE**。每个任务先由三个独立训练的 GEARS 模型给出预测，再从预测向量得到分歧和预测幅度；读取真实效应只发生在最后计算 RMSE 时。

- 固定未见单基因任务：24
- GEARS PredictionRecord：72
- 训练 seeds：11, 22, 33
- strict 合同问题：0
- 训练支持度：所有 held-out gene 都是 0，因此不是本 setting 的可排序特征。

## 任务级结果

| task | GEARS ensemble RMSE | seed disagreement | predicted magnitude | seed RMSE SD |
|---|---:|---:|---:|---:|
| RNASEH2A+ctrl | 0.0481 | 0.0126 | 1.8937 | 0.0005 |
| PFN1+ctrl | 0.0462 | 0.0117 | 1.8289 | 0.0002 |
| ACSL3+ctrl | 0.0433 | 0.0120 | 1.6772 | 0.0006 |
| NDUFA13+ctrl | 0.0422 | 0.0141 | 1.8343 | 0.0009 |
| TXNDC17+ctrl | 0.0413 | 0.0125 | 1.5263 | 0.0004 |
| UCN2+ctrl | 0.0366 | 0.0130 | 1.6224 | 0.0007 |
| LGALS3+ctrl | 0.0357 | 0.0116 | 1.5418 | 0.0002 |
| HLA-E+ctrl | 0.0356 | 0.0172 | 1.5916 | 0.0021 |
| CTSA+ctrl | 0.0347 | 0.0143 | 1.5191 | 0.0004 |
| NGFR+ctrl | 0.0344 | 0.0124 | 1.5486 | 0.0011 |
| LAMP2+ctrl | 0.0332 | 0.0138 | 1.6367 | 0.0009 |
| TRIM22+ctrl | 0.0329 | 0.0132 | 1.5111 | 0.0004 |
| EVA1A+ctrl | 0.0324 | 0.0138 | 1.5374 | 0.0009 |
| IDH2+ctrl | 0.0324 | 0.0144 | 1.4947 | 0.0014 |
| DAG1+ctrl | 0.0322 | 0.0135 | 1.5051 | 0.0009 |
| SEC11C+ctrl | 0.0320 | 0.0133 | 1.4617 | 0.0012 |
| SGK1+ctrl | 0.0320 | 0.0128 | 1.5503 | 0.0004 |
| NMRK1+ctrl | 0.0314 | 0.0120 | 1.4199 | 0.0008 |
| CCND2+ctrl | 0.0312 | 0.0147 | 1.4263 | 0.0013 |
| PFDN4+ctrl | 0.0311 | 0.0118 | 1.4932 | 0.0004 |
| AGA+ctrl | 0.0310 | 0.0130 | 1.4627 | 0.0008 |
| SMAD3+ctrl | 0.0309 | 0.0154 | 1.4599 | 0.0015 |
| TM4SF1+ctrl | 0.0305 | 0.0126 | 1.4482 | 0.0006 |
| HLA-DRB5+ctrl | 0.0294 | 0.0149 | 1.4361 | 0.0014 |

## 分数对 GEARS 实际误差的关联

| score | 可部署 | ρ | bootstrap 95% CI | top20 高误差富集 |
|---|---|---:|---:|---:|
| risk_ensemble_disagreement | 是 | -0.289 | [-0.652, 0.176] | 0.910 |
| risk_predicted_magnitude | 是 | 0.869 | [0.682, 0.951] | 1.216 |
| true_l2_diagnostic | 否（oracle） | 0.869 | — | nan |

## 扰动特异性核查

在 24 个冻结测试任务中，GEARS ensemble 的 panel-level centroid accuracy 为 0.083。计算方式：每个预测 effect 与所有测试任务的真实 effect centroid 都比较余弦相似度，看最近的 centroid 是否是它自己的扰动。它参考 Systema 的扰动特异性思想，但这里是固定面板诊断，不把它称为完整 Systema 复现。

## 边界

1. 这是一个真正模型输出的固定任务实验，但只覆盖 GEARS 和 Frangieh；不能代替 GEARS、scGPT、CPA 的统一对照。
2. 所有测试基因都在训练中完全未见，支持度为常数 0；这个设置用于检验未见扰动与 seed 分歧，不能用来评价 support feature。
3. `true_l2_diagnostic` 只保留用于核查，不能作为部署时的风险分数。

## 文件

- 任务表：`tables/E60_TASK_RISK_TABLE.csv`
- 分数表：`tables/E60_RISK_ERROR_SUMMARY.csv`
- 扰动特异性表：`tables/E60_PERTURBATION_SPECIFIC_EVAL.csv`
- 图：`figures/F1_gears_disagreement_vs_error.svg`
- 原始每 seed 输出：`raw_gears/seed_*/frangieh/seed_*/`
