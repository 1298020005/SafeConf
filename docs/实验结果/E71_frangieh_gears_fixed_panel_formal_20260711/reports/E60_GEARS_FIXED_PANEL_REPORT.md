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
| JAK1+ctrl | 0.0615 | 0.0138 | 1.3468 | 0.0011 |
| JAK2+ctrl | 0.0560 | 0.0150 | 1.4152 | 0.0014 |
| NOLC1+ctrl | 0.0462 | 0.0143 | 1.9394 | 0.0006 |
| BOLA2B+ctrl | 0.0395 | 0.0131 | 1.6626 | 0.0009 |
| SET+ctrl | 0.0388 | 0.0144 | 1.6145 | 0.0007 |
| DDX39A+ctrl | 0.0382 | 0.0144 | 1.6783 | 0.0008 |
| CD63+ctrl | 0.0368 | 0.0201 | 1.7166 | 0.0045 |
| HLA-B+ctrl | 0.0365 | 0.0170 | 1.6260 | 0.0016 |
| SHMT2+ctrl | 0.0362 | 0.0144 | 1.5967 | 0.0007 |
| EEA1+ctrl | 0.0358 | 0.0143 | 1.6153 | 0.0004 |
| SDCBP+ctrl | 0.0347 | 0.0166 | 1.5648 | 0.0030 |
| SAT1+ctrl | 0.0343 | 0.0165 | 1.6256 | 0.0007 |
| IRF4+ctrl | 0.0342 | 0.0130 | 1.4736 | 0.0006 |
| CDK4+ctrl | 0.0341 | 0.0135 | 1.5046 | 0.0010 |
| B2M+ctrl | 0.0338 | 0.0181 | 1.5969 | 0.0042 |
| KLF4+ctrl | 0.0336 | 0.0141 | 1.5223 | 0.0014 |
| LRPAP1+ctrl | 0.0328 | 0.0134 | 1.4860 | 0.0005 |
| S100A6+ctrl | 0.0320 | 0.0156 | 1.4592 | 0.0008 |
| NSG1+ctrl | 0.0320 | 0.0139 | 1.4770 | 0.0009 |
| STOM+ctrl | 0.0320 | 0.0150 | 1.4726 | 0.0023 |
| APOD+ctrl | 0.0317 | 0.0133 | 1.4463 | 0.0008 |
| DDR1+ctrl | 0.0317 | 0.0149 | 1.4889 | 0.0016 |
| ENPP1+ctrl | 0.0310 | 0.0136 | 1.5010 | 0.0003 |
| PSAP+ctrl | 0.0310 | 0.0170 | 1.4620 | 0.0006 |

## 分数对 GEARS 实际误差的关联

| score | 可部署 | ρ | bootstrap 95% CI | top20 高误差富集 |
|---|---|---:|---:|---:|
| risk_ensemble_disagreement | 是 | 0.028 | [-0.382, 0.442] | 0.938 |
| risk_predicted_magnitude | 是 | 0.430 | [-0.086, 0.884] | 1.070 |
| true_l2_diagnostic | 否（oracle） | 0.870 | — | nan |

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
