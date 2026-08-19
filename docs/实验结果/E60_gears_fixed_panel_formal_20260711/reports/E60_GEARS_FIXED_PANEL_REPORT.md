# E60｜GEARS 固定任务三 seed 正式审计

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
| EIF2S1+ctrl | 0.1215 | 0.0087 | 1.1412 | 0.0020 |
| IARS2+ctrl | 0.0863 | 0.0088 | 1.2102 | 0.0025 |
| PSMD4+ctrl | 0.0857 | 0.0086 | 1.3150 | 0.0025 |
| NEDD8+ctrl | 0.0719 | 0.0084 | 1.2176 | 0.0002 |
| TTI1+ctrl | 0.0576 | 0.0076 | 1.1182 | 0.0008 |
| DNAJC19+ctrl | 0.0523 | 0.0086 | 2.0096 | 0.0049 |
| COPZ1+ctrl | 0.0517 | 0.0095 | 1.8219 | 0.0003 |
| SPCS3+ctrl | 0.0383 | 0.0086 | 1.1096 | 0.0023 |
| SRP68+ctrl | 0.0371 | 0.0076 | 1.0175 | 0.0019 |
| SEC61G+ctrl | 0.0352 | 0.0106 | 1.4607 | 0.0030 |
| MRPL39+ctrl | 0.0326 | 0.0086 | 1.2159 | 0.0022 |
| SEL1L+ctrl | 0.0304 | 0.0091 | 1.4084 | 0.0044 |
| PDIA6+ctrl | 0.0291 | 0.0079 | 1.0717 | 0.0020 |
| ATF4+ctrl | 0.0284 | 0.0089 | 1.2328 | 0.0002 |
| IDH3A+ctrl | 0.0279 | 0.0087 | 1.0926 | 0.0007 |
| DAD1+ctrl | 0.0272 | 0.0086 | 1.4321 | 0.0018 |
| TIMM23+ctrl | 0.0266 | 0.0087 | 2.5203 | 0.0022 |
| DERL2+ctrl | 0.0258 | 0.0088 | 1.1086 | 0.0006 |
| TELO2+ctrl | 0.0257 | 0.0081 | 0.9919 | 0.0017 |
| OST4+ctrl | 0.0251 | 0.0092 | 1.2385 | 0.0029 |
| SYVN1+ctrl | 0.0220 | 0.0086 | 1.1510 | 0.0043 |
| PTDSS1+ctrl | 0.0210 | 0.0095 | 1.1865 | 0.0042 |
| SLC35B1+ctrl | 0.0198 | 0.0086 | 1.2865 | 0.0045 |
| GBF1+ctrl | 0.0173 | 0.0086 | 1.1463 | 0.0034 |

## 分数对 GEARS 实际误差的关联

| score | 可部署 | ρ | bootstrap 95% CI | top20 高误差富集 |
|---|---|---:|---:|---:|
| risk_ensemble_disagreement | 是 | -0.091 | [-0.475, 0.289] | 0.787 |
| risk_predicted_magnitude | 是 | 0.095 | [-0.250, 0.425] | 0.929 |
| true_l2_diagnostic | 否（oracle） | 0.943 | — | nan |

## 扰动特异性核查

在 24 个冻结测试任务中，GEARS ensemble 的 panel-level centroid accuracy 为 0.083。计算方式：每个预测 effect 与 24 个真实 effect centroid 都比较余弦相似度，看最近的 centroid 是否是它自己的扰动。它参考 Systema 的扰动特异性思想，但这里是固定面板诊断，不把它称为完整 Systema 复现。

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
