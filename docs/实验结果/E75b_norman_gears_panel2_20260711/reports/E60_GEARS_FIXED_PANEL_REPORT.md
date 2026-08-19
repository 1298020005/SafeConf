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
| TP73+ctrl | 0.1123 | 0.0049 | 1.7716 | 0.0023 |
| CEBPA+ctrl | 0.0834 | 0.0098 | 9.4601 | 0.0007 |
| HOXA13+ctrl | 0.0788 | 0.0036 | 1.8623 | 0.0014 |
| DLX2+ctrl | 0.0734 | 0.0073 | 5.7307 | 0.0028 |
| COL2A1+ctrl | 0.0666 | 0.0047 | 3.3602 | 0.0004 |
| COL1A1+ctrl | 0.0654 | 0.0057 | 2.0432 | 0.0005 |
| ATL1+ctrl | 0.0572 | 0.0035 | 1.7102 | 0.0015 |
| CEBPB+ctrl | 0.0546 | 0.0064 | 5.6503 | 0.0011 |
| CSRNP1+ctrl | 0.0544 | 0.0062 | 2.9058 | 0.0017 |
| SLC4A1+ctrl | 0.0529 | 0.0046 | 1.5029 | 0.0003 |
| TBX3+ctrl | 0.0526 | 0.0110 | 3.6762 | 0.0008 |
| BPGM+ctrl | 0.0460 | 0.0033 | 1.4393 | 0.0009 |
| CDKN1C+ctrl | 0.0435 | 0.0085 | 3.3663 | 0.0021 |
| SGK1+ctrl | 0.0402 | 0.0067 | 2.5866 | 0.0004 |
| SAMD1+ctrl | 0.0399 | 0.0082 | 3.9900 | 0.0045 |
| SLC38A2+ctrl | 0.0385 | 0.0042 | 2.0680 | 0.0015 |
| CELF2+ctrl | 0.0382 | 0.0042 | 2.3945 | 0.0016 |
| RREB1+ctrl | 0.0377 | 0.0035 | 1.5946 | 0.0009 |
| CBFA2T3+ctrl | 0.0374 | 0.0041 | 1.4303 | 0.0003 |
| ZC3HAV1+ctrl | 0.0360 | 0.0064 | 1.8211 | 0.0004 |
| SLC6A9+ctrl | 0.0339 | 0.0052 | 1.7557 | 0.0005 |
| NCL+ctrl | 0.0336 | 0.0043 | 1.9765 | 0.0011 |
| HK2+ctrl | 0.0291 | 0.0054 | 1.8547 | 0.0005 |
| ARRDC3+ctrl | 0.0252 | 0.0040 | 1.9163 | 0.0009 |

## 分数对 GEARS 实际误差的关联

| score | 可部署 | ρ | bootstrap 95% CI | top20 高误差富集 |
|---|---|---:|---:|---:|
| risk_ensemble_disagreement | 是 | 0.210 | [-0.181, 0.561] | 1.142 |
| risk_predicted_magnitude | 是 | 0.339 | [-0.056, 0.674] | 1.185 |
| true_l2_diagnostic | 否（oracle） | 0.789 | — | nan |

## 扰动特异性核查

在 24 个冻结测试任务中，GEARS ensemble 的 panel-level centroid accuracy 为 0.167。计算方式：每个预测 effect 与所有测试任务的真实 effect centroid 都比较余弦相似度，看最近的 centroid 是否是它自己的扰动。它参考 Systema 的扰动特异性思想，但这里是固定面板诊断，不把它称为完整 Systema 复现。

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
