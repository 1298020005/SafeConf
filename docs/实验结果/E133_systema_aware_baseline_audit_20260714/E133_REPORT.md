# E133｜Systema-aware 简单基线与方向误差审计

## 结论

六数据集、30 个不重复 fold 标签、2,953 个测试任务均按冻结方案复核。SafeConf 对训练扰动均值中心化后的 Pearson 方向误差，数据集等权宏平均 Spearman 为 **-0.109**；对 cosine 方向误差为 **-0.094**。

对“正式模型相对简单扰动均值的超额 RMSE”，SafeConf 宏平均 Spearman 为 **-0.087**。该结论只描述排序能力，不把 SafeConf 写成新的扰动预测器。

两模型均值预测相对 training perturbed-mean 的数据集等权平均 RMSE 优势，聚类 bootstrap 中位数为 **-0.0003**，95% CI **[-0.0009, 0.0003]**。

## 每个数据集的简单基线对照

| 数据集 | tasks | scGPT RMSE | GEARS RMSE | ensemble RMSE | training perturbed-mean RMSE | ensemble − simple | ensemble 胜出任务比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frangieh | 837 | 0.0526 | 0.0566 | 0.0532 | 0.0535 | -0.0003 | 64.0% |
| Lara_exvivo | 345 | 0.1073 | 0.1005 | 0.0983 | 0.1054 | -0.0071 | 69.0% |
| Liang | 612 | 0.0825 | 0.0827 | 0.0794 | 0.0804 | -0.0010 | 47.1% |
| Santinha | 255 | 0.0534 | 0.0520 | 0.0502 | 0.0492 | +0.0010 | 32.9% |
| Shifrut | 172 | 0.0877 | 0.0604 | 0.0662 | 0.0564 | +0.0098 | 8.1% |
| Tian_CRISPRi | 732 | 0.0975 | 0.0991 | 0.0961 | 0.0965 | -0.0004 | 51.9% |

## 六数据集等权排序结果

| 风险分数 | 原 RMSE | centered Pearson error | centered cosine error | excess RMSE vs simple |
|---|---:|---:|---:|---:|
| baseline_predicted_magnitude | 0.093 | -0.048 | -0.065 | 0.053 |
| risk_model_disagreement | 0.049 | -0.008 | -0.034 | 0.087 |
| safeconf_calibrated_pair_risk | 0.204 | -0.109 | -0.094 | -0.087 |
| safeconf_frozen_pair_risk | 0.145 | -0.069 | -0.071 | 0.001 |

## 数据边界检查

- Frangieh：3 folds，837 tasks；保存真值与从原始数据重建真值的最大绝对差 4.77e-06，一致性检查=通过。
- Lara_exvivo：5 folds，345 tasks；保存真值与从原始数据重建真值的最大绝对差 0.00e+00，一致性检查=通过。
- Santinha：5 folds，255 tasks；保存真值与从原始数据重建真值的最大绝对差 0.00e+00，一致性检查=通过。
- Shifrut：4 folds，172 tasks；保存真值与从原始数据重建真值的最大绝对差 0.00e+00，一致性检查=通过。
- Liang：9 folds，612 tasks；保存真值与从原始数据重建真值的最大绝对差 0.00e+00，一致性检查=通过。
- Tian_CRISPRi：4 folds，732 tasks；保存真值与从原始数据重建真值的最大绝对差 0.00e+00，一致性检查=通过。
- training perturbed-mean 只由各 fold 的训练任务真值构造；验证和测试真值未进入参考量。
- RMSE 对共同平移不敏感，因此没有报告数学等价的“中心化 RMSE”。

## 使用边界

E133 是看到 2025 年方法学文献后增加的次要稳健性审计，不能回写成最初预注册的主终点。它用于回答审稿人对简单基线、系统性平均变化和误差定义的质疑。完整逐任务值、fold 统计与 3,000 次聚类 bootstrap 均在 `tables/`。
