# E134｜Systema 表达空间精确定义审计

## 结果

按训练受扰动表达质心的精确定义，SafeConf 对 centered Pearson 方向误差的六数据集等权宏平均 Spearman 为 **-0.039**，对 centered cosine 方向误差为 **-0.032**，对正式模型相对简单质心预测器的超额 RMSE 为 **-0.052**。

两模型 ensemble 相对训练受扰动表达质心的 RMSE 优势，扰动聚类 bootstrap 中位数为 **0.1651**，95% CI **[0.1621, 0.1679]**。正值表示 ensemble 更好。

## 每数据集简单基线

| 数据集 | tasks | ensemble RMSE | perturbed-centroid RMSE | ensemble − simple | ensemble 胜出任务比例 |
|---|---:|---:|---:|---:|---:|
| Frangieh | 837 | 0.0532 | 0.1470 | -0.0938 | 98.8% |
| Lara_exvivo | 345 | 0.0983 | 0.2295 | -0.1312 | 98.6% |
| Liang | 612 | 0.0794 | 0.2724 | -0.1930 | 100.0% |
| Santinha | 255 | 0.0502 | 0.4197 | -0.3695 | 100.0% |
| Shifrut | 172 | 0.0662 | 0.2780 | -0.2118 | 100.0% |
| Tian_CRISPRi | 732 | 0.0961 | 0.0864 | +0.0097 | 14.3% |

## 六数据集等权排序

| 分数 | original RMSE | centered Pearson | centered cosine | excess RMSE |
|---|---:|---:|---:|---:|
| baseline_predicted_magnitude | 0.093 | 0.038 | 0.028 | 0.034 |
| risk_model_disagreement | 0.049 | 0.011 | 0.004 | -0.013 |
| safeconf_calibrated_pair_risk | 0.204 | -0.039 | -0.032 | -0.052 |
| safeconf_frozen_pair_risk | 0.145 | -0.040 | -0.039 | -0.063 |

## 与 E133 的关系

E133 检查的是训练平均效应，E134 检查的是训练受扰动表达质心。跨背景时两者不同，两个结果并列保留。E134 是 E133 解封后对文献公式的技术校正，不能写成最初主终点。

## 完整性

- Frangieh：3 folds，837 tasks；真值重建最大绝对差 4.77e-06，检查=通过。
- Lara_exvivo：5 folds，345 tasks；真值重建最大绝对差 0.00e+00，检查=通过。
- Santinha：5 folds，255 tasks；真值重建最大绝对差 0.00e+00，检查=通过。
- Shifrut：4 folds，172 tasks；真值重建最大绝对差 0.00e+00，检查=通过。
- Liang：9 folds，612 tasks；真值重建最大绝对差 0.00e+00，检查=通过。
- Tian_CRISPRi：4 folds，732 tasks；真值重建最大绝对差 0.00e+00，检查=通过。
