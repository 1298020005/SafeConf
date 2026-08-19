# E147｜扰动身份检索审计

这是已用于 SafeConf 开发与评价的七数据上的新终点，不是新增独立验证。查询为模型预测效应；候选库为同一 dataset、fold 和 context 的测试真值。归一化正确秩越高越好，随机期望约 0.5；检索误差等于 1−正确秩。

## 检索本身

| predictor | similarity | 七数据等权正确秩 | top-1 | top-5 | 有效查询 |
|---|---|---:|---:|---:|---:|
| GEARS | cosine | 0.621 | 0.166 | 0.359 | 2441 |
| scGPT | cosine | 0.500 | 0.037 | 0.183 | 2441 |
| GEARS | pearson | 0.621 | 0.167 | 0.360 | 2441 |
| scGPT | pearson | 0.500 | 0.037 | 0.183 | 2441 |

## 风险与检索误差

正相关表示风险分数越高，模型越难在同背景候选中找回正确扰动。点估计按 fold→dataset→七数据等权聚合；区间按 dataset 内唯一 context×perturbation 整簇 bootstrap。每次重抽后，dataset×fold×predictor×similarity 的加权有效查询仍须不少于 10，否则该折在该次抽样中记为缺失。

| score | predictor | similarity | 七数据等权 ρ | 95% CI |
|---|---|---|---:|---:|
| baseline_predicted_magnitude | GEARS | cosine | -0.198 | [-0.245, -0.139] |
| baseline_predicted_magnitude | scGPT | cosine | -0.079 | [-0.134, -0.020] |
| baseline_predicted_magnitude | GEARS | pearson | -0.197 | [-0.245, -0.137] |
| baseline_predicted_magnitude | scGPT | pearson | -0.028 | [-0.083, 0.031] |
| directional_risk_frozen | GEARS | cosine | 0.120 | [0.062, 0.174] |
| directional_risk_frozen | scGPT | cosine | -0.044 | [-0.098, 0.012] |
| directional_risk_frozen | GEARS | pearson | 0.133 | [0.074, 0.186] |
| directional_risk_frozen | scGPT | pearson | 0.001 | [-0.056, 0.059] |
| risk_model_disagreement | GEARS | cosine | -0.192 | [-0.246, -0.132] |
| risk_model_disagreement | scGPT | cosine | -0.053 | [-0.106, 0.002] |
| risk_model_disagreement | GEARS | pearson | -0.187 | [-0.242, -0.128] |
| risk_model_disagreement | scGPT | pearson | 0.004 | [-0.055, 0.056] |
| safeconf_calibrated_pair_risk | GEARS | cosine | -0.039 | [-0.097, 0.018] |
| safeconf_calibrated_pair_risk | scGPT | cosine | -0.057 | [-0.115, 0.000] |
| safeconf_calibrated_pair_risk | GEARS | pearson | -0.037 | [-0.095, 0.022] |
| safeconf_calibrated_pair_risk | scGPT | pearson | -0.026 | [-0.084, 0.033] |

## Directional-SafeConf 的逐数据集结果

| dataset | GEARS Pearson | GEARS cosine | scGPT Pearson | scGPT cosine |
|---|---:|---:|---:|---:|
| Frangieh | 0.028 | 0.015 | -0.030 | -0.033 |
| Lara_exvivo | 0.499 | 0.521 | -0.122 | -0.098 |
| Liang | 0.191 | 0.195 | 0.052 | 0.054 |
| Nadig_two_cellline | 0.210 | 0.186 | 0.008 | -0.141 |
| Santinha | -0.153 | -0.174 | 0.050 | -0.058 |
| Shifrut | 0.121 | 0.086 | 0.049 | -0.025 |
| Tian_CRISPRi | 0.033 | 0.013 | 0.001 | -0.009 |

## 判读

Directional-SafeConf 对 GEARS 的两个预指定检索误差均为稳定正相关：Pearson 检索 95% CI [0.074, 0.186]，cosine 检索 [0.062, 0.174]；七个数据集中六个方向为正，Santinha 为负。
相对原 SafeConf，Directional-SafeConf 的条件差值为 Pearson +0.168 [+0.111, +0.220]、cosine +0.157 [+0.100, +0.212]。这是方向风险模型在既有数据上的终点一致性，不是独立确认。
scGPT 的正确秩在两种相似度下均约为随机期望 0.5。这里审计的是保存为 `scGPT_context_mean_finetuned` 的具体均值型预测记录，不能外推为所有 scGPT 模型都没有扰动身份信息。
magnitude 和 disagreement 对 GEARS 检索误差为负，说明大效应任务在身份检索上反而更容易；原 SafeConf 因包含绝对难度成分，在该终点上没有正相关。检索身份和逐基因方向误差回答的是不同问题。

## Bootstrap 门槛修复与敏感性

3,000 次抽样共产生 384000 个 fold×predictor×similarity 评价，其中 1396 个加权有效查询少于 10；受影响的 bootstrap draw 为 339/3000。按不重复的 dataset×fold 计，共 349/96000 个评价低于门槛，涉及 339/3000 个 draw。主区间已经排除这些折，不再沿用修复前的三查询下限。

| Directional-SafeConf 终点 | 严格≥10：median [95% CI] | 不设10门槛诊断：median [95% CI] | median变化 |
|---|---:|---:|---:|
| scGPT / pearson | 0.001 [-0.056, 0.059] | 0.001 [-0.056, 0.059] | +0.0000 |
| scGPT / cosine | -0.043 [-0.098, 0.012] | -0.043 [-0.098, 0.012] | +0.0002 |
| GEARS / pearson | 0.130 [0.074, 0.186] | 0.130 [0.074, 0.186] | +0.0000 |
| GEARS / cosine | 0.117 [0.062, 0.174] | 0.118 [0.062, 0.174] | -0.0001 |

全部直接关联和差值指标中，严格门槛相对不设10门槛诊断的最大 median 变化为 0.0003。该敏感性只用于核查门槛实现，不替代主结果。
低于门槛最频繁的 fold-endpoint 及次数已保存到 `tables/E147_BOOTSTRAP_THRESHOLD_QC_BY_FOLD_ENDPOINT.csv`；该表共有 128 行。

共有 124 个 metric-specific 候选池满足规则；无效查询 0 条。

## 边界

身份检索衡量预测效应是否更像正确扰动，而逐基因 RMSE/cosine 衡量向量误差，二者不可互换。候选库难度受同背景内候选数和扰动相似性影响；本分析通过同 fold、同 context 和归一化秩限制了明显偏差，但数据仍全部被此前实验使用过。
bootstrap 固定这七个数据集并只在各数据集内部重抽生物任务簇，因此区间是对当前七数据的条件不确定性，不包含‘换一批数据集’的研究间异质性。
