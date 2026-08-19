# E141｜七数据 PROGENy 通路忠实度审计

## 预注册 gate：通过

该分析把每条 RNA 预测投影到 14 类有符号 PROGENy 通路活动；资源、风险分数和分析规则在打开预测/真值向量前冻结。

| 风险分数 | 通路误差 | 七数据等权 ρ | 95% CI | 相对 magnitude Δρ（95% CI） |
|---|---|---:|---:|---:|
| safeconf_calibrated_pair_risk | progeny_activity_rmse_mean | 0.117 | [0.029, 0.206] | +0.043 [-0.060, +0.135] |
| directional_risk_frozen | progeny_activity_cosine_error_mean | 0.005 | [-0.099, 0.105] | +0.113 [-0.038, +0.255] |

## 每数据集

| dataset | SafeConf→pathway RMSE | Directional→pathway cosine | pathways |
|---|---:|---:|---:|
| Frangieh | 0.176 | -0.128 | 13 |
| Lara_exvivo | 0.267 | 0.191 | 11 |
| Liang | 0.156 | 0.141 | 13 |
| Nadig_two_cellline | 0.182 | -0.073 | 10 |
| Santinha | 0.079 | -0.106 | 14 |
| Shifrut | -0.110 | 0.124 | 13 |
| Tian_CRISPRi | 0.069 | -0.115 | 12 |

## 高风险任务中残差较大的通路

- TNFa: mean absolute activity residual=0.1356
- JAK-STAT: mean absolute activity residual=0.1177
- Hypoxia: mean absolute activity residual=0.1154
- NFkB: mean absolute activity residual=0.1136
- MAPK: mean absolute activity residual=0.1025
- EGFR: mean absolute activity residual=0.0943
- TGFb: mean absolute activity residual=0.0934
- Trail: mean absolute activity residual=0.0898

## 边界

PROGENy 是由外部扰动实验得到的转录响应足迹，不等于蛋白磷酸化或因果通路真值。512 基因固定面板使部分通路只能由少量响应基因覆盖；覆盖数逐数据集完整落盘，不能把本分析写成湿实验因果证明。
