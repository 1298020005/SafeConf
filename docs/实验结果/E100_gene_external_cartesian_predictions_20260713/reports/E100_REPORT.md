# E100｜两套独立多背景遗传扰动矩阵外部复制

Lara ex vivo 与 Santinha 的任务、背景和切分来自 E99 冻结合同。每个数据集先只用所选背景的 control 细胞确定 3,000 基因面板，再统一做 library-size 10,000、log1p 和背景内 mean-difference。测试扰动细胞不参与基因选择、预测、风险特征、校准或阈值。

## 100% 训练量

| dataset | setting | calibrated pair risk ρ | disagreement ρ | magnitude ρ |
|---|---|---:|---:|---:|
| Lara_exvivo | all_test_settings_pooled | 0.255 | 0.085 | 0.043 |
| Lara_exvivo | context_and_perturbation_unseen | -0.007 | -0.007 | 0.043 |
| Lara_exvivo | context_unseen_row | 0.658 | 0.565 | 0.617 |
| Lara_exvivo | perturbation_unseen_column | 0.344 | 0.217 | 0.329 |
| Lara_exvivo | random_missing_pair | 0.624 | 0.661 | 0.493 |
| Santinha | all_test_settings_pooled | 0.176 | 0.342 | 0.385 |
| Santinha | context_and_perturbation_unseen | -0.020 | -0.020 | 0.300 |
| Santinha | context_unseen_row | 0.658 | 0.635 | 0.792 |
| Santinha | perturbation_unseen_column | 0.223 | 0.224 | 0.380 |
| Santinha | random_missing_pair | 0.186 | 0.105 | 0.105 |

## 分层聚类 bootstrap

| dataset | comparator | Δρ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---:|
| Lara_exvivo | safeconf_frozen_pair_risk | 0.026 | [-0.112, 0.129] | 0.662 |
| Lara_exvivo | risk_model_disagreement | 0.170 | [0.024, 0.294] | 0.987 |
| Lara_exvivo | baseline_predicted_magnitude | 0.212 | [0.074, 0.364] | 1.000 |
| Santinha | safeconf_frozen_pair_risk | -0.180 | [-0.438, 0.022] | 0.050 |
| Santinha | risk_model_disagreement | -0.166 | [-0.430, 0.061] | 0.102 |
| Santinha | baseline_predicted_magnitude | -0.209 | [-0.522, 0.115] | 0.122 |

## 合同边界

scGPT embedding 的常规映射采用小鼠符号大写后匹配人类词表，该步骤是 symbol match，不单独声称每个基因都完成一对一同源证明。Gltscr1→BICRA 与 Dgcr14→ESS2 使用 NCBI Gene 记录核对后的别名。模型仍是 embedding/transfer predictor，不写成端到端 scGPT 或 GEARS。任何 cluster CI 跨 0 的增量都只按趋势解释。

strict PredictionRecord issue_count = 0；任务行 2760，预测记录 5520。

- `tables/E100_TASK_RISK_TABLE.csv`
- `tables/PREDICTION_RECORDS.csv`
- `tables/E100_RISK_SUMMARY.csv`
- `tables/E100_CLUSTER_BOOTSTRAP.csv`
- `tables/E100_ORTHOLOG_MAPPING.csv`
- `figures/F1_*_four_setting_risk_spearman.svg`
