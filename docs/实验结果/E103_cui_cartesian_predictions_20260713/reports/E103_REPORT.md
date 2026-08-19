# E103｜Cui 六背景细胞因子刺激矩阵

E103 使用 E102 事先冻结的 41 个直接 scGPT-token 映射刺激。目标背景 control、训练 pair 效应和预训练 token embedding 可用于预测；测试刺激后的表达在预测、风险特征和 validation 校准阶段锁定。

## 100% 训练量

| setting | calibrated pair risk ρ | frozen pair risk ρ | disagreement ρ | magnitude ρ |
|---|---:|---:|---:|---:|
| all_test_settings_pooled | 0.413 | 0.303 | 0.288 | 0.391 |
| context_and_perturbation_unseen | 0.289 | 0.097 | 0.097 | 0.353 |
| context_unseen_row | 0.717 | 0.622 | 0.619 | 0.700 |
| perturbation_unseen_column | 0.155 | 0.061 | 0.061 | 0.179 |
| random_missing_pair | 0.627 | 0.417 | 0.417 | 0.571 |

## 聚类 bootstrap

| primary | comparator | Δρ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---:|
| safeconf_calibrated_pair_risk | safeconf_frozen_pair_risk | 0.111 | [0.034, 0.191] | 0.999 |
| safeconf_calibrated_pair_risk | risk_model_disagreement | 0.125 | [0.039, 0.230] | 0.997 |
| safeconf_calibrated_pair_risk | baseline_predicted_magnitude | 0.022 | [-0.104, 0.193] | 0.641 |

## 边界

E103 是 cytokine stimulus，不与 gene knockout 混成同一生物主表；它回答周老师提出的“不同扰动类型都看看”。仅 41/86 个刺激有无需手工别名的直接词表映射，结果只代表该可审计子集。预测器使用 scGPT embedding，不是端到端 scGPT 或 GEARS。

strict PredictionRecord issue_count = 0；任务行 2832，预测记录 5664。

- `tables/E103_TASK_RISK_TABLE.csv`
- `tables/PREDICTION_RECORDS.csv`
- `tables/E103_RISK_SUMMARY.csv`
- `tables/E103_CLUSTER_BOOTSTRAP.csv`
- `figures/F1_Cui_direct41_four_setting_risk_spearman.svg`
