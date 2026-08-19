# E98｜Frangieh 三背景遗传扰动矩阵正式预测与风险审计

## 回答周老师的输入问题

每个测试任务的预测输入只有：训练子矩阵内已观测的扰动效应、目标背景的未扰动 control 表达、扰动基因的 scGPT 预训练 embedding。SourceEffect-scGPTKNN 先查同一扰动在训练背景的效应；整列新扰动没有历史效应时，按 scGPT embedding 找相邻训练基因。ContextRidge 使用 scGPT embedding、目标背景 control 和两者交互项，由训练 pair 拟合。两个预测向量的分歧和预测幅度在 30 个 validation pair 上校准，训练支持数与背景 control 距离提供结构性新颖度。测试扰动后的真实表达没有进入预测、分数或阈值，只在这些量冻结后计算 RMSE。

## 100% 训练子矩阵结果

| setting | SafeConf 校准 pair risk ρ | disagreement ρ | magnitude ρ |
|---|---:|---:|---:|
| all_test_settings_pooled | 0.693 | 0.596 | 0.643 |
| context_and_perturbation_unseen | 0.420 | 0.418 | 0.417 |
| context_unseen_row | 0.809 | 0.780 | 0.806 |
| perturbation_unseen_column | 0.546 | 0.201 | 0.511 |
| random_missing_pair | 0.684 | 0.680 | 0.704 |

表中数值是三个整行留出 fold 的 Spearman 宏平均。`all_test_settings_pooled` 把四类任务放回同一个待质检队列，同时检查任务类型之间和同类型内部的排序。完整的 25%/50%/75%/100% 训练量、各 fold、风险覆盖率和验证阈值结果保存在 `tables/E98_RISK_SUMMARY.csv`。

## 与强基线的配对 bootstrap

| comparator | bootstrap unit | Δρ（SafeConf−基线） | 95% CI | P(Δ>0) |
|---|---|---:|---:|---:|
| safeconf_frozen_pair_risk | task_row | 0.006 | [-0.027, 0.039] | 0.650 |
| safeconf_frozen_pair_risk | outer_fold_plus_perturbation_cluster | 0.006 | [-0.073, 0.088] | 0.572 |
| risk_model_disagreement | task_row | 0.097 | [0.049, 0.145] | 1.000 |
| risk_model_disagreement | outer_fold_plus_perturbation_cluster | 0.097 | [-0.022, 0.253] | 0.911 |
| baseline_predicted_magnitude | task_row | 0.050 | [0.004, 0.096] | 0.981 |
| baseline_predicted_magnitude | outer_fold_plus_perturbation_cluster | 0.050 | [-0.098, 0.255] | 0.675 |

## validation q80 阈值审计

| setting | 实际接受比例 | 接受任务 RMSE | 全部任务 RMSE |
|---|---:|---:|---:|
| all_test_settings_pooled | 0.577 | 0.0413 | 0.0446 |
| context_and_perturbation_unseen | 0.644 | 0.0447 | 0.0438 |
| context_unseen_row | 0.356 | 0.0433 | 0.0470 |
| perturbation_unseen_column | 0.989 | 0.0389 | 0.0392 |
| random_missing_pair | 0.856 | 0.0417 | 0.0437 |

q80 是 validation 分位阈值，不是覆盖率保证。尤其在双未见任务上，validation 的任务类型不匹配，接受集误差没有下降；该结果保留为下一轮分层校准的直接依据。

## 解释边界

两个预测器是可复现的训练数据预测器：一个做来源效应迁移与 scGPT embedding 邻域插值，一个做 scGPT embedding 和背景 control 的监督 Ridge。第二个使用了 scGPT 预训练表示，但不是 scGPT Transformer 的端到端微调；第一个也不是 GEARS。E98 因而直接验证矩阵 setting、验证集校准和输入防泄漏，不替代后续同合同下的 GEARS/scGPT 正式重训。配对 bootstrap 若跨过 0，只能写成趋势。跨数据集结果仍引用已完成的 sciPlex 压力测试，不把其负结果改写为成功。

strict PredictionRecord issue_count = 0；共 3708 个任务行、7416 条双预测器记录。

## 文件

- 任务级分数与误差：`tables/E98_TASK_RISK_TABLE.csv`
- 严格预测合同：`tables/PREDICTION_RECORDS.csv`
- 每折每 setting 汇总：`tables/E98_RISK_SUMMARY.csv`
- Ridge 验证选参：`tables/E98_RIDGE_VALIDATION.csv`
- 校准风险配对 bootstrap：`tables/E98_POOLED_BOOTSTRAP.csv`
- 白底结果图：`figures/F1_four_setting_risk_spearman.svg`
