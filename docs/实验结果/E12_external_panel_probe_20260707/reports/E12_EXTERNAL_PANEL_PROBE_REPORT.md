# E12 外部面板扩展探针报告

生成时间：2026-07-07 06:34

## 1. 结论

E12 在 6 个外部候选数据集上启动任务级探针。`KaggleCrossPatient`、`crossPatient`、`sciplex3` 可进入 held-out pair 测试；`kangCrossCell`、`kangCrossPatient`、`TCDD` 因只有 1 个 perturbation，在当前 held-out pair 规则下没有 test pair。

可评估部分共 308 条 test 记录。overall 最强信号是 `model_disagreement_risk`：aligned Spearman = 0.734，80% coverage 平均 RMSE 改善 = 10.04%。`simple_combined_confidence` 的 aligned Spearman = 0.537，80% coverage 改善 = 11.51%。`learned_risk_score` 为正但弱于 disagreement：aligned Spearman = 0.446。

## 2. 数据集适配结果

| dataset | n_tasks | n_contexts | n_perturbations | context_col | perturbation_col | n_test | e12_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kangCrossCell | 8 | 8 | 1 | cell_type | perturbation | 0 | not_evaluable_under_heldout_pair |
| kangCrossPatient | 8 | 8 | 1 | condition1 | perturbation | 0 | not_evaluable_under_heldout_pair |
| KaggleCrossPatient | 40 | 6 | 10 | cell_type | perturbation | 40 | evaluable |
| crossPatient | 10 | 6 | 2 | condition1 | perturbation | 8 | evaluable |
| TCDD | 6 | 6 | 1 | condition1 | perturbation | 0 | not_evaluable_under_heldout_pair |
| sciplex3 | 106 | 3 | 36 | cell_type | perturbation | 106 | evaluable |

## 3. 每数据集主结果

| dataset_name | simple_combined_confidence_aligned_rho | simple_combined_confidence_risk_cov_80_improve_pct | learned_risk_score_aligned_rho | learned_risk_score_risk_cov_80_improve_pct | model_disagreement_risk_aligned_rho | model_disagreement_risk_risk_cov_80_improve_pct |
| --- | --- | --- | --- | --- | --- | --- |
| KaggleCrossPatient | 0.618747 | 14.3832 | 0.00923478 | -1.9435 | 0.468595 | 12.0946 |
| crossPatient | 0.0753331 | 2.44117 | 0 | 0 | 0.597644 | -0.167093 |
| sciplex3 | 0.622404 | 17.6907 | 0.425516 | 7.80326 | 0.884577 | 18.1931 |

## 4. Overall 信号排序

| score_name | score_type | n | direction_aligned_spearman | risk_cov_80_improve_pct | high_low_rmse_gap |
| --- | --- | --- | --- | --- | --- |
| model_disagreement_risk | risk | 308 | 0.734406 | 10.0402 | 0.0646967 |
| support_count_score | confidence | 308 | 0.62474 | 7.14468 | 0.0378299 |
| historical_residual_risk | risk | 308 | 0.596586 | -0.809355 | 0.0149588 |
| simple_combined_confidence | confidence | 308 | 0.537476 | 11.505 | 0.0520479 |
| learned_risk_score | risk | 296 | 0.446422 | 1.95325 | 0.000757266 |
| context_similarity_score | confidence | 308 | 0.335464 | -0.735439 | -0.00375039 |
| random_score | confidence | 308 | 0.0804813 | 2.44913 | 0.00740409 |
| ood_distance_risk | risk | 308 | -0.0806449 | -6.36369 | -0.0351426 |
| perturbation_stability_score | confidence | 178 | -0.219792 | -3.33817 | -0.0314556 |
| prediction_magnitude_risk | risk | 308 | -0.27135 | 1.01228 | 0.00787534 |

## 5. 必须保留的边界

- E12 实际可评估数据集为 3 个；另外 3 个候选需要换验证定义。
- `sciplex3` 是当前最强外部 chemical 信号：model disagreement aligned rho = 0.885。
- `crossPatient` 的 simple combined 只有 0.075，说明组合分数跨数据集仍不稳。
- learned risk 没有超过 model disagreement；这与 E10 一致。
- 当前仍是轻量参考预测器探针；GEARS/CPA/scGPT 逐模型向量验证尚未完成。

## 6. 下一步

1. 对 `sciplex3` 单独做 chemical-focused frozen split，保留 disagreement、support、simple combined、magnitude 的统一比较。
2. 对 `KaggleCrossPatient` 做 context/patient 层解释，检查 support_count 与 context similarity 为什么比 learned 更稳定。
3. 对 `kangCrossCell/kangCrossPatient/TCDD` 改用 leave-context-out 或 chemical dose split；held-out pair 不适合只有一个 perturbation 的数据。
