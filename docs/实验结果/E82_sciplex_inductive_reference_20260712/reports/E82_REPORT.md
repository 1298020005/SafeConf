# E82｜sciPlex3 四象限参考预测管线

E82 用 E81 冻结的 9 个子矩阵设置跑通两个来源域参考预测器。预测阶段只读取训练任务的 perturbed expression；新 context 只使用 vehicle control，新药只使用外部 SMILES 字符结构和剂量。预测文件落盘后，评价阶段才读取测试任务真值。

这一步验证合同和统计流程，不把 ridge/kNN 写成 CPA 或化学主结果。正式模型仍需 CPA/chemCPA。

- 测试预测：688 tasks per predictor（按 manifest 计）
- 真实生物任务：103
- target truth 进入 score：否
- gene order hash：`sha256:e53bfe14230003dbba9396958fae945532976804da8c5950a489b1f269953618`

## 四象限汇总（9 个 manifest 的描述统计）

| quadrant | score_name | target_error | n_manifests | mean_spearman | median_spearman | min_spearman | max_spearman | positive_manifests |
|---|---|---|---|---|---|---|---|---|
| new_context_new_perturbation | model_disagreement_rmse | pair_mean_rmse | 9 | 0.321 | 0.158 | -0.066 | 0.738 | 8 |
| new_context_new_perturbation | predicted_magnitude_mean | pair_mean_rmse | 9 | 0.706 | 0.754 | 0.263 | 0.976 | 9 |
| new_context_seen_perturbation | model_disagreement_rmse | pair_mean_rmse | 9 | 0.588 | 0.649 | 0.143 | 0.841 | 9 |
| new_context_seen_perturbation | predicted_magnitude_mean | pair_mean_rmse | 9 | 0.879 | 0.889 | 0.648 | 1.0 | 9 |
| seen_context_new_perturbation | model_disagreement_rmse | pair_mean_rmse | 9 | 0.434 | 0.449 | 0.074 | 0.811 | 9 |
| seen_context_new_perturbation | predicted_magnitude_mean | pair_mean_rmse | 9 | 0.528 | 0.524 | 0.343 | 0.709 | 9 |
| seen_context_seen_perturbation_pair_holdout | model_disagreement_rmse | pair_mean_rmse | 9 | 0.589 | 0.636 | 0.2 | 0.886 | 9 |
| seen_context_seen_perturbation_pair_holdout | predicted_magnitude_mean | pair_mean_rmse | 9 | 0.354 | 0.5 | -0.8 | 1.0 | 7 |

## 这批结果暴露出的边界

- 在新 context、双未见和新药三个较难象限，predicted magnitude 的平均相关分别为 0.879、0.706、0.528，均高于参考模型分歧的 0.588、0.321、0.434。分歧只在随机缺失 pair 象限的描述统计中高于 magnitude（0.589 vs 0.354）。这不支持把参考模型分歧写成难 setting 下的稳定增量。
- 整行只留出一个 context 时，context similarity 对该行所有任务是同一个数，无法在行内排序；整列新药时，历史支持都为 0，也无法在列内排序。表里的 NaN 是特征在该 setting 下退化为常数，不是程序漏算。
- p25 的子矩阵内 pair holdout 每个 manifest 只有 4 个任务，相关系数区间很宽，只能作为管线检查。正式统计以较大象限、跨 manifest 汇总和 CPA/chemCPA 复核为准。

这说明后续不能把四项特征在所有 split 中机械相加。每个 setting 需要先列出可辨识特征：整行依赖扰动侧与模型输出，整列依赖结构表征、context 差异与模型输出，双未见主要依赖外部表征和模型输出。

完整表：`tables/E82_RISK_ERROR_SUMMARY.csv` 与 `tables/E82_MANIFEST_AGGREGATE.csv`。单 manifest 的负相关、区间跨 0 和四象限差异全部保留；上表只是描述统计，不把 9 个 manifest 当成 9 个独立数据集。
