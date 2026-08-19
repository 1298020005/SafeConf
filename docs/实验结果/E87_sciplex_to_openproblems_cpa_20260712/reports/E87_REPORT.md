# E87｜sciPlex3 → OpenProblems 跨数据集 CPA 审计

CPA-RDKit 与 inductive ridge 只在 sciPlex3 的 108 个扰动任务上学习；OpenProblems 只提供 4 类 PBMC 的 control 表达、141 个药物的 SMILES 和剂量。553 个目标 perturbed truth 在预测文件落盘后才读取。

- strict PredictionRecord：1106，issues=0
- source/target 同名药：0
- pair mean/max 下界违反：0/0

## 风险排序

| score_name | target_error | n_tasks | spearman | bootstrap_ci95_low | bootstrap_ci95_high | top20_error_enrichment |
| --- | --- | --- | --- | --- | --- | --- |
| model_disagreement_rmse | pair_mean_rmse | 553 | 0.993 | 0.989 | 0.996 | 1.624 |
| model_disagreement_rmse | pair_max_rmse | 553 | 0.978 | 0.967 | 0.985 | 1.9 |
| predicted_magnitude_mean | pair_mean_rmse | 553 | 0.994 | 0.989 | 0.996 | 1.625 |

## 预测器与幅度诊断

| group | n_tasks | rho_disagreement_pair_mean | rho_magnitude_pair_mean | delta_rho_disagreement_minus_magnitude | bootstrap_delta_ci95_low | bootstrap_delta_ci95_high | cpa_beats_zero_fraction | ridge_beats_zero_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 553 | 0.993 | 0.994 | -0.0 | -0.001 | -0.0 | 0.0 | 0.0 |
| context::B cells | 137 | 0.996 | 0.997 | -0.0 | -0.002 | 0.001 | 0.0 | 0.0 |
| context::Myeloid cells | 137 | 0.81 | 0.81 | -0.0 | -0.015 | 0.018 | 0.0 | 0.0 |
| context::NK cells | 138 | 0.994 | 0.995 | -0.001 | -0.005 | 0.0 | 0.0 | 0.0 |
| context::T cells | 141 | 0.999 | 0.999 | -0.0 | -0.001 | 0.0 | 0.0 | 0.0 |

分歧与 pair-mean error 的相关性很高，但预测幅度得到几乎相同的排序。两者的差值为 -0.0005，bootstrap 95% CI 为 [-0.0010, -0.0001]。CPA 和 ridge 分别只在 0.0%、0.0% 的目标任务上优于零效应预测；分歧与 ridge 预测幅度的相关性为 0.996。

因此，E87 证明了整个跨数据集预测—落盘—解封—评估链路可以运行，也保留了 pair-risk 下界；它没有证明分歧稳定优于幅度。两个源域预测器在 PBMC 目标域严重失准，当前的高相关主要来自 ridge 外推幅度同时主导分歧与误差。这个结果按负面边界保留，不能作为 SafeConf 跨域独立增益的主证据。
