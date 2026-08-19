# E89｜sciPlex3 → sciPlex4 CPA 同族外部验证

CPA-RDKit 与 source-dose interpolation 只使用 sciPlex3 的 24 个源任务以及两个数据集的 control 细胞。sciPlex4 的 28 个 perturbed truth 在预测文件落盘后才读取。

- strict PredictionRecord：56，issues=0
- pair mean/max 下界违反：0/0
- pooled disagreement ρ=0.784；magnitude ρ=0.835
- Δρ=-0.051，bootstrap 95% CI [-0.200,0.068]
- CPA / interpolation 胜过零效应比例：35.7% / 67.9%

| group | n_tasks | rho_disagreement_pair_mean | rho_magnitude_pair_mean | delta_rho | bootstrap_delta_ci95_low | bootstrap_delta_ci95_high | top20_error_enrichment | cpa_beats_zero_fraction | interpolation_beats_zero_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 28 | 0.784 | 0.835 | -0.051 | -0.2 | 0.068 | 1.19 | 0.357 | 0.679 |
| exact_source_dose | 16 | 0.703 | 0.862 | -0.159 | -0.39 | 0.009 | 1.186 | 0.375 | 0.625 |
| interpolated_dose | 12 | 0.832 | 0.783 | 0.049 | -0.229 | 0.329 | 1.184 | 0.333 | 0.75 |
| context::A549 | 14 | 0.824 | 0.833 | -0.009 | -0.225 | 0.174 | 1.18 | 0.429 | 0.714 |
| context::MCF7 | 14 | 0.903 | 0.807 | 0.097 | -0.049 | 0.36 | 1.199 | 0.286 | 0.643 |
| drug::Abexinostat | 14 | 0.837 | 0.842 | -0.004 | -0.257 | 0.248 | 1.169 | 0.357 | 0.786 |
| drug::Pracinostat | 14 | 0.719 | 0.798 | -0.079 | -0.416 | 0.211 | 1.21 | 0.357 | 0.571 |

E89 只有两个共享药物、28 个目标任务，属于独立批次复核。解释时同时报告 pooled、精确剂量、插值剂量、细胞系和药物分层；不以单个分层的点估计替代总体区间。
