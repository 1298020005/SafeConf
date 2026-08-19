# E90｜基因侧难设置总表（冻结 GEARS+scGPT 合同）

本实验**不重新训练**预测器，只在已冻结的双模型 PredictionRecord 上统一汇总周老师关心的难设置，并标明哪些还缺真实重训。

## 合同

- predictor_pair: GEARS 三 seed ensemble + 正式微调 scGPT
- 误差: pair_mean_rmse = 0.5*(GEARS_RMSE + scGPT_RMSE)
- 可部署分数: model_disagreement_rmse, predicted_magnitude_mean（真值仅用于评价）
- 数据: Adamson / Norman / Frangieh × 两套不重叠未见基因面板

## 1. 整列未见基因 holdout（主硬设置，已完成）

| group | n | ρ_disagree | ρ_magnitude | Δρ (d−m) | Δ 95% CI | CI>0 |
|---|---:|---:|---:|---:|---:|:---:|
| Adamson_P1 | 24 | 0.494 | 0.250 | 0.243 | [-0.133, 0.646] | N |
| Adamson_P2 | 24 | 0.496 | 0.624 | -0.129 | [-0.299, 0.017] | N |
| Frangieh_P1 | 24 | 0.349 | 0.253 | 0.096 | [-0.043, 0.262] | N |
| Frangieh_P2 | 24 | 0.743 | 0.679 | 0.064 | [-0.159, 0.342] | N |
| Norman_P1 | 24 | 0.589 | 0.361 | 0.228 | [-0.127, 0.623] | N |
| Norman_P2 | 24 | 0.263 | 0.047 | 0.216 | [-0.090, 0.466] | N |

**分层池化（6 面板 / 144 任务）**: ρ_disagree=0.489, ρ_magnitude=0.369, Δ=0.120 CI=[0.007, 0.238] (CI 不含 0)。

此行与 E77 主张一致：基因整列未见扰动上，跨模型分歧相对预测幅度有正增量。

## 2. 跨面板分数迁移（同数据集、不重叠测试基因）

| group | α* | source ρ | target ρ(transfer) | target ρ(dis) | target ρ(mag) | Δ vs mag |
|---|---:|---:|---:|---:|---:|---:|
| Adamson_P1_to_P2 | 0.70 | 0.505 | 0.535 | 0.496 | 0.624 | -0.090 |
| Adamson_P2_to_P1 | 0.00 | 0.624 | 0.250 | 0.494 | 0.250 | 0.000 |
| Frangieh_P1_to_P2 | 0.75 | 0.365 | 0.723 | 0.743 | 0.679 | 0.044 |
| Frangieh_P2_to_P1 | 0.60 | 0.753 | 0.349 | 0.349 | 0.253 | 0.096 |
| Norman_P1_to_P2 | 0.95 | 0.589 | 0.263 | 0.263 | 0.047 | 0.216 |
| Norman_P2_to_P1 | 0.95 | 0.263 | 0.589 | 0.589 | 0.361 | 0.228 |

说明：α 只在 source 面板上按 Spearman 网格选择；target 真值不参与拟合。这是设置难度的中间台阶，不是完整训练子矩阵。

## 3. 跨数据集风险校准器迁移（E69，已有）

| group | metric | value |
|---|---|---|
| Adamson->Norman::gears_ensemble_rmse::magnitude | target Spearman | -0.377 (top20=1.100) |
| Adamson->Norman::gears_ensemble_rmse::magnitude_plus_disagreement | target Spearman | 0.708 (top20=1.284) |
| Adamson->Norman::scgpt_finetuned_rmse::magnitude | target Spearman | 0.340 (top20=1.139) |
| Adamson->Norman::scgpt_finetuned_rmse::magnitude_plus_disagreement | target Spearman | 0.355 (top20=1.139) |
| Adamson->Norman::task_mean_rmse::magnitude | target Spearman | 0.434 (top20=1.191) |
| Adamson->Norman::task_mean_rmse::magnitude_plus_disagreement | target Spearman | 0.547 (top20=1.191) |
| Adamson->Norman::task_max_rmse::magnitude | target Spearman | 0.403 (top20=1.132) |
| Adamson->Norman::task_max_rmse::magnitude_plus_disagreement | target Spearman | 0.543 (top20=1.456) |
| Norman->Adamson::gears_ensemble_rmse::magnitude | target Spearman | 0.076 (top20=0.937) |
| Norman->Adamson::gears_ensemble_rmse::magnitude_plus_disagreement | target Spearman | 0.335 (top20=1.377) |
| Norman->Adamson::scgpt_finetuned_rmse::magnitude | target Spearman | 0.371 (top20=1.229) |
| Norman->Adamson::scgpt_finetuned_rmse::magnitude_plus_disagreement | target Spearman | 0.557 (top20=1.453) |
| Norman->Adamson::task_mean_rmse::magnitude | target Spearman | 0.284 (top20=1.146) |
| Norman->Adamson::task_mean_rmse::magnitude_plus_disagreement | target Spearman | 0.471 (top20=1.418) |
| Norman->Adamson::task_max_rmse::magnitude | target Spearman | 0.369 (top20=1.121) |
| Norman->Adamson::task_max_rmse::magnitude_plus_disagreement | target Spearman | 0.590 (top20=1.481) |
| Adamson->Norman::gears_ensemble_rmse | Δρ combined−mag CI | 1.085 [0.489, 1.598] reliable=True |
| Adamson->Norman::scgpt_finetuned_rmse | Δρ combined−mag CI | 0.015 [-0.116, 0.167] reliable=False |
| Adamson->Norman::task_mean_rmse | Δρ combined−mag CI | 0.113 [-0.026, 0.289] reliable=False |
| Adamson->Norman::task_max_rmse | Δρ combined−mag CI | 0.139 [-0.061, 0.377] reliable=False |
| Norman->Adamson::gears_ensemble_rmse | Δρ combined−mag CI | 0.259 [-0.393, 0.813] reliable=False |
| Norman->Adamson::scgpt_finetuned_rmse | Δρ combined−mag CI | 0.186 [-0.029, 0.473] reliable=False |
| Norman->Adamson::task_mean_rmse | Δρ combined−mag CI | 0.187 [-0.049, 0.483] reliable=False |
| Norman->Adamson::task_max_rmse | Δρ combined−mag CI | 0.221 [-0.171, 0.678] reliable=False |

E69 结论保持：仅部分方向 combined 稳定超过 magnitude；不能写普遍跨数据集增益。

## 4. 整行 holdout / 训练子矩阵（E97/E98 后续补充）

| panel | n_train | n_test | row_holdout | train_submatrix |
|---|---:|---:|---|---|
| Adamson_P1 | 48 | 24 | N/A（单 context） | 需重训 GEARS/scGPT |
| Adamson_P2 | 48 | 24 | N/A（单 context） | 需重训 GEARS/scGPT |
| Norman_P1 | 183 | 24 | N/A（单 context） | 需重训 GEARS/scGPT |
| Norman_P2 | 183 | 24 | N/A（单 context） | 需重训 GEARS/scGPT |
| Frangieh_P1 | 170 | 24 | 旧导出丢失真实 context | 由 E97/E98 另建合同 |
| Frangieh_P2 | 170 | 24 | 旧导出丢失真实 context | 由 E97/E98 另建合同 |

E90 生成时根据旧 PredictionRecord 将 Frangieh 误判为单 context。E97 回查原始 h5ad 后确认其含 `Control`、`IFNγ`、`Co-culture` 三个真实背景；按每个 pair 至少 50 个细胞可形成完整的 3×189 遗传扰动矩阵。E98 已在该矩阵执行 25%/50%/75%/100% 训练子矩阵、整行、整列、双未见和随机 pair，并生成 7,416 条 strict PredictionRecord。

E98 的两个预测器为 SourceEffect-scGPTKNN 与 scGPTEmbedding-ContextRidge。它们完成了矩阵合同和输入防泄漏核验，不等同于 E90 的 GEARS+端到端 scGPT 双模型重训。100% 训练量 pooled ρ 为 0.693，分歧为 0.596，magnitude 为 0.643；但 outer-fold+perturbation cluster bootstrap 的增量区间仍跨 0。因此“GEARS/scGPT 合同下整行已闭环”仍不能写。

## 5. 现在能写 / 不能写

**能写**：基因扰动、整列未见基因、双模型分歧相对预测幅度的排序增益（E77/E90 col_holdout）；Frangieh 三背景矩阵上的四类难设置和训练量敏感性已经可计算（E97/E98）。

**不能写**：基因侧小矩阵训练与整行 holdout 已在 GEARS+端到端 scGPT 合同下完成；E98 稳定超过 magnitude；跨数据集普遍超过 magnitude。

生成时间：2026-07-12T22:17:07  git：`77b476ee7991ef662071a615a2a1122becf5166f`
