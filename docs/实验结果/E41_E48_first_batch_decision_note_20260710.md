# E41-E48 第一批实验决策记录

这份记录把 E40 的“数据线总览”往前推了一步：哪些已经实际跑过，哪些能进正式实验，哪些先作为字段审计留着。

## 1. 已经实际跑完的内容

| 编号 | 内容 | 输出目录 | 状态 |
| --- | --- | --- | --- |
| E41A | OpenProblems / NeurIPS 2023 Kaggle DGE：官方 prediction vs test logFC，计算真实误差和风险代理 | `E41_multidim_first_batch_smoke_20260710` | 已完成 |
| E41B | Tahoe raw 已完成 shard 字段审计：drug、cell line、MoA、SMILES、PubChem、plate | `E41_multidim_first_batch_smoke_20260710` | 已完成 |
| E42 | sciplex3 cell-line holdout | `E42_E48_local_first_batch_smoke_20260710` | 已完成 smoke |
| E43 | TCDD dose holdout / celltype 检查 | `E42_E48_local_first_batch_smoke_20260710` | 已完成 smoke |
| E44 | KaggleCrossPatient donor holdout | `E42_E48_local_first_batch_smoke_20260710` | 已完成 smoke |
| E45 | crossSpecies species holdout | `E42_E48_local_first_batch_smoke_20260710` | 已完成 smoke |
| E46 | Norman single-to-combo additive baseline | `E42_E48_local_first_batch_smoke_20260710` | 已完成 smoke |
| E47 | Gasperini regulatory sparsity audit | `E42_E48_local_first_batch_smoke_20260710` | 已完成字段审计 |
| E48 | Papalexi RNA-protein consistency | `E42_E48_local_first_batch_smoke_20260710` | 已完成 smoke |

## 2. 现在能讲出来的结果

- OpenProblems：255 个测试任务，官方预测和真实 logFC 可直接对齐。`risk_predicted_magnitude` 与 `rmse_all` 的 Spearman 约 0.84，top 20% 高风险任务的平均误差约为全体的 1.53 倍。这里说明小分子 benchmark 上确实有很强的“任务难度/效应幅度”信号。
- sciplex3：cell-line holdout 下，`risk_safeconf_smoke` 与 error 的 Spearman 约 0.80–0.83，top 20% enrichment 约 1.36；`risk_predicted_magnitude` 更强，约 0.86–0.88。这条线适合进入正式补充实验。
- KaggleCrossPatient：donor holdout 有中等信号，预测幅度 Spearman 约 0.49–0.54。它适合回答“供体变化是否造成风险”，但样本任务数只有 30，正式实验要谨慎解释。
- TCDD dose：剂量任务已经修通，48 个任务、8 个非零剂量。当前 smoke 里 disagreement 只有弱正相关，SafeConf smoke 接近 0。这不是坏事，它提示 TCDD 需要 dose-aware 设计，不能强行套普通 support/context/disagreement。
- crossSpecies：12 个任务，物种留出任务太少，只能作为压力测试，不适合单独支撑大结论。
- Norman：125 个组合扰动，单基因加和 baseline 的 error 能被 predicted magnitude 中等解释，Spearman 约 0.43，top 20% enrichment 约 1.27。可作为组合扰动方向的第一条证据。
- Gasperini：字段非常稀疏，16532 个 perturbation，其中 15969 个低于 15 个细胞，15225 个像 genomic coordinate。它暂时更适合当“regulatory 稀疏标签为什么难”的证据，不适合直接做主结果。
- Papalexi：94 个 RNA/protein 对齐扰动，RNA effect magnitude 与 protein effect magnitude 的 Spearman 约 0.50。这能作为多模态一致性的补充线。

## 3. 不能过度解读的地方

- OpenProblems 这版用的是官方 `prediction.h5ad`，不是 SafeConf 自己训练的预测模型。它能用于风险排序 smoke，但不能写成“我们的方法在 OpenProblems 上已经正式打败谁”。
- OpenProblems 中很多 test drug 在 train 中已有 SMILES / drug 支持，`nearest_train_smiles_jaccard` 大量为 1，因此这版 chemical similarity 没拉开差异。后续要做真正 compound holdout 或 MoA holdout。
- TCDD 当前是 dose-as-perturbation 的 smoke。剂量关系有顺序和单调性，后面要改成 dose-aware risk，不要只用普通分类标签。
- Gasperini 的字段审计已经说明稀疏很严重。正式建模前需要先定 target 粒度：guide、gene/TSS、coordinate 不能混成一个标签。

## 4. 下一步真正值得做的正式化顺序

1. sciplex3 formal：扩大基因数，做 3 个 cell-line holdout，和 magnitude baseline 正面对比。
2. OpenProblems formal：用 Kaggle DGE 做 public/private、B/Myeloid、compound holdout；如果 MoA annotation 完整，再加 MoA holdout。
3. Norman formal：把 single-to-combo 拓展成 gene-overlap split，区分“两个单基因都见过”和“组合从没见过”。
4. TCDD redesign：把 dose 当连续变量，加入 dose rank / log dose / monotonic trend，而不是只当普通 perturbation。
5. Tahoe raw：继续等下载，同时用已完成 shard 做 drug/cell_line/MoA/plate 采样表，后面和 Tahoe D1-D5 pseudobulk 结论对上。
6. Papalexi / Frangieh：作为多模态补充，不抢主线位置。

## 5. 给老师汇报时的自然口径

可以这样说：

“我没有只继续堆 Tahoe。现在先把老师说的多维度数据拆开了：一个是 OpenProblems 这种独立小分子 benchmark，一个是 sciplex3/TCDD 这种药物、剂量、细胞系数据，一个是 Norman 的组合基因扰动，还有 Gasperini 和 Papalexi 这种调控/多模态方向。第一批 smoke 已经跑出来了，sciplex3 和 OpenProblems 的风险排序信号比较明显；TCDD 剂量这条线反而提醒我，剂量不能当普通类别，要单独设计 dose-aware 版本。接下来我会把 sciplex3、OpenProblems 和 Norman 先正式化，Tahoe raw 继续作为大规模化学数据和 batch/cell-line/MoA 分层的支撑。”
