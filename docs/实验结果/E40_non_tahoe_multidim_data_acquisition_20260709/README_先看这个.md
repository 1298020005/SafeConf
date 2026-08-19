# E40 非 Tahoe 多维数据补强记录

- 更新时间：2026-07-10T18:20:00
- 目的：把老师要求的“更多数据、多维度数据”先落到可追踪的数据账本里。
- 这份记录不替代实验结果，它回答：现在手里有哪些数据，哪些已经完整，哪些正在下载，下一步能设计哪些实验。

## 1. 当前数据层次

- 基因单扰动：45 个 h5ad，33 个 study family，约 3,145,339 个观测；代表数据：PapalexiSatija2021, LaraAstiasoHuntly2023, AdamsonWeissman2016, DixitRegev2016, Replogle, TianKampmann2021, FrangiehIzar2021, TianKampmann2019
- 基因组合扰动：18 个 h5ad，11 个 study family，约 4,352,022 个观测；代表数据：Replogle, ReplogleWeissman2022, NadigOConner2024, SchiebingerLander2019, SchraivogelSteinmetz2020, Schmidt, Norman, NormanWeissman2019
- 药物/化学单扰动：8 个 h5ad，7 个 study family，约 555,752 个观测；代表数据：SrivatsanTrapnell2020, sciplex3, TCDD, McFarland 等。注意：旧自动表里个别文件被粗略归到 chemical，后续以人工字段识别为准。
- 药物/化学组合扰动：6 个 h5ad，3 个 study family，约 1,409,693 个观测；代表数据：sciplex3, SrivatsanTrapnell2020, LotfollahiTheis2023 等。
- 增强子/调控元件扰动：6 个 h5ad，3 个 study family，约 1,623,311 个观测；代表数据：GasperiniShendure2019, JoungZhang2023, XieHon2017

## 2. 新补的外部数据

- Tahoe-100M pseudobulk differential expression：1026/1026 parquet present，已落盘约 88.86 GB；用途：已用于 Tahoe chemical D1-D5；可继续做细胞系/药物/分片稳定性。
- Tahoe-100M raw single-cell shards：继续下载中，当前本地约 93 GB，仍有 `.aria2` 残片；raw schema 已确认，包含 `genes / expressions / drug / cell_line_id / moa-fine / canonical_smiles / pubchem_cid / plate`。
- OpenProblems / NeurIPS 2023 single-cell perturbations：processed/workflow DGE 已能用于实验；剩余 raw、pseudobulk、multiome、MoA annotations 部分文件因 `openproblems-bio.s3.amazonaws.com` TLS EOF 暂未完整。当前 E41/E49/E55 不依赖这些残片。

## 3. 对后续实验的直接意义

- 基因单扰动：继续做 GEARS / scGPT / 简单基线的普通基因敲除风险评估。
- 基因组合扰动：检查 SafeConf 面对组合扰动时，support、context、model disagreement 是否还能解释失败。
- 化学扰动：Tahoe 和 OpenProblems 形成两个独立来源，可测试 chemical setting 是否只是 Tahoe 特例。
- 增强子/调控元件扰动：用于补一个 regulatory/enhancer 方向，不让论文只停在 gene/drug 两类。
- 多供体/多细胞类型：OpenProblems 可以直接设计 donor holdout、cell-type holdout、compound holdout。
- 原始单细胞层：Tahoe raw 与 OpenProblems raw 后续能做更细的细胞状态分层，不只依赖 pseudobulk。

## 4. 文件说明

- `tables/non_tahoe_local_inventory.csv`：本地 scPerturb + scPerturBench 的 study-level 总表。
- `tables/source_coverage_by_dimension.csv`：按扰动类型汇总的覆盖表。
- `tables/external_acquisition_status.csv`：Tahoe raw、Tahoe pseudobulk、OpenProblems 的下载状态。
- `RUN_STATUS.json`：机器可读状态记录。
- `人工识别记录_哪些数据真的能用.md`：逐个看字段后的人工判断，说明哪些数据能直接变成实验。
- `数据线总览_尽可能多但不乱.md`：按老师问题拆出的完整数据线，覆盖 chemical、cell context、combination、dose、regulatory、multimodal、immune、tumor、neuron、Tahoe raw、OpenProblems。

## 5. 后续触发顺序

1. 先跑 OpenProblems workflow Kaggle DGE smoke，因为 train/test/id_map/prediction/score 已完整。
2. 同时跑 sciplex3、TCDD、KaggleCrossPatient、crossSpecies、Norman、Gasperini 这些本地已完整数据。
3. Tahoe raw 先做 metadata/shard-level 字段审计，不等全量下载。
4. OpenProblems raw / multiome 完整后，再做 compound holdout、MoA holdout、cell-type holdout、donor holdout、multiome context。

## 6. 已经触发的第一批结果

- `../E41_multidim_first_batch_smoke_20260710/`：OpenProblems DGE error-vs-risk smoke，以及 Tahoe raw shard 字段审计。
- `../E42_E48_local_first_batch_smoke_20260710/`：sciplex3、TCDD、KaggleCrossPatient、crossSpecies、Norman、Gasperini、Papalexi 的本地第一批 smoke / 字段审计。
- `../E41_E48_first_batch_decision_note_20260710.md`：把第一批结果整理成“哪些能正式化，哪些先留作边界”的决策记录。
- `../E49_E52_formalization_batch_20260710/`：OpenProblems、sciplex3、Norman、TCDD 的第二阶段正式化 split。
- `../E53_tahoe_raw_expanded_audit_20260710/`：Tahoe raw 128 个 shard / 361 万行的扩展字段审计。
- `../E54_sciplex3_gene_sensitivity_20260710/`：sciplex3 1000/3000/5000 基因数敏感性。
- `../E49_E53_second_stage_decision_note_20260710.md`：第二阶段结果判断和下一步实验顺序。
- `../E55_cross_dataset_transfer_20260710/`：老师要求的“一个数据集见过，到另一个数据集预测”跨数据集 transfer 审计。
- `../E56_cross_dataset_source_size_ablation_20260710/`：跨数据集 source-size ablation，检查源矩阵只给 25%/50%/75%/100% 时风险排序变化。
- `../E55_teacher_requirement_action_note_20260710.md`：周老师要求逐条核对、下载状态、E55/E56 结论和下一步安排。
- `../E57_dataset_expansion_cross_dataset_20260710/`：新增 Lara、Dixit、Tian、Replogle、Adamson、SciPlex2/4 的跨数据集扩容审计；26/26 个方向成功打分。
