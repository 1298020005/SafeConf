# Target-gene control expression：元数据可行性审计

**结论：Orion 可用已有完整 38606-gene TRAIN NTC sidecar 为固定队列构造零新增调用的 control-only 代理。当前 Source／McFaline readout 不能完整覆盖，故停止其全队列版本；不补零、不更换轴、不抽取新表达。尚未生成任何分数。**

本次只读 manifest、gene-axis、task identities、方法 ID／代码和 `.npy` header；Orion TSV 只选择 `gene_id` 列。未读取控制数值、预测／truth 数组或错误数值，未采样、拟合、调用模型或改变方法。覆盖使用现有精确 gene 标识，不增加 HGNC alias／任意 symbol mapping。

## 固定队列的坐标覆盖

| 固定集合 | 原 tasks／gene IDs | target 在当前 readout：tasks／gene IDs | 已有完整控制输入可评分：tasks／gene IDs |
|---|---:|---:|---:|
| Source 两个方向共同生物任务，2840 axis | 1808／575 | 654／209 | native3352 也仅 724／232 |
| Source Orion-adapted 3285 axis，同一任务集合 | 1808／575 | 705／227 | 同上 native3352，不覆盖全集 |
| McFaline common2840 | 543／380 | 137／99 | 当前 cached control 为 2840 列，无已定位的完整 target 覆盖向量 |
| Orion primary common3285 | 232／144 | 87／55 | full38606 TRAIN control：232／144 |
| Orion all prediction-only QC 集合 | 2993／1750 | 466／278 | full38606 TRAIN control：2993／1750 |
| Orion no-Source-history QC 集合 | 2761／1606 | 379／223 | full38606 TRAIN control：2761／1606 |
| Orion 已用 competence／validation-reuse pool | 2790／1661 | 444／263 | full38606 TRAIN control：2790／1661 |

这些是 metadata 坐标可评分数，未重新判 QC／历史资格，也未检查数值大小或性能。Source 1808 个任务不能因两个架构重复计为 3616 个独立任务；MC 的 380 是冻结 target-name strings，其中 `RcontrolSEL::t98g::lapatinib` 是现存无 gene 坐标的特殊条件，不能人为当作表达为零。

context 覆盖（当前 readout／原 tasks）：Source2840 为 K562 `207/566`、RPE1 `151/416`、hepg2 `145/405`、jurkat `151/421`；MC 为 a172 `47/189`、t98g `43/177`、u87mg `47/177`。Orion primary endpoint 为 HCT116 `37/107`、HEK293T `50/125`，而 full TRAIN control 分别完整覆盖 `107/107`、`125/125`。Orion all QC endpoint 为 `214/1420`、`252/1573`；full control 全覆盖。这里未按 outcome 挑子集。

## 控制身份、维度和效应定义

**Source。** 原文件：

- `/home/yyf/data/txpert_official_20260802/e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy`
- `/home/yyf/data/txpert_official_20260802/e205/pretruth_vectors/E205_CONTROL_CENTROIDS.npy`

两者 header 均为 `2008×3352 float32`；既有 `orion_source_core_20261002_v1/SOURCE_INPUT_HASHES.json` 给两者相同 SHA `c2d6f56bec576b5154d574d91f37b5fdbb036d869573cdbaf9fda0e544454d3e`。本次未读取数组 body 或重算其 SHA。Source tasks 以 `source_mean_delta_row` 定位，1808 个唯一行，范围 0–2007；与 common2840 的 `SOURCE_TASKS.csv` 行序精确相同。原 true effect、两种预测 effect 都减去这套冻结 matched-batch control；native 坐标为全库 CP4000 后 log1p 再投影，列投影不再归一化。

axis：`/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201/gene_ids.json` 的3352 symbols；common axis：`/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/GENE_IDS.json` 的2840 symbols。Source3285 可直接复用 `/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_source_core_20261002_v1/SOURCE_CONTROLS.npy`（header `1808×3285 float64`）及同目录 `SOURCE_TASKS.parquet`、`GENE_MANIFEST.csv`；两种 `SOURCE_TxPert_*_CONTROLS.npy` header 同形状。其 manifest 合同是 `Source3285_CP4000_log1p_matched_batch_delta_v1`。

**McFaline。** `/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/TEST_CONTROLS.npy` header 为 `543×2840 float32`；`VALIDATION_CONTROLS.npy` 为 `542×2840 float32`。测试行轴来自同目录 `TEST_TASKS.csv` 的 contiguous `condition_index`、`task_id`。原 control-only helper 按 split=test、control=1、精确 `(cell_type,treatment)` 聚合 cell 均值；common closure 按相同 `(context,treatment)` 写 row，与 truth effect／预测 effect 所减控制一致。三个冻结测试状态是 a172/nintedanib、t98g/lapatinib、u87mg/none，不能只按 context 忽略 treatment。

代码直接使用官方 processed X 的原 log1p 坐标，不进行后续库量归一化；原 H5 metadata 为 `878229×15009`，uns.log1p 存在但无 scale 参数。它不足以单独证明 CP4000／CP10000 或逐细胞分母；本次不重建这些分母。该限制不妨碍同一 MC outcome 合同内的 scalar ranking 比较，但禁止把其控制表达原值同 Source／Orion 当成共同物理单位或 pooled threshold。

**Orion。** `/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_lm_bridge_20261002_v1/actual_authorized_v3_decimal_tokens/{HCT116,HEK293T}/OWN_CONTEXT_NTC_MEAN.tsv` 均有38606个唯一Ensembl IDs，schema 为 `gene_id,mean_cell_logCP4000`。本次只读 ID 列。两份 sidecar 已被原 expanded risk seal 绑定，SHA 分别 `a9d4576b3de1fc0ac435f531a37b149b7adb9a48dd6979b40c6bc5c942aaaf99`、`31772ae9aa9b4f5eac0b057e7c5373c7698b221de330060439f3a11c030ffb1e`。

同 bridge 的 `BRIDGE_MANIFEST.json` 标记全38606／endpoint3285、science6b8、真实 caa producer＋a7d3 technical amendment；选择每个 context 自己的 TRAIN NTC，并声明 VAL／TEST numeric rows read=0。原数学 sealer 用 Ensembl lookup 投影该 full control 到 `orion_published_lm_20261002_v3_decimal_tokens/{context}/OUTPUT_GENE_AXIS.tsv` 的3285轴，`PREDICTIONS_DELTA`／后来的固定 truth effect 均使用 own TRAIN NTC、mean-cell-log1p-CP4000 坐标。完整控制中的 target 标量因此不需 Public treated effect、错误标签或新的 biological read。

这里的“matched”仅指 own context 与冻结 effect 基准完全一致；Orion sidecar 是 TRAIN NTC 的 context 均值，不是每个 query 的 plate／batch-specific control。不能将三种资产的控制精度或配对层级写成相同。

## 已有强规则与同信息比较边界

已核对 common MATRIX、SUPPORT_CONTROL、state-reference 方法 ID 及其实现：存在 Magnitude、Direct／WeightedHistoryDistance、NegativeHistorySupport／EffectiveSources、StateMean／MatchedKStateMean 和 control-RMS amplitude diagnostic。P6 是预测向量摘要，PUBLIC7 是历史参照摘要；pair control RMSE／cosine 使用全向量。**在这些实际方法和原 Orion13候选中，未发现“query target-gene 的 matched control expression”单标量等价规则。**此为有限当前资产查重，不声称整个仓库从未出现过类似规则。

若另行固定执行，合理对象是 control-only 输入代理与原冻结规则在**完全相同任务、context、error estimand**上的配对比较。Orion 可保留完整232主队列，0额外upstream／Public调用、0新label／expression acquisition；然而 full38606 scalar 使用了现有 P/PUBLIC readout 之外的输入坐标，必须明确登记，不能称完全相同13-feature信息预算。高表达／低表达哪一方向代表更高风险须在读性能前固定，不得事后择符号、caliper或亚群。

Source／MC 当前完整队列不可闭合。其 metadata-supported 子集可另行事先登记并对所有规则使用同一身份交集，但它只回答该子集，不能替代原1808／543全队列主比较。当前审计不发起这一子集实验，不读取更宽 raw controls、不补零、不重投影风险模型，不重做 SourceMean／scale 或上游候选。

## 元数据绑定与停止点

- common2840 `GENE_IDS.json`：`be612f9c4638146b41eee26f607ab12fbed53b524cda7a76eee36ca554c28f97`。
- common `SOURCE_TASKS.csv`：`72c6620d797db9115e67f6c9067d54879950733827801190ea25161ee03a118d`；`TEST_TASKS.csv`：`0b2f5be1b4e8911157cf317d7f950185be3f88235bd9350221ce93f52dc9982e`。
- Source `SOURCE_INPUT_HASHES.json`：`12e058d3725fa989b3da543c136c169efd90dbcba28a8a8c2cce4c4033769715`；3285 `GENE_MANIFEST.csv`：`c1c9c8041458275248514d960c6c263a72f43a12a8d48c16f77759d2342fa08f`。
- 实际 bridge manifest：`13c90f2feb21f841138f366fd140ff6a194bdd20d1337bd3c8d7e15500d0d478`。

本阶段在身份／覆盖可行性处结束。不存在新分数、方向选择、数值复算、bootstrap或模型更新，也不产生新确认资格。
