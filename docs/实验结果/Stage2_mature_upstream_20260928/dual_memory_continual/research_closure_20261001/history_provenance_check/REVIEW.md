# HistoryEligibility 与原始 study provenance：只读核查

结论：现有 TxPert 历史符合**每个 query 上游 source train 可见、目标 context treated 完整排除**的既定 E201 合同；它不符合把“同 study 的合法 outer-train history”解释为**风险模型 gene outer-train 才可供历史**的严格字面合同。没有发现 gene/task 跨上游模型进入不同风险 fold。另有实际原始 study provenance 被 umbrella ID 掩盖，顶级审计将全部历史归为同 study 的断言需要更正。

本次只读 Python/Markdown 合同、manifest、split 清单、parquet metadata 和 H5AD 的 `obs`/类别 schema；没有读取 `X`、layers、表达矩阵、effect/control/test vectors，没有训练或运行 TabPFN，没有修改冻结方法和原始资料。

## 1. 两个训练分区必须区分

| 层次 | 冻结单位与可见资料 | 实际实现 | 核查结果 |
|---|---|---|---|
| 每个 query 的上游 TxPert | 三个非 target context 的 source train perturbations；当前 target treated 被物理删除 | `build_e201_txpert_blind_training_view.py:70` 允许 train/val conditions，`:80` 排除 target context；官方 data module `:364` 联合 condition 与 source cellline 判 train | 本次主 cohort 的全部 4,737 历史 edges 实际都是 train condition（val 0），source context 均非 query context |
| 风险模型 outer folds | 按 gene 分组；同 gene 的全 context 与所有 upstream predictions 保持同 fold | `run_dual_memory_txpert_public_biology.py:161`；冻结 `TX_TASK_SPLIT.csv`；source cache metadata | 1,808 tasks、575 gene；source cache 两上游 3,616 行，所有 gene/task 均只有一个 fold，完全通过 |
| Public transfer 学习与 error-rank labels | 监督 labels 只用 outer-train；训练特征还做 inner OOF | `run_safeconf_research_closure.py:82`、`:125`、`:132` | 代码显式检查 fit/query gene 分离；没有将 held-out source error/transfer labels用于 learner fit |
| Public Memory retrieval | 每个 query 使用同 gene、其它 context 的已观测 source effects；无需记忆项属于 risk outer-train | `build_dual_memory_txpert_public_bank.py:232` 生成 query-specific eligibility；`run_dual_memory_txpert_public_biology.py:185` 只按 target/condition lookup | 这是已观测同扰动跨背景历史辅助风险，不是将整个 gene 的生物资料都封存的任务 |

原条件 split **不是 condition-only 互斥分区**：train/test 有 580 个 condition 交集，val/test 有 202 个交集。主 cohort 历史的 condition 全部也在 test 条件清单；官方 `gspp/data/datamodule.py:364–368` 使用 source context 判 train，`:422` 起使用 target context 判 test。因此 condition 落在 test list 本身不构成来源泄漏。见 `TXPERT_UPSTREAM_HISTORY_ROLE_SUMMARY.csv`。

主 cohort 的 4,737 history edges 中，4,320 条的 source 物理实验也在该主风险任务 cohort，全部与 query 位于**同一个** outer-test gene fold；417 条来源不是 ≥30-cell 主风险任务。other-risk-fold edges 为 0，同 context edges 为 0，扰动 gene 不匹配 edges 为 0。逐 fold 见 `TXPERT_RISK_FOLD_HISTORY_OVERLAP.csv`。

这解释了表面矛盾：同一个物理 source 实验可以是 A 上游预测的已知训练输入，同时是 B 上游预测任务的目标真值。资格必须相对于具体 query 上游判断。全 gene 同 fold仍可保护风险监督 labels；它不会把这些预先可见 source 生物资料变成“风险 outer-train 资料”。本次没有将其误记为当前 query 的 target truth 输入。

**具体不一致与最小修复**：`build_safeconf_governance_artifacts.py:309` 输出 `eligible_outer_train_only`，而 `docs/方法设计/20260928_证据驱动开题/开题报告与进展.md:149` 写同 study 只能合法 outer-train history。应把审计分区明确命名为 `eligible_query_upstream_source_train`，另列 `risk_label_partition=outer_train`，并注明检索允许同 gene 的其它已观察 source context。该项是合同/审计澄清；若坚持风险 gene outer-train 才可供历史，则现有同 gene history 会被清空，构成新信息条件和新实验，不能用文字修正宣称现有分数已经符合它。

## 2. 原始 study 不等于包装 study ID

pinned TxPert commit `08d82eea86746b044cf7531f4ec8c5f60e1cb73f` 的 `README.md:66–70` 明确来源：K562/RPE1 为 Replogle 2022，HepG2/Jurkat 为 Nadig 2025，K562_adamson 为 Adamson 2016。原缓存 obs 有五个 cellline 类别，E201 的 `PUBLIC_CELL_TYPES` 明确排除 K562_adamson，四个 blind manifests 均显示 target treated=0。

| context | 原始研究 | canonical memory items |
|---|---|---:|
| K562 | Replogle 2022 | 580 |
| RPE1 | Replogle 2022 | 467 |
| hepg2 | Nadig 2025 | 480 |
| jurkat | Nadig 2025 | 481 |

`build_dual_memory_txpert_public_bank.py:129` 将全部 2,008 个实验写成 `study_id=E201_TxPert_official_multicontext`；这标识官方合并缓存/项目数据集，不是原始研究 provenance。builder 的 5,238 条全 cohort eligibility edges 经 `:213–238` 去重为 2,008 physical context×condition items，没有把重复 blind view 计为新实验，此项正确。

每个 query 可供其它三个 context 的来源，至多含一个同原始研究 context 和两个不同原始研究 context；不是每个 query 都拥有全部三项。实际 ≥30-cell 主 cohort 有 **1,585 same-original-study edges 和 3,152 cross-original-study edges**，见 `TXPERT_STUDY_EDGE_SUMMARY.csv`。四个 context不能自动当四项独立 study；统一包装 ID 同样不能自动把两项原始研究当成一项。它们是两篇来源研究的屏幕、共享系列实验和合并预处理，不能计作两个新的独立盲确认。

**具体不一致与最小修复**：`build_safeconf_governance_artifacts.py:305–316` 对全部 history 标 `H_internal`、`same_study_as_target=true`、`independent_public_source=false`，并生成“no verified external”。应新增一个带官方 commit citation 的原始 study sidecar，同时保留 `benchmark_dataset_id=E201` 和现有 experiment IDs。审计按 query/source 原始 study 匹配分别计数，区分“同 benchmark cache 内部”与“原始跨 study 来源”。本次未修改 bank、gene folds或 learner 输入；此映射是 provenance 修复，不应把 original study重新加入数字风险特征或拆散既有全 gene cluster。

官方来源：[TxPert 原始仓库](https://github.com/valence-labs/TxPert/tree/08d82eea86746b044cf7531f4ec8c5f60e1cb73f)、[当前官方说明](https://github.com/valence-labs/TxPert)。本次同时核验了本地 pinned 文本与在线官方说明，来源一致。

## 3. McFaline train/val 历史的实验单位与限制

McFaline memory 有 7,173 items，统一原始来源 `McFalineFigueroa23_PerturBench`；provenance suffix 为 train 6,631 items、val 542 items。builder `build_mcfaline_quality_memory.py:160` 定义 unit 为 **gene×cell_type×drug treatment**，在该 unit 内按 gRNA、PCR_plate、随机 cell half 计算独立分组间 effect 一致性。public effect 为各 guide effect 的等权均值（`:301`），control为 train/val 的 context×treatment control。memory item 已把多 guides/plates聚合为一个实验单位，不包含 replicate group 明细 ID。

`seal_mcfaline_dual_memory_risk.py:158–167` 对每个 query先取相同 gene，再排除 exact context×treatment item；因此它具备 task-level 历史排除，而不是 guide-disjoint 或 PCR-plate-disjoint 来源筛选。metadata审计：train/val与test共享 2,148 个 gRNA IDs、4 个 PCR plates、12 个 cell_type×PCR_plate units。完全一致的 gene×context×treatment 仅共享 3 个 control单位；treated task combinations仍为官方 covariate holdout。见 `MCFALINE_METADATA_UNIT_OVERLAP.csv`。

PCR plate 是否属于独立生物重复、同 guide 的其它药物/细胞背景是否应被视为同 experimental-unit replicate group，现有聚合 schema没有足够身份字段证明。现有结果只能声称该 task/context/treatment 排除与训练/验证历史一致性；不能从多个PCR plate或guide计数推出独立 study，也不能将这些一致性评分称作另做了一次 guide/plate-disjoint held-out模型评价。最小修复是写明当前 unit 与过滤规则；若严格replicate-group合同有额外禁止范围，须先注册 compound group ID，再单独审计资格，不可默认为满足。

**实际访问字段的代码矛盾**：512-gene builder `build_mcfaline_quality_memory.py:191–203` 和 common-gene builder `build_safeconf_common_gene_biology.py:195–207` 都按整个数据行范围构造 CSR块，从 `X/data` 读取后才在 `accumulate` 按train/val codes过滤。根据代码路径，它们会物理读取含 test 行的表达块，而不是只读取 train/val行。`n_test_cells_aggregated=0` 可由 mask逻辑支持，但 source manifest `test_expression_opened=false`（512 builder `:318`、common builder `:322`）与物理访问语义不一致。本次只读代码发现此项，没有读取任何表达。

最小修复：现有访问账本将“test表达物理读取”和“test表达用于聚合/拟合”分别记录，保留test聚合为0的事实；未来builder应在碰`X/data`前只生成 dev contiguous row spans并读取这些span，不能在含test的整个CSR块读完后才mask。无须为审计重跑历史实验，也不能事后把已发生访问改写成没有发生。

## 4. 预处理合同的附加范围限制

官方 pinned README `:72–76` 表明 cross-cell cache 的 gene axis包含 test cellline 的HVG选择；blind builder复制原 axis，public-memory builder `:86–89` 只检查3352个gene。因而该固定官方3352轴、及其common2840交集，不能单靠“blind target treatments=0”宣称原始gene选择也是risk outer-train拟合、完全无target-derived preprocessing。`MEMORY_ALIGNMENT_CONTRACT.md:17` 的严格外折HVG/PCA规则和`:18`禁止target-derived preprocessing需与继承官方axis这一现实分别说明。当前risk数值标准化/CDF训练隔离并未因此失败；但其上游官方预处理有已继承限制。最小修复是登记此限制、收窄资格证明范围，而不是更换冻结轴或在本次审计重做预处理。

## 交付证据

- `CONCRETE_FINDINGS.csv`：五个具体不一致/范围限制、代码行号和最小修复。
- `TXPERT_CONTEXT_STUDY_PROVENANCE.csv`、`TXPERT_STUDY_EDGE_SUMMARY.csv`：原始study映射与query条件下的实际edge计数。
- `TXPERT_RISK_FOLD_HISTORY_OVERLAP.csv`、`TXPERT_UPSTREAM_HISTORY_ROLE_SUMMARY.csv`：风险fold与上游训练分区分开核查。
- `MCFALINE_METADATA_UNIT_OVERLAP.csv`：只读metadata的guide/PCRplate/task重合计数。

范围结束：只读审计及新审计目录中的证据文件；没有改变既有 frozen prediction、truth、risk scores、算法、gene axis或论文正文。
