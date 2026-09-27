# UNIFIED_PERTURBATION_TASK_SPEC

**版本**：v0.1（2026-09-27）
**用途**：为 SafeConf 建立同时覆盖基因扰动、化学药物扰动及后续其他扰动的任务接口，并设计只依赖现有数据的接口验证实验。

## 0. 本轮边界

本轮解决的是“SafeConf 看见一条预测时，最少需要知道什么，以及这套表示能否覆盖现有数据”。本轮不训练新的风险网络、不追加 E274、不下载数据、不打开 E208/Tahoe 的封存测试真值。验证对象是数据契约、适配器和来源追踪。

已有元数据盘点结果：仓库和服务器 manifest 中的 83 个本地数据记录全部以 `backed='r'` 成功读取，未载入表达矩阵，`read_errors=0`、`missing=0`。原始清单见 [`docs/数据审计/20260927_perturbation_inventory/PERTURBATION_DATA_INVENTORY.csv`](docs/数据审计/20260927_perturbation_inventory/PERTURBATION_DATA_INVENTORY.csv)。

## 1. 设计结论：分成任务层和预测层

候选结构中的公共字段是正确起点，但 `upstream_model_id` 不应成为任务身份的一部分。一个任务可以被 GEARS、scGPT、CPA 或另一个模型分别预测；它们应共享同一个 `task_id`，并各自产生一条 `PredictionRecord`。因此接口分两层：

1. **UnifiedTaskRecord**：描述真实的扰动任务和参考状态，与任何预测器无关。
2. **PredictionRecord**：描述某个冻结上游模型对该任务的输出、输出空间、版本和可审计来源。

这样才能在同一任务上比较多个预测器，也才能把“当前预测”“公共历史”和“错误记忆”严格绑定到同一任务身份和同一预测器版本。

## 2. 统一任务接口

### 2.1 硬性字段（所有进入 SafeConf 的任务都必须有）

| 字段 | 类型 | 规则 |
|---|---|---|
| `task_id` | string | 由规范化后的数据集、背景、扰动对象和条件生成；不含预测器；稳定、可复现。 |
| `dataset_id` | string | 原始数据集和版本标识，不能只写论文简称。 |
| `context.context_id` | string | 细胞系、细胞类型、供体或等价生物背景的稳定 ID。单背景数据也必须填写。 |
| `perturbation_type` | enum | 见 2.2；未知类型不得默认为基因或药物。 |
| `perturbation_target` | object/list | 至少有规范化对象 ID；组合扰动按排序后的成员列表保存。 |
| `perturbation_condition` | object | 条件字段必须有值状态：`known`、`unknown` 或 `not_applicable`，禁止把缺失默认为零。 |
| `control_reference` | object | 明确 untreated、non-targeting、vehicle、matched control 等参考状态及其来源。 |
| `output_contract` | object | 固定输出空间、基因轴 hash、归一化/聚合方式和数值单位。 |

任务层的统一最小语义是：**在什么背景中，以什么方式作用于什么对象，在什么条件下，相对于哪个参考状态产生什么响应**。

### 2.2 `perturbation_type` 取值

建议使用可扩展枚举：

- `genetic_knockout`
- `genetic_repression`（包括 CRISPRi）
- `genetic_activation`（包括 CRISPRa）
- `genetic_overexpression`
- `genetic_combinatorial`
- `chemical_small_molecule`
- `chemical_combinatorial`
- `regulatory_element`
- `extrinsic_stimulus`
- `infection_or_environmental`
- `unknown`（只能进入接口审计，不能进入类型特异学习）

`genetic_single`、`chemical_single` 等 manifest 标签是数据盘点标签，进入规范记录时必须细化到上述枚举，无法细化时保留 `unknown` 并记录 `subtype_confidence=low`。

### 2.3 背景、对象和条件对象

```yaml
context:
  context_id: K562
  context_type: cell_line       # cell_line/cell_type/donor/organism/other
  organism: Homo_sapiens
  cell_line: K562
  cell_type: null
  donor_id: null
  batch_id: optional

perturbation_target:
  - target_namespace: HGNC
    target_id: TP53
    canonical_name: TP53
    role: primary

perturbation_condition:
  mechanism: CRISPRi
  guide_id: optional
  guide_sequence: optional
  dose: {value: null, unit: null, status: not_applicable}
  duration: {value: 24, unit: h, status: known}
  vehicle: {value: null, status: not_applicable}
  multiplicity_or_intensity: {value: null, unit: null, status: unknown}
```

`condition` 是类型化对象，不是一个拼接字符串。剂量、处理时间、guide、MOI、批次等字段可以缺失，但必须显式说明缺失状态和缺失原因（原始数据没有、该扰动不适用、或尚未完成映射）。

### 2.4 对照和响应

`control_reference` 至少包含：

- `kind`：`untreated`、`non_targeting`、`vehicle`、`matched_control` 或 `other`；
- `definition`：原始数据中的字段、取值和筛选规则；
- `reference_task_id` 或 `reference_vector_ref`：如果对照是一个可复现任务或已聚合向量；
- `matched_on`：背景、批次、时间、板号等匹配条件。

SafeConf 的统一响应优先使用相对对照的 `predicted_effect`，即预测状态和匹配对照状态在固定输出空间中的差。若上游只给出 post-expression，适配器负责在同一背景、同一条件和同一输出轴上减去对照，并记录转换过程；若上游已经给 effect，则不重复相减。

## 3. 预测层接口

```yaml
prediction:
  upstream_model_id: GEARS
  model_version: checkpoint_or_commit_hash
  adapter_id: gears_npz_v1
  adapter_version: 1.0
  prediction_ref: path/to/prediction.npz#key
  output_kind: predicted_effect  # predicted_effect/post_expression
  output_space:
    axis_id: human_gene_15473
    axis_hash: sha256:...
    normalization: log1p_or_dataset_defined
    aggregation: perturbation_mean
    value_unit: expression_effect
  split:
    role: train/validation/test_sealed
    fold_id: fold0
    split_manifest_hash: sha256:...
  provenance:
    train_data_hash: optional
    target_truth_accessed: false
    generated_at: timestamp
```

`predicted_response_ref` 必须指向向量或可复现数组；只有 RMSE、相关系数、均值等指标的表不能直接进入 SafeConf。`target_truth_accessed=false` 在预测冻结前必须可审计，尤其适用于 E208 和 Tahoe。

## 4. 哪些字段共用，哪些字段类型特有

### 4.1 可完全共用的输入和学习逻辑

基因和化学扰动都可共用：

1. `context_id` 及背景层级；
2. `perturbation_type`、对象 ID 和条件对象的状态编码；
3. 匹配对照定义；
4. 固定的 `predicted_effect` 向量及其输出轴；
5. 上游模型 ID、版本、适配器版本和预测来源；
6. 预测向量的幅度、稀疏性、非零比例、基因集/通路聚合、预测不确定性或多模型分歧等派生特征；
7. 按任务划分的 train/validation/test、OOF 错误来源和历史可用性标记；
8. SafeConf 的风险目标：在给定预测和可用历史下估计误差或覆盖风险。

共同学习逻辑应作用于“预测相对参考状态的响应是否可靠”，而不是作用于“这个对象的名字像不像药物”。

### 4.2 必须作为可选的类型特有字段

| 类型 | 类型特有字段 | 缺失时的处理 |
|---|---|---|
| 基因 | intervention mechanism、KO/CRISPRi/CRISPRa/OE、guide ID/序列、编辑效率、MOI、靶基因 namespace、组合成员和 guide 关系 | 缺失机制可进入类型无关接口，但不能声称为某种机制；guide 特征分支应关闭。 |
| 化学 | compound ID、SMILES/结构、药物名、dose value/unit、处理时间、vehicle、MoA、板/批次、组合药物成员 | 缺失剂量或时间可进入类型无关接口，但不能进入 dose-response 或时间条件分析。 |
| 调控/刺激/感染 | enhancer/TF ID、刺激物或病原体、强度、感染复数、暴露时间、物种 | 只能在公共字段完整时进入统一接口；特有分析缺字段则 fail closed。 |

因此推荐**共享主干 + 扰动类型 embedding + masked 类型特有字段 + 少量类型特有校准器**。当前没有证据支持一个完全扁平、所有类型共用同一组参数的模型；也没有理由为基因和药物各自复制整套 SafeConf。

## 5. 缺失字段与准入规则

### 5.1 硬失败

以下缺失时记录不能进入正式 SafeConf 评分：

- 无法确定背景 `context_id`；
- 无法确定扰动对象，且不是明确登记的环境/刺激任务；
- 无法确定扰动类型；
- 没有可复现的对照定义；
- 没有固定输出轴或无法定位预测向量；
- `PredictionRecord` 没有模型 ID、版本、适配器和 split/provenance；
- 只有汇总指标，没有逐任务预测向量；
- 测试真值已被提前用于预测、校准或历史构造。

### 5.2 软缺失

dose、time、guide、SMILES、MoA、batch、replicate 等可以为空，但必须编码为 `unknown` 或 `not_applicable`。允许公共 SafeConf 分支运行；依赖该字段的专用分支必须关闭并返回明确原因。禁止用零剂量、默认 24h 或“unknown gene mechanism”替代真实缺失。

## 6. 现有数据的现实映射

| 数据线 | 类型和规模（当前盘点） | 可映射字段 | 预测/输出情况 | split 与用途判断 |
|---|---|---|---|---|
| E131/LOPO 基因线（Frangieh、Lara、Cui、Tian 等） | 基因单扰动；例如 Frangieh 3 背景×212 对象=636 任务，Lara 有 8/19/6 背景线 | 背景、基因对象、对照和任务级 effect 可从既有协议映射；机制/guide 部分数据缺失 | 既有 V0StrongBaseline、ContextSim、GEARS/scGPT 任务产物；需用 adapter 统一向量指针 | 已有 LOPO fold，适合开发和适配器确认；必须复核 OOF 来源 |
| E208 Jiang24 | CRISPR 扰动；K562/MCF7/HT29/HAP1 与 IFNG/INS/TGFB 状态，224 个测试任务，15,473 基因轴 | `cell_type`→context，`condition`→gene target，`treatment`→状态，`control`→参考；机制按实验协议登记 | 有冻结的预测任务协议；测试真值封存，不能在本轮打开 | 适合外部确认；只能在 prediction/risk freeze 后评估 |
| E258 Feng | CRISPRi、四 train/两 validation/四 test donors | donor/context、gene target、CRISPRi 可作为类型特有字段 | 预测接口可接入，最终 test truth 当前不开放 | 适合 donor-held-out 确认 |
| E84/E87/E89/E118 CPA 化学线 | sciPlex3、OpenProblems、sciPlex4；单药及组合药物，含 dose/time 的数据子集 | context、compound、dose、time、vehicle、组合成员可映射 | CPA-RDKit/CPA-Ridge 的 `predicted_effect_key` 和向量可适配；E118 报告显示严格契约通过，但化学独立增量门未通过 | 适合化学开发/跨数据集确认；不能把当前失败隐藏为成功 |
| Tahoe E260/E263 | 24 cell lines×100 drug-dose；1200 train、600 validation、600 test_sealed；E263 为处理和匹配对照元数据/表达 | `cell_line`、drug、dose、unit、plate、matched control 可映射 | 当前有任务和真值封存面，但本轮没有新的上游预测产物，故只能进入 TaskRecord | 适合作为化学外部确认任务面；不在本轮追加模型 |
| Norman/Wessels 等 | 基因组合扰动；Wessels 当前盘点含 187 个组合任务 | 组合成员列表、背景、对照可映射；guide/机制需按原始协议核对 | 可作组合类型适配器测试；不作为当前主 headline | 适合接口扩展和组合任务压力测试 |
| TianActivation/TianInhibition、调控/刺激数据 | CRISPRa、CRISPRi、enhancer/刺激等 | 可检验机制枚举和非药物特有字段 | 大多只有原始任务元数据，预测层需后续接入 | 先做接口覆盖，不用于本轮模型结论 |

完整的逐文件结果（含 obs 列、剂量/时间/药物字段、上下文和任务计数）在 `PERTURBATION_DATA_INVENTORY.csv`；不能把 manifest 中的 `genetic_single` 自动解释成 KO，也不能把有 `SMILES` 的记录未经协议核对就标成药物预测线。

## 7. 真实统一记录示例

下表是接口实例，不代表本轮重新生成预测。

| 来源 | 统一任务记录 | 预测层/限制 |
|---|---|---|
| Frangieh LOPO | `context=Co-culture; type=genetic_single→机制待核对; target=ACSL3; control=协议定义的非靶向/匹配对照; task_id=Frangieh::task_00001` | `V0StrongBaseline` 的 `run_dir`/向量需由 adapter 解析；若只有指标则拒绝。 |
| E83 CPA 示例 | `context=A549; type=chemical_small_molecule; target=Alvespimycin; dose=10.0(原始单位); control=vehicle/matched; task_id` 包含 dose | `CPA_RDKIT`，`predicted_effect_key` 指向 effect 向量；manifest 中 `target_truth_used_for_prediction=false` 才能进入。 |
| E208 Jiang24 | `context=hap1; type=genetic_repression/机制待协议确认; target=CLK1; state=IFNG; control=control; task_id=hap1|IFNG::CLK1` | 输出轴为 15,473 基因；test truth 封存，`target_truth_accessed=false`。 |
| Tahoe E260 | `context=CVCL_0131; type=chemical_small_molecule; target=(R)-Verapamil (hydrochloride); dose=0.05 uM; plate=plate4; control=matched_control` | 该样例属于 `test_sealed`；当前仅验证任务层，不能填造预测向量。 |
| Wessels | `context=THP-1(以原始记录为准); type=genetic_combinatorial; targets=[DOT1L,EP300]; control=非靶向/协议定义` | 用于组合对象和成员排序的 round-trip 测试；没有合法预测向量时不进入风险训练。 |

## 8. 接口验证实验设计（本轮真正要做的实验）

所有实验均是 metadata/adapter/contract 级别，使用现有文件，不训练模型、不重新下载、不读取封存测试真值。每个实验输出机器可读表和状态 JSON。

### T0：数据基线冻结

**输入**：`dataset_manifest.json`、83 个本地 h5ad 的 backed metadata、E131/E118/E208/E260 既有报告。
**动作**：记录文件 hash、读取状态、字段覆盖和原始 split；保留现有清单作为不可变输入。
**通过条件**：83/83 可读、0 下载、0 表达矩阵载入；若数据状态变化则停止后续实验。

### T1：统一字段覆盖审计

**目标**：回答“候选结构能否覆盖现实数据”。
**动作**：将每个 manifest 记录映射到 `UnifiedTaskRecord`，对每个硬字段输出 `present/unknown/not_applicable/invalid`，并记录 subtype confidence。
**指标**：硬字段覆盖率、类型细化率、背景碰撞数、显式缺失率、未分类记录数。
**通过条件**：所有记录都能得到合法状态（valid 或 scope_limited），无静默空值；基因和化学线各至少有一套 `valid` 数据集。

### T2：任务身份和规范化 round-trip

**目标**：保证多个上游模型对同一真实任务共享同一 `task_id`。
**动作**：对 E131 基因、E118 化学和 E260 Tahoe 元数据生成 canonical JSON→task_id→JSON；交换预测器 ID 不得改变 task_id；改变 dose、time、context 或组合成员不得产生相同 task_id；组合成员排序前后应相同。
**通过条件**：0 collision、0 非预期 duplicate；同任务跨预测器键一致，条件变化键必变。
**额外检查**：Unicode、单位规范化、浮点 dose 序列化和缺失状态不能造成隐性重复。

### T3：预测适配器 round-trip

**目标**：确认 GEARS/scGPT/V0/ContextSim/CPA 的已有产物能转换到同一 `predicted_effect` 契约。
**动作**：优先抽取已有 NPZ/向量；对 post-expression 仅在匹配对照存在时执行“预测状态−对照”；对原生 effect 保留原值并登记 `output_kind`；校验 shape、有限值、gene-axis hash、任务键和 provenance。
**通过条件**：抽样记录全部能定位到向量；0 个 metric-only 表被接收；基因轴不一致时明确 fail，而不是静默截断；`target_truth_accessed` 可追溯。

### T4：跨类型契约平行性实验

**目标**：证明同一套公共输入和失败码同时适用于基因和化学，而不证明两类生物学误差相同。
**样本**：一套 E131/E208 基因任务和一套 E83/E118 化学任务；另取 Tahoe 任务层记录。
**动作**：运行同一 schema validator、effect 特征提取器和 missingness mask；分别检查 context、target、control、output contract、split/provenance。
**通过条件**：公共字段名称、数据类型、失败码完全一致；类型特有字段只进入对应 mask/adapter；报告必须分基因和化学，不合并一个 headline 指标。

### T5：split 和泄漏审计

**目标**：确认“公共历史”和错误记忆不会偷看当前任务真值。
**动作**：核对 E131 LOPO fold、E118 数据集外验证、E208 test sealed、Tahoe train/validation/test_sealed；逐条记录模型训练数据 hash、split manifest hash 和 `target_truth_accessed`。
**通过条件**：测试任务的 treated truth 在预测、校准和历史构造冻结前均为 false；历史只能来自允许的训练/OOF 任务；无法证明 OOF 的错误记录标记为 `provenance_unverified`，不进入正式 error memory。

### T6：类型特有字段缺失压力测试

**目标**：避免把药物字段或基因字段错误变成统一模型的硬依赖。
**动作**：在不改变公共字段的副本上逐一遮蔽 dose/time、guide/mechanism、SMILES、batch；运行 validator 和 feature builder。
**通过条件**：公共 SafeConf 特征可以在硬字段满足时运行；依赖被遮蔽字段的专用特征明确关闭；不能把未知编码成零；缺少 context/target/control/output 时必须 fail closed。

### T7：接口通过后的模型实验（后续，不在本轮运行）

只有 T0–T6 全部通过后才进入模型比较：

1. 基因验证线：E131 LOPO 开发、E208 或 E258 donor/context-held-out 确认；
2. 化学验证线：E118/sciPlex 开发、Tahoe 或独立化学任务确认；
3. 比较三种风险器：共享公共特征、共享主干+类型特有 mask、完全分类型校准器；
4. 训练和校准严格按任务/背景/药物/基因的外推 split；不把基因和化学结果混成一个 headline；
5. 只有在两条线都达到预先登记的覆盖/风险门槛后，才讨论“统一 SafeConf”是否成立。

## 9. 接受标准、停止标准和产物

- **停止条件**：任何硬字段的静默填充、task_id collision、gene-axis 不一致未报错、测试真值泄漏、或预测向量无法定位；停止后修复契约，不扩大样本。
- **本轮不报告**：新的 RMSE、coverage、模型提升或 E274 曲线；接口实验通过不等于 SafeConf 已在两类扰动上有效。
- **本轮产物**：本规范、全量 CSV 盘点、状态 JSON、T0–T6 的 validator/adapter 输出。所有记录必须包含输入文件和 hash，方便下午复核。

## 10. 实验计划文件

可执行的 T0–T7 实验矩阵、资源边界、验收顺序和后续模型登记见 [`UNIFIED_PERTURBATION_EXPERIMENT_PLAN.md`](UNIFIED_PERTURBATION_EXPERIMENT_PLAN.md)。本轮只完成 T0 元数据冻结和接口设计，T1–T6 在契约实现后执行，T7 保持 pending。

## 11. 最终判断

这个接口**足以作为同时覆盖基因扰动和化学扰动的任务表示与适配层**：两类数据都能表达为背景、对象、条件、参考状态和固定输出响应。它还不足以单独证明一个完全共享参数的风险模型合理。当前最稳妥、可证伪的路线是：**共享 SafeConf 主干，加入扰动类型 embedding 和显式缺失 mask，再用很小的类型特有校准层；训练和报告仍分基因线、化学线验证**。这保留统一方法的可迁移性，也避免 Tahoe 的药物字段把接口或结论锁死。
