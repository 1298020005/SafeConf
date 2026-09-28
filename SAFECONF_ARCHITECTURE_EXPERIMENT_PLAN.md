# SafeConf 主方法决定性实验设计

> 2026-09-28 更新：9 月 30 日交开题与进展，论文稿后续推进。当前实验顺序见 [最小决定性实验](docs/方法设计/20260928_证据驱动开题/实验与投稿路线.md)，优先同信息、同反馈预算与完整 Q；本旧计划中的估时不代表任务已排队或启动。

**版本**：v0.2（2026-09-28）
**目标**：决定 SafeConf 是否应保留 `Prediction Evidence + Public History + Model Error Memory`，以及是否有资格升级到 residual correction 和 learned gates。

> 这份文件是方法实验设计。`UNIFIED_PERTURBATION_TASK_SPEC.md` 只是任务/预测接口的前置契约，不能替代本文件。

## 0. Phase -1 研究结果如何进入实验

外部研究先确定比较对象，再决定是否值得增加模型复杂度：

| 研究发现 | 实验动作 |
|---|---|
| PertEMA 已有冻结预测器、预测时特征、OOF error、GBM 和后置校准的近邻合同 | V0 必须加入 PertEMA-style comparator；SafeConf 的主增量只能来自 `Q/H`、冷启动、跨类型和严格 provenance。PertEMA 当前按软件证据登记。 |
| PRESCRIBE/GPerturb 提供上游内生 uncertainty | 将 native uncertainty 作为可选 `P` 特征和消融项，不能把它与 SafeConf 的外接风险层混为同一方法。 |
| HopCPT 已使用相似历史 error regime 加权 conformal | 把 retrieval + error memory + calibration 作为已有组合；加入 history shuffle、no-history、same-condition leakage 和冷切分对照。 |
| 2026 可靠性预印本显示实验测量质量影响模型比较 | `Q` 先于 `H` 做增量；source/cell/replicate/noise 必须单独报告，禁止把质量标签直接写成模型错误。 |
| TabM 提供高效表格 ensemble；TabR/Deep Sets/Set Transformer 增加计算和数据需求 | 先跑 Ridge/GBM/TabM-style V0，只有固定汇总有稳定 residual signal 才启动集合编码或 learned retrieval。 |
| scArchon 与近期 benchmark 强调 split/metric/provenance | 所有主结果采用冷切分、任务级样本和 provenance ledger；随机行切分只作 sanity check。 |

本文件因此安排的是“信息增量决策实验”，不是预设复杂网络的训练计划。若 V0 不能证明 `H` 或 `E` 的合法增量，后续对应分支自动停止。

## 1. 要验证的主张

SafeConf 不是第二个扰动预测器，而是对冻结上游预测的 post-hoc 风险判断器。最终只允许在以下主张被实验支持后写入论文：

1. **基础风险**：预测本身的 evidence（尤其是 magnitude）能够预测上游真实误差，并提供比原始幅度排序更稳定的风险排序。
2. **历史质量**：来源数、细胞数、重复数和测量稳定性等 `Q` 能在 `P` 之外提供可复现信息。
3. **公共历史内容**：在控制 `Q` 后，相关历史实验的效应、离散度和 prediction-history gap 仍能解释 residual error。
4. **错误记忆**：只有在严格 OOF/held-out 合同成立时，模型自身历史错误 `E` 才能增加风险判断价值。
5. **证据退化**：没有 `H` 或 `E` 时，模型可以退回到合法的 `P` 或 `P+Q`；不能把缺失证据伪装成低风险。
6. **跨扰动类型**：基因和化学只共享风险语义与输出契约；性能必须分别验证，不能用一条混合分数掩盖某一类型失败。

## 2. 反事实假设和决定性比较

| 假设 | 实验比较 | 通过标准 |
|---|---|---|
| H1：magnitude 是可靠锚点 | `Raw M` vs `Learned P` | 在扰动/背景冷切分中，`P` 的 Utility@20、Spearman 不低于 Raw M，且跨单元方向稳定。 |
| H2：历史质量本身有增量 | `P` vs `P+Q` | `Delta_Q = U(P+Q)-U(P)` 在预登记单元中稳定为正；报告分层区间。 |
| H3：历史内容有超出质量的增量 | `P+Q` vs `P+Q+H` | `Delta_H = U(P+Q+H)-U(P+Q)` 在多数外层单元正向，且不是由 source count/quality proxy 造成。 |
| H4：模型反馈能适应 | `P+Q+H` vs `P+Q+H+E` | 仅使用合法 OOF error memory 时 `Delta_E` 正向；无合同时不得报告 E 增益。 |
| H5：复杂结构值得保留 | concat vs residual vs residual+gates | 复杂模型需在相同标签预算和参数级别下重复优于简单模型；否则删除复杂组件。 |
| H6：冷启动可用 | `P+H` 与 `P+H+E`，E=0/少量/充足 | E=0 时仍有可用排序；随合法反馈增加，收益曲线可复现或明确饱和。 |

## 3. 统一样本、标签和输出

### 3.1 样本单位

所有训练和评价以 `task × upstream_model × fold` 为逻辑样本。不要把一个任务的 5000 个基因值当成 5000 个独立监督样本。

每条样本必须含：

```text
record_id
dataset_id
context_id
perturbation_id
perturbation_type
upstream_model_id
split/fold
predicted_effect_ref
true_effect_ref（仅在允许的训练/验证阶段）
actual_error
P/Q/H/E availability flags
provenance
```

### 3.2 固定误差标签

第一轮沿用仓库已经冻结的 task-level error object，不同时换 RMSE、相对误差和 rank label。默认主标签为预测 effect 与真实 effect 在固定基因轴上的 RMSE；可报告绝对/相对误差作为辅助，但不能用结果选择主标签。

### 3.3 输出

主输出：`expected_error` 和风险排序。
辅助输出：`Utility@10/20/30`、Spearman、AURC/risk-coverage、top-k error capture、MAE/Huber。
决策输出：`REVIEW`、`PASS`、`ABSTAIN`，必须在独立 calibration split 上冻结阈值；没有校准证据时只报告 ranking，不声称概率置信度。

## 4. 数据线路和角色

### 4.1 混合类型开发面板（旧标题“基因开发线路”已纠正）

使用已有 E131/LOPO 任务级预测和误差表，当前可直接核验的开发单元包括：

- CuiHacohen2023
- Frangieh
- LaraAstiasoHuntly2023_exvivo
- LaraAstiasoHuntly2023_invivo
- McFarlandTsherniak2020
- SantinhaPlatt2023
- SrivatsanTrapnell2020_sciplex3

这些单元已有开发材料，其中 McFarland 与 sci-Plex3 为化学线，不应列作七个基因研究。E273 的 Q 只是相似度/支持数代理，E 的上游 OOF 合同仍需核验。基因机制不完整的记录保留 unknown，不能因名称猜测为 KO、CRISPRi 或 CRISPRa。

确认线路使用：

- E208 Jiang24：context/state-held-out，test treated truth 继续封存；
- E258 Feng：donor-held-out，最终 test truth 不提前打开。

### 4.2 化学开发线路

使用 E118/E84/E87/E89 中已有的 sciPlex/CPA 任务级预测和 effect 记录，按 `context × compound × dose × time` 组织。当前 E118 的化学独立增量门结果必须原样保留，不可用新的特征选择改写成成功。

确认线路使用 Tahoe E260/E263：

- train：1200 个 cell-line × drug-dose 任务；
- validation：600 个任务；
- final test：600 个 `test_sealed` 任务；
- 当前 E263 只允许使用 train/validation treatment 与匹配 control；final test treated truth 保持封存。

Tahoe 在本轮首先用于公共历史和上游 competence 合同；上游预测器未通过 competence gate 前，不把 Tahoe 风险结果写成 SafeConf 外部证据。

### 4.3 跨类型报告

基因、化学分开报告每个方法和每个 split 的指标，再报告宏平均与层级 bootstrap 区间。禁止把两类任务直接拼接后给出一个 headline Utility@20。

## 5. 切分和泄漏规则

1. **外层冷切分**：开发数据使用扰动冷桶、context-held-out 或 donor-held-out；同一 perturbation/目标不可同时进入 fit 和 evaluation。
2. **公共历史**：当前目标的 treated truth 永远不进入自己的 `H`；历史只能来自允许的 train/reference pool。
3. **错误记忆**：`E` 只能来自 OOF/held-out 上游预测；训练内误差必须标记 invalid。
4. **同一行排除**：风险器拟合行不能检索自己的真实误差；至少使用 leave-one-out 或独立 OOF memory。
5. **幅度标准化**：`magnitude_percentile` 只用允许训练参考分布计算，不使用 validation/test treated truth。
6. **模型隔离**：GEARS 的 `E` 不迁移给 scGPT/CPA；跨模型仅在另行登记的 leave-one-model-out 试验中验证。
7. **预先冻结**：split、标签预算、feature list、Top-K 候选、主指标和停止门先写入状态文件，再运行。

## 6. V0：证据增量实验（第一优先级）

V0 只用简单、可审计的风险学习器：Ridge、HistGradientBoosting 或小型 MLP。所有候选使用同一任务表、同一外层 split、同一调参预算和同一真实误差。

### 6.1 逐步模型

| 编号 | 输入 | 作用 |
|---|---|---|
| B0 | 常数/训练集均值 | 误差预测下限，检查任务标签是否有结构。 |
| B1 | `Raw M` | 原始 prediction magnitude 排序基线。 |
| B2 | `P` | magnitude、绝对均值、稀疏度/shape summary、合法 ensemble disagreement/native uncertainty。 |
| B3 | `P+Q` | 加 history availability、source/cell/replicate、split-half noise、provenance、coverage。 |
| B4 | `P+Q+H` | 加历史效应均值/离散度、prediction-history gap、相关性和固定 Top-K 汇总。 |
| B5 | `P+Q+E` | 加合法 model-specific OOF memory；无合法 E 时标记 NOT AVAILABLE。 |
| B6 | `P+Q+H+E` | 完整双历史候选，仅在 B5 合同通过后运行。 |
| B7 | PertEMA-style | 使用同一 `P` 与合法 OOF `E` 的轻量 GBM + isotonic/split-conformal 组合，作为最近邻外部合同基线；不能把未公开实现细节臆测进模型。 |

### 6.2 公共历史最小实现

第一轮不训练检索器：

1. exact perturbation + exact dose/time 优先；
2. 按预先登记的 context 相似性排序；
3. 固定候选 `K ∈ {8,16,32}`，只在开发阶段选择一次；
4. 每条历史保留 relevance、quality、coverage；
5. 用均值、加权均值、离散度和当前预测 gap 汇总；
6. 当前目标背景的 treated truth 必须剔除。

如果 B4 没有稳定 residual signal，不升级到 Deep Sets/Set Transformer。

### 6.3 V0 统计

对每个 `dataset × predictor × cold fold` 计算差值：

```text
Delta_Q = U(B3) - U(B2)
Delta_H = U(B4) - U(B3)
Delta_E = U(B6) - U(B4)
```

另外报告 `U(B7)-U(B2)` 和 `U(B7)-U(B5)`。若 B7 已经达到 B6 的性能，SafeConf 只能围绕公共历史质量、冷启动或跨类型审计提出差异；不能把 post-hoc error judge 重新命名为新方法。

同时报告每个单元的正负方向、层级 bootstrap 区间和数据覆盖率。不能只报宏平均。

### 6.4 V0 停止门

- B2 不优于 B1：先检查 P 特征/上游输出合同，不做 V1。
- `Delta_Q` 不稳定：保留 availability/support 审计，但不宣称 quality learning。
- `Delta_H` 不稳定：公共历史降为 optional，不实现复杂 history encoder。
- `E` 无合法 OOF：主线暂时写 `P+Q+H`，E 只做协议待办。
- B6 只在少数单元变好：按数据集/模型拆开解释，不写“普遍有效”。

## 7. V1：结构实验（只有 V0 通过才启动）

V1 不再增加新信息，只比较结构是否值得复杂化。

### A. 简单拼接

```text
risk = SmallModel([P, Q, H, E])
```

### B. 残差校正

```text
base = BaseRisk(P)
correction_H = HistoryModel(P, Q, H)
correction_E = ErrorMemoryModel(P, E)
risk = base + correction_H + correction_E
```

### C. 残差 + 小型 gate

```text
risk = base + gH(P,Q,H) * correction_H + gE(P,E) * correction_E
```

gate 只允许使用 evidence availability、support、similarity、dispersion 和 memory count。第一版不使用 MoE、Transformer 或大模型。

### V1 接受标准

在相同参数量级、相同标签预算和相同外层 split 下：

- C 必须相对 A/B 在多数外层单元稳定提升；
- 若 A≈B≈C，保留 A；
- 若 B≈C，删除 gate；
- 若固定汇总≈record-level encoder，删除 encoder。

## 8. E：错误记忆预算与冷启动实验

### 8.1 反馈预算曲线

固定 `P/Q/H`、上游模型和任务集合，只改变合法 E memory 的可用比例：

```text
0%, 5%, 10%, 25%, 50%, 100%
```

比例由任务 key 的固定 hash 决定，每个比例不可根据结果换样本。输出 `Utility@20`、Spearman、正向单元比例和饱和点。

现有 E274 的 0–100% 曲线只能作为开发敏感性证据；在逐行 OOF provenance 完成前，不升级为主结论。

### 8.2 证据缺失矩阵

固定四种状态：

```text
P only
P + H
P + E
P + H + E
```

每个状态都显式写 `has_history`、`has_error_memory` 和 coverage。缺失证据时允许退化，不允许把缺失填成低风险。`risk high + support low` 进入 `ABSTAIN` 候选，而不是 `PASS`。

## 9. 上游 competence gate

在 Tahoe 上训练或接入风险模型前，先确认冻结上游预测器本身合格：

1. 与 no-change、control-relative mean、strong simple baseline 比较；
2. 只用 train/validation 选择模型和输出轴；
3. 不通过 competence gate 的上游模型只进入接口测试，不进入 SafeConf 主结果；
4. 记录 model version、gene axis、normalization、control 和任务覆盖率。

这一步防止用下游风险模型掩盖上游预测失败。

## 10. 校准和决策层实验

校准不是主创新，必须置于风险模型之后：

1. 在独立 validation/calibration split 上将 raw risk 映射到 expected error；
2. 比较 isotonic 与简单线性校准，候选数很小；
3. 只在 calibration 固定后报告 conformal/split-conformal interval；
4. 用 risk-coverage 曲线评价 `REVIEW/PASS/ABSTAIN`，不把 support 直接命名为 confidence probability。

主要问题是：在固定 review budget 下，SafeConf 是否能捕获更多高误差任务；不是让模型输出一个看起来精确的小数。

## 11. 跨扰动类型实验

这是方法通用性实验，不是字段盘点：

### 11.1 分线验证

- 基因：E131 开发，E208/E258 确认；
- 化学：E118/sciPlex 开发，Tahoe 确认；
- 每条线分别训练和评估 B1–B6。

### 11.2 统一与分类型结构对照

在公共字段完全一致、类型特有字段使用 mask 的条件下比较：

```text
S1: 基因和化学分别训练
S2: 共享主干 + perturbation_type embedding + masked typed fields
S3: 完全扁平共享模型
```

主判据：外推 split 上的平均风险、最差分线风险、校准误差和 coverage。完全共享模型若只靠某一类型贡献提升宏平均，应判为失败。

### 11.3 只做合法 transfer

只有任务 definition、输出轴、误差对象和 provenance 对齐时，才做 leave-one-model-out 或 leave-one-dataset-out；否则标记 NOT RUN，不用接口兼容替代风险迁移证据。

## 12. 最终确认实验

在所有方法、特征、阈值、模型版本和 hash 冻结后：

1. 选择唯一最终方案；
2. 只解封一次 E208/Tahoe final test treated truth；
3. 先报告上游 competence，再报告 SafeConf risk/coverage；
4. 按基因/化学分线报告，同时给出 pooled 结果作为附录信息；
5. 不允许根据 final test 结果回改模型。

## 13. 资源和执行顺序

本轮方法判断优先复用已存在的 task-level 表和预测产物。接口、V0 表格和泄漏审计以 CPU 为主；GPU 只用于 V1 小型 MLP 的固定种子复现，不用 GPU 空转证明科研进度。

执行顺序：

```text
A. 冻结输入、split、error object 和 leakage ledger
B. 运行/复核 upstream competence gate
C. 完成 V0 B0–B7（含 PertEMA-style 合同基线）
D. 输出 Delta_Q / Delta_H / Delta_E 和分层区间
E. 仅当 V0 通过才运行 V1 A/B/C
F. 运行 E budget + evidence missing
G. 运行基因/化学分线及共享主干对照
H. calibration、risk-coverage、abstain
I. 最终冻结后一次性 sealed confirmation
```

### 13.1 可执行的时间盒

下面给出排程用时间盒，实际耗时以缓存和日志为准；Phase -1 完成后才允许进入相应门控。

| 时间盒 | CPU 工作 | GPU 工作 | 产物 | 启动条件 |
|---|---|---|---|---|
| 0–45 分钟 | 锁定 schema、split、error label、feature hash、leakage ledger | 不启动 | `RUN_MANIFEST` 与输入审计 | 本文件版本冻结 |
| 45–150 分钟 | B0–B7、每个外层单元 delta、bootstrap | 不启动 | V0 表和停止门 | 输入审计通过 |
| 150–210 分钟 | provenance/quality 分层和 history shuffle | 小型 GPU seed 复核 B2–B6（若有必要） | 质量控制报告 | V0 不是明显失败 |
| 210 分钟以后 | V1 A/B/C、E budget、跨类型汇总 | 仅运行小 MLP/gate 固定 seeds | 架构选择报告 | 对应增量门通过 |

这个时间盒不会把 GPU 空转当成进度：GPU 只运行有明确假设和 CPU 结果支持的结构复核；没有通过门时继续做 CPU 审计或停止分支。

## 14. 当前仓库结果如何归位

- E273：已经运行了开发阶段的 `P`、`P+Q`、`P+Q+H`、`P+Q+E`、`P+Q+H+E` 比较，可作为 V0 开发底稿；需继续受 OOF provenance 限制。
- E274：完成 Tahoe raw panel 合同审计、E memory 预算敏感性和 GPU 稳定性，可作为开发证据；不能替代 Tahoe 上游 competence 或 sealed test。
- E230/E234/E235/E258/E259：用于历史容量、简单基线、固定历史离散度、外部确认和历史质量分解，不重复包装为新结果。
- `UNIFIED_PERTURBATION_TASK_SPEC.md`：提供任务/预测接口，不是方法效果证据。

## 15. 结果到方法的决策表

| 观察 | 方法决定 |
|---|---|
| 只有 Raw M/P 稳定 | 收缩为 prediction-evidence risk audit。 |
| Q 稳定、H 不稳定 | 保留 P+Q，H 作为可选诊断。 |
| H 稳定、E 无合法 provenance | 主方法 P+Q+H，E 延后。 |
| E 稳定、H 弱 | 重新评估与 PertEMA 的区分，不能强留 public-history 叙事。 |
| H/E 都稳定，简单拼接最好 | 采用简单可审计模型，不保留 gate。 |
| residual+gate 明显稳定优于简单模型 | 才把 gated residual 写入主方法。 |
| 基因有效、化学失败（或反之） | 方法结论按类型收缩，不能宣布统一 SafeConf。 |
| 缺证据时风险虚假下降 | 增加 abstain/support 约束，禁止直接部署。 |

## 16. 本轮交付

本文件和已有 E273/E274 结果共同构成 SafeConf 方法实验设计。下一次运行只允许由本文件的通过门触发；不再用接口盘点代替方法实验，也不追加没有科学问题对应的模型。
