# ARCHITECTURE_DERIVATION

> 2026-09-28 更新：本文件保留为候选架构调研。当前开题按用户纠正后的“现有证据 → 条件风险 → 下一实验”组织，以 [证据驱动开题](docs/方法设计/20260928_证据驱动开题/README.md) 为主入口。新增推导明确区分 E273 的代理 Q、E274 的记忆稀疏化与真正零反馈，并允许删除 H。

**版本**：v0.2（2026-09-28）
**输入**：`WEB_RESEARCH_MAP.md`、`RELATED_METHOD_MATRIX.csv`、仓库 E230/E234/E235/E258/E259/E273/E274 证据和当前基因/化学数据合同。

## 0. 外部研究对推导的新增约束

本轮检索不是为了证明既有“双历史”一定成立，而是先检查相邻工作是否已经覆盖了同一问题。

1. **PertEMA 是必须加入的近邻基线。** 它的公开软件实现已经采用冻结预测器的预测时特征、OOF error、梯度提升元模型和后置校准。因此“训练一个 error judge”不能单独构成 SafeConf 的新颖性。PertEMA 当前按软件证据记录，尚无可核验的正式同行评审论文；实验中必须把它当作同合同 comparator，而不是引用成既定科学结论。
2. **质量变量和模型错误必须分层。** 2026 年可靠性预印本在多数据集上显示扰动测量质量会显著改变模型比较。`Q`（细胞数、重复、split-half 噪声、来源）必须作为独立输入和控制项；`H` 只有在控制 `Q` 后仍有 residual signal 才能进入主叙事。
3. **检索历史错误不是空白。** HopCPT 已将相似历史 error regime 与加权 conformal 结合。SafeConf 的可证伪差异只能来自单细胞扰动的公共实验历史、冻结黑盒跨模型协议和 gene/chemical 类型适配，不能声称 retrieval + error memory 本身原创。
4. **简单模型优先级上升。** TabM 提供了比注意力检索器更便宜的表格 ensemble baseline。当前任务表规模和 E259 的负结果都不支持直接进入 Set Transformer、TabR 或大门控网络。
5. **最新 benchmark 结果要求更严格的 split。** scArchon 和新一轮 perturbation benchmark 都强调数据集、扰动、context、donor、dose/time 的切分会改变结论；因此随机行切分只用于 sanity check，主结果使用已经登记的冷切分。

这些约束使推荐架构更加保守：先做 A（最小后置 judge），再按增量结果决定 B/C；跨类型只共享风险语义和小主干，不承诺零样本类型迁移。

## 1. 先排除三个错误起点

### 错误起点 1：把 SafeConf 做成第二个 perturbation predictor

GEARS、CPA、scGPT、scFoundation 和 PRESCRIBE 已经覆盖了上游 response prediction 或模型内 uncertainty。SafeConf 的输入必须是冻结上游的 canonical effect；它只能估计预测风险、排序和是否复核，不能再次生成表达结果。

### 错误起点 2：固定 `0.4M+0.3H+0.3E`

公共历史的数量、质量和相关性因任务而变，模型错误历史在新模型 cold-start 时不存在。固定权重既不能处理证据缺失，也不能区分“有历史”和“有可信历史”。

### 错误起点 3：一开始做 retrieval Transformer / 两个复杂 gate

TabR、Deep Sets 和 Set Transformer 说明这些模块在相应任务上可用，但没有证明当前 SafeConf 的历史内容有 residual signal。E259 已经提示，控制 source count、cell count、split-half noise 后，部分历史增量会消失。因此复杂结构必须由 V0 的增量结果触发。

## 2. 从外部方法拆出的零件

### 零件 A：冻结预测器后的 error judge

来源：PertEMA、post-hoc confidence/error estimation。
结论：这是 SafeConf 的基础范式，但不再是充分创新。必须复现同等严格的 prediction-time features + OOF label 合同，并寻找 public history/cold-start 的独立价值。

### 零件 B：selective risk / abstention

来源：risk-coverage、AURC、learning-to-defer、selective regression。
结论：SafeConf 评价的最终单位是 review budget 下的 selective risk，而不是一个没有行动含义的 confidence number。`ABSTAIN` 需要 evidence support 或 calibration 依据。

### 零件 C：后置 calibration / conformal risk control

来源：Conformal Risk Control 和多源 shift calibration。
结论：把 raw risk 映射成 expected error、interval 或风险受控阈值；不把 conformal 包装成风险表示的原创模块。shift 场景必须单独标记假设。

### 零件 D：外部 retrieval / history memory

来源：TabR、case-based/retrieval learning 和项目已有 Public History。
结论：历史内容可以作为当前任务的额外证据，但先做 exact condition + fixed K + quality weighting；对 retrieval 的价值做增量检验。

### 零件 E：无序集合聚合

来源：Deep Sets、Set Transformer。
结论：历史记录数量不固定时需要 permutation-invariant 聚合；第一版用加权统计，第二版才允许 Deep Sets，Set Transformer 只作为大 K 或交互明显时的候选。

### 零件 F：append-only memory update

来源：experience replay/continual memory。
结论：新模型无错误历史时退回 P/P+H；真实 OOF feedback 回来后先只追加 E memory，达到固定预算再周期 retrain。逐条在线改参数不是第一版。

### 零件 G：domain-conditioned shared core

来源：多域 adapter、missing-modality 和 domain generalization。
结论：基因/化学共享 effect/risk 语义，使用 perturbation-type embedding、typed optional masks 和小型 type-specific calibration；不使用完全扁平输入或两套大模型。

### 零件 H：predictor/judge 角色分工

来源：verifier/critic/process-supervision。
结论：只借鉴角色划分和标签溯源原则。LLM judge、未公开产品架构和 verbalized confidence 不进入 SafeConf。

## 3. 候选架构 A：最小后置风险器

```text
Frozen predictor
  -> Prediction Adapter
  -> P features + Q/H/E summary
  -> Ridge / GBDT / small MLP
  -> raw risk
  -> calibration + review/abstain
```

### 优点

- 与 PertEMA 的黑盒 judge 范式可公平对照；
- 小样本、CPU 可训练、解释和复现最容易；
- 新模型只需 canonical prediction，E 缺失时仍能运行；
- 适合先验证信息增量。

### 科学风险

- 可能只是已有 error predictor 的重命名；
- 简单汇总会丢失历史成员差异；
- 如果 H 的增量来自质量 proxy，生物历史主张会失败。

### novelty 风险

最高。只有 public history/cold-start/type-generalization 的实验差异能建立区分。

### leakage 风险

中等：必须防止同任务 history 回读、训练内 error memory 和 test distribution percentile。

### 数据需求

已有 task-level prediction/error 表即可；需要合法 OOF E 才能开启 E 分支。

### 基因/化学适用性

两者都适用；化学 dose/time、基因 mechanism/guide 通过 optional masks。

### cold-start

强：E=0 时天然退化到 P/P+Q/H。

### 服务器与时间

CPU 即可；7 个基因数据集×3 predictor 的 V0 已在 E273 做过开发版，重新核验通常以小时计，不需大 GPU。

### reviewer 攻击点

“这和 PertEMA 有什么实质区别？”必须用同合同对照、历史增量、冷启动和跨类型结果回答。

## 4. 候选架构 B：幅度锚定的分层残差 judge

```text
P -> BaseRisk(P)
Q/H -> HistoryCorrection(P,Q,H)
E -> ErrorCorrection(P,E)
raw risk = base + gH * correction_H + gE * correction_E
```

第一版 `gH/gE` 可以先固定为 availability/quality mask 经过小线性层；只有 V0 证明 correction 有价值才学习 gate。

### 优点

- 直接对应当前 E258/E259/E273 的事实：先控制 P/Q，再判断 H/E 的 residual；
- 能表达新模型无 E 时的退化路径；
- 把 public biological history 与 model-specific error memory 分开，语义清楚；
- 对 review/abstain 可单独输出 support。

### 科学风险

- residual label 可能被 base risk 误差放大；
- H/E 相关时分解不唯一；
- gate 容易在小数据上学习到 split/domain shortcut。

### novelty 风险

中等偏高：结构本身接近已有 meta-assessor，但双证据来源、严格 residual 增量和 cold-start 可能形成可证伪差异。

### leakage 风险

高于 A：两套 memory 都可能把 label 间接放回来；必须逐行 provenance ledger、leave-one-out/OOF 和固定历史池。

### 数据需求

需要 prediction-time P、公共历史质量/内容、同模型 OOF E；没有 E 时仍可运行 P+Q+H。

### 基因/化学适用性

适用，但 H retrieval 必须按类型定义 relevance；共享 residual head 前需分别报告。

### cold-start

强：`gE=0` 时不依赖 E；新扰动没有 H 时退回 P。

### 服务器与时间

V0 用 CPU；小 gate/MLP 可在单张 RTX 6000 上做固定 seed 稳定性；无需重训 GEARS/CPA/scGPT。

### reviewer 攻击点

“residual/gate 只是重新包装的 concat。”必须用同参数预算的 A/B/C 比较；若没有稳定差异，主动删 gate。

## 5. 候选架构 C：检索增强的集合 judge

```text
Prediction encoder
+ Public History Retriever -> set encoder
+ Error Memory Retriever -> set encoder
+ conditional fusion
-> risk / support / abstain
```

候选集合聚合顺序：weighted mean → Deep Sets → Set Transformer。不能跳过前两级。

### 优点

- 能保留每条历史的 relevance/quality，而非只存均值；
- Top-K 数量不固定，符合 history memory 的真实形式；
- 可以直接输出 retrieval support 和 evidence coverage。

### 科学风险

- 当前 task-level 样本量可能不足以学习稳定相似度；
- retrieval 结果和历史来源高度相关，容易把 context/dataset identity 当作风险；
- 高 K 和 attention 增加计算/调参，同时不一定提升。

### novelty 风险

中等：TabR/Deep Sets/Set Transformer 已有通用结构，SafeConf 的新意只能来自风险任务和历史合同。

### leakage 风险

最高：训练 candidate pool、query embedding 和 error memory 必须都按 outer split 隔离；同一任务或重复来源被召回会造成 label leakage。

### 数据需求

需要足够多且可比的历史记录，及固定的 task/context/condition similarity；Tahoe 的化学 panel更适合，基因跨研究需要额外协议。

### 基因/化学适用性

化学同 drug-dose 跨 context 可能较自然；基因需要机制/guide/背景对齐。不能假设两类共享同一 similarity。

### cold-start

Public history cold-start 仍困难；E=0 可运行，但无 history 时只能退回 base risk。

### 服务器与时间

固定 K 的小 Deep Sets 可运行；动态 learned retrieval 需要更多内存和实验时间，当前不优先。

### reviewer 攻击点

“复杂 retrieval 是否真的超过 quality-aware summary？”需要 permutation test、K 消融、same-history content shuffle 和 no-history 对照。

## 6. 候选架构 D：共享风险主干 + 类型适配器

```text
P/Q/H/E public feature contract
             |
     shared small risk backbone
             |
   perturbation-type embedding
             |
  gene adapter / chemical adapter
             |
 separate calibration + decision layer
```

### 优点

- 直接回应基因与化学输入差异；
- 共享风险语义但允许 dose/time 与 guide/mechanism 有不同尺度；
- 参数小，适合当前服务器和样本规模。

### 科学风险

- 类型 embedding 可能只学到 dataset shortcut；
- 共享主干若两个类型的误差规律不一致，会拖累强线；
- 当前合法跨类型预测/error 记录数量可能不足。

### novelty 风险

中等；共享主干+adapter 是成熟范式，本身不是贡献。

### leakage 风险

和 A/B 相同，另需防止 type-specific calibration 看见外部 test。

### 数据需求

至少一套合法基因开发线和一套合法化学开发线，公共 output/error contract 对齐。

### 基因/化学适用性

这是最稳妥的统一方案；若跨类型 transfer 失败，保留共享接口、分线模型。

### cold-start

对新上游模型强；对新扰动类型需要重新校准 adapter，不能承诺零样本。

### 服务器与时间

小主干+小 adapter 可在 CPU/GPU 快速训练；不需要大型多域模型。

### reviewer 攻击点

必须和“完全分类型”和“完全共享”做外推对照，报告最差类型而非只报 pooled mean。

## 6.1 资源和时间预算（不含数据准备）

以下按现有约 11,000 条任务级开发记录、固定特征表和 2–8 个 seed 估算；它们是排程上限，不是已经运行出的耗时。

| 架构 | 训练资源 | 单次拟合估计 | 7 数据集×3 predictor 的开发核验 | 进入条件 |
|---|---|---:|---:|---|
| A 最小后置风险器 | CPU | 1–10 分钟 | 0.5–2 小时 | 默认第一步 |
| B 分层残差 judge | CPU；小 MLP 可用单张 GPU | 5–30 分钟 | 1–4 小时 | `Delta_H` 或 `Delta_E` 通过 |
| C 检索+集合 judge | CPU/单 GPU，固定 K | 10–60 分钟 | 2–8 小时 | 固定汇总已有稳定 residual |
| D 共享主干+类型适配器 | CPU；小 MLP 可用单张 GPU | 5–30 分钟 | 1–4 小时 | 两条线公共字段和标签合同对齐 |

这些估计不包括 GEARS/CPA/scGPT 重训；SafeConf 实验必须冻结上游。若实际日志超过预算，应先检查 split、历史检索和特征缓存，而不是直接增加 GPU 或模型规模。

## 7. 推导后的推荐

推荐不是直接把 C 的复杂实现写成主方法，而是采用分阶段的 **B-with-A-first**：

```text
第一阶段：A0
Prediction Evidence -> small post-hoc error judge

第二阶段：A1
加入 quality-aware public-history summary

第三阶段：B0
只有 Delta_H 稳定时，改写成 residual correction

第四阶段：B1
只有 residual 相对 concat 稳定提升，才加入 learned gates

跨类型：共享公共 risk backbone + type embedding/masks + type-specific calibration
```

推荐理由：

1. PertEMA 已经把“prediction-time features + OOF error -> reliability”做成近邻基线，必须先把同类基线做扎实；
2. 当前 E259/E273 的事实只足以要求增量实验，尚不足以承诺复杂双历史；
3. B 能表达 cold-start 和双历史语义，但 A 先行能防止 residual/gate 只是装饰；
4. C 的可变集合表示只有在历史内容确实有增量时才值得付出复杂度；
5. D 解决跨基因/化学的实际输入差异，但不把“共享架构”写成已验证的性能结论。

## 8. 必须允许推翻推荐的结果

- `Delta_H≈0`：删公共历史内容，保留 Q/support；
- `Delta_E≈0` 或无法证明 OOF：E 降为 future extension；
- `A≈B≈C`：采用 A；
- 化学和基因方向相反：分线方法，统一只保留接口和评估协议；
- C 只在随机 split 有效、冷 split 失败：判为 retrieval leakage/shortcut；
- 上游 competence 不过：停止解释 SafeConf 风险。
