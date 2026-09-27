# SafeConf：主方法审查、独立判断与实验迭代计划

- **版本**：2026-09-27 v1
- **依据**：`Desktop/SafeConf_Codex_主方法构想与实验安排_20260927.md`、`Desktop/SafeConf_幅度锚定双历史风险模型_方法设计沉淀_20260927.md`
- **执行分支**：`exp/e220-reviewer-closure-20260921`
- **本轮性质**：方法审查 + 开发实验；不是最终盲测，不打开 Tahoe sealed test treated truth。

## 1. 我的结论

我**部分同意**“Prediction 基础风险 + Public History + Model Error Memory”的研究问题，但不同意现在直接把候选 V1（双分支 encoder、两个 gate、evidence dropout）当作主方法实现。

当前应该把 SafeConf 收缩成一个可证伪的 V0：

```text
P = prediction-side evidence
Q = history availability / measurement quality
H = history content, only when provenance is legal
E = model-specific OOF error memory, optional

risk = simple learner(P, Q, H, E)
```

`magnitude` 保留为无监督强基线和风险锚点；它不能被包装成新发现。复杂 gate 只有在同一任务、同一标签预算、同一外层 split 下稳定超过简单拼接时才有资格进入 V1。

### 为什么收缩

1. E259 已经显示，在补齐来源细胞数和 split-half 质量后，Cui、Lara 和 Feng 的历史内容增量没有稳定复现；未控制质量时出现的增益不能直接称为生物历史增益。
2. PertEMA 已经公开了“固定上游预测器之后，用预测时特征和折外误差训练可靠性估计器”的近邻路线。因此 SafeConf 不能再以“首次给扰动预测加可靠性”作为贡献；必须证明双历史、冷启动退化或严格来源审计带来可重复的额外价值。
3. 当前 E273 的 `E` 是从已有任务级误差表构造的开发特征。除非逐行证明上游预测是 OOF/held-out 产生，不能把它写成合法的 model-specific error memory。E273 正向数字保留，但在本计划中标记为**临时可行性证据**。
4. Deep Sets 和 TabR 只说明集合池化或检索架构可以使用，不说明该架构适合 SafeConf，也不构成生物学贡献。先用 Ridge/小 MLP 解决信息增量问题，避免以结构复杂度替代证据。

## 2. 当前事实清单

| 项目 | 当前判断 | 证据/限制 |
|---|---|---|
| SafeConf 定位 | 已明确是上游扰动预测后的可靠性/风险层，不重训预测器 | `docs/学习导航/README.md`、E220–E235 |
| 幅度 `M` | 强基线，应始终保留；不能直接当创新 | E201、E206、E220、E234 |
| Public History | 有结构化设计，但七研究尚未完成同口径 raw provenance 审计 | E259、E260–E263 |
| Error Memory | 可在开发表上计算；当前 OOF 合同仍需逐行核验 | E273 `STATUS.json` 与本计划第 4 节 |
| Tahoe E263 | 265,984 个已选表达记录；230,400 train/validation treatment，35,584 available control，0 个 test treated 输出 | `E263_tahoe_pretruth_expression_20260925/STATUS.json` |
| Tahoe test treated | 仍 sealed；本轮不读取、不聚合、不评估 | E260–E263 冻结协议 |
| 外部近邻 | PertEMA 是最接近的 post-hoc reliability estimator；PRESCRIBE 是内生不确定性方法 | 公开 README/论文，链接见第 7 节 |
| 现有 E273 | CPU 和双 GPU 已完成 train/val 扰动冷开发检查；结果不能替代 raw provenance 和 blind panel | `E273_dual_history_review_20260927/README.md` |

## 3. 方法定义：先固定语义，再选模型

### 3.1 Prediction Evidence（P）

最小字段固定为：预测效应的 RMS/magnitude、绝对均值、稀疏度/形状摘要、模型间分歧（若确实有多成员）。跨模型比较同时保留 raw magnitude 与仅用允许训练参考任务计算的 percentile；不得用目标 test 分布做标准化。

### 3.2 Public History（Q 与 H 分离）

- `Q`：来源数、细胞数、replicate 数、split-half 质量、批次和 provenance 标记。
- `H`：同 perturbation/dose/time 的历史效应、历史离散度、与当前预测的差异。
- 当前目标背景的 treated truth 永远不能进入它自己的历史库。
- `H_bio_approx` 只能叫测量波动扣除后的近似分解，不能声称纯生物差异。

第一版只做 `Retrieve → Judge → Pool` 的确定性检索：优先 same perturbation + same dose/time，再按允许的 context similarity 取预先固定的 Top-K（8/16/32 中只在开发阶段登记一个候选）。不先做语义检索、图网络或 Transformer。

### 3.3 Error Memory（E）

`E` 只接收同一个 `model_id` 的合法历史。每条误差记录必须同时有：上游模型版本、任务 key、预测产生的 split/fold、实际误差定义、是否 OOF/held-out、可用时间点。训练当前风险器的样本不能把自己的真实标签放回自己的记忆（leave-one-out 或独立 OOF memory）。

如果上述字段缺任意一项，当前实验输出 `E=NOT_AVAILABLE`，退回 `P+Q+H` 或 `P`，不自动填造。

### 3.4 输出

V0 输出风险排序和 expected error；`support` 单独报告 history/error-memory coverage、相似度、离散度和缺失标记。只有校准数据独立且规则冻结后，才增加 interval 或 abstain 阈值。不能把 support 直接叫 neural confidence。

## 4. 关键泄漏审计

1. **目标 treated truth**：Tahoe test treated 不能出现在 history、normalization、feature selection 或阈值选择中。
2. **误差记忆**：禁止把训练内 error 直接当 E；必须有 OOF/held-out 记录或显式 leave-one-out。
3. **同一任务回读**：风险器拟合行不能在自己的 error memory 中；检索要记录排除规则。
4. **跨模型转移**：GEARS 的失败经验不能默认作为 scGPT/CPA 的失败经验；第一版 model-specific。
5. **标签预算**：P、P+Q、P+Q+H、P+Q+H+E 必须使用相同外层 split、相同真实误差、相同调参次数和相同 Utility@20 复核预算。
6. **上游能力**：上游预测器若不能超过预登记的 no-change/strong simple baseline，停止在该研究上解释下游风险排序。

## 5. V0 实验矩阵与停止门

### V0-A：信息增量（必须完成）

固定同一 task-level 表和扰动冷外层 split，比较：

```text
Raw M
Learned M
P
P + Q
P + Q + H
P + Q + H + E（仅在 E 合同通过时）
```

主指标 `Utility@20`，辅助 `Utility@10/30`、Spearman、risk-coverage/AURC。按数据集 × predictor 配对报告，不把所有行当独立样本；bootstrap 至少按任务簇/细胞背景分层。

**升级门**：`P+Q+H` 相对 `P+Q` 在预先固定的多数外层单元中正向，且分层区间不跨 0；否则历史内容降为 optional，保留 Q 和 P。`E` 只有在 OOF provenance 通过后才比较；若 `Delta_E` 不稳定，删除 E 主线。

### V0-B：误差记忆预算

固定 P、Q、H、上游模型、任务集合，只改变 E 可用比例：0/5/10/25/50/100%。每个比例使用固定哈希抽样，保持 held perturbation 不进入 memory pool。输出增量曲线和饱和点，不按结果重新选比例。

### V0-C：证据缺失退化

预先固定四个部署状态：`P`、`P+H`、`P+E`、`P+H+E`。要求缺失证据时显式缺失标志、性能下降可解释；不能因缺失数据自动使用目标答案。

### V0-D：Raw Tahoe 合同

先审计 E263 task key、细胞数量、表达列表有限性和同板 control；再构造 train/validation 的固定基因轴和 pseudobulk。上游预测器能力门通过后，才进入 risk learner。测试 treated truth 只能在预测、风险分数、配置和哈希固定后一次性解封。

### V1 升级条件

只有 V0 证明 `H` 或 `E` 有稳定 residual signal，才实现：

```text
simple concat  vs  residual correction  vs  residual + two small gates
```

如果简单拼接与 gate 无稳定差异，删 gate；如果 pooled summary 与 record-level encoder 无差异，删 encoder。任何复杂组件都有删除资格。

## 6. 今天 17:35–18:00 及之后的执行安排

| 时段 | 任务 | 资源 | 交付 |
|---|---|---|---|
| 17:35–17:50 | E274 error-memory availability 0/5/10/25/50/100% | CPU | `error_memory_curve_cpu/` |
| 17:35–18:00 | E274 Tahoe raw metadata + sparse-expression audit | CPU/IO | `tahoe_raw_audit/` |
| 17:35–18:00 | E274 GPU multi-seed stability，两个 RTX 6000 | GPU0/GPU1 | `gpu_stability_gpu0/`、`gpu_stability_gpu1/` |
| 18:00 后 | 汇总 E274、写入变更日志和 Git 提交 | CPU | `E274` README、摘要、哈希 |
| 下一轮 | 训练/验证 pseudobulk 与上游 competence gate | CPU+GPU | 只有合同通过才启动 |

GPU/CPU 的使用服从可回答的实验问题；如果某一任务提前完成，设备不做无关占用，转入下一条已登记实验或保持空闲并记录原因。

## 7. 相关工作边界

- [PertEMA 官方仓库](https://github.com/OfficialBishal/PertEMA)：最接近的 post-hoc reliability 路线，明确使用 prediction-time features 和 out-of-fold errors，并要求换屏幕重新拟合。SafeConf 必须做同合同对照，不能把自建 Ridge/MLP 称为 PertEMA 复现。
- [PRESCRIBE NeurIPS 2025 论文](https://papers.nips.cc/paper_files/paper/2025/file/d6383e7643415842b48a5077a1b09c98-Paper-Conference.pdf)：学习式、模型内的单细胞响应不确定性；与外挂式、固定上游输出的 SafeConf 端点不同，但“预测可靠性”问题本身已有先例。
- [Deep Sets](https://arxiv.org/abs/1703.06114)：为无序集合池化提供通用结构；不代表 SafeConf 的生物历史编码已经有效。
- [TabR](https://arxiv.org/abs/2307.14338)：检索增强表格模型；可以作为以后架构参考，不作为当前 V0 的必要依赖。

## 8. GitHub 迭代纪律

每一轮提交必须包含：

1. 执行前协议（输入、禁止字段、split、标签预算、主要指标、停止门）；
2. 脚本和固定命令；
3. 小型摘要表、`STATUS.json`、输入/输出 SHA-256；
4. 结果解释、负结果和未完成项；
5. 下一轮只允许由本轮结果触发的变更。

不提交 Tahoe 原始大文件、模型权重、缓存和凭据；只提交可再生脚本、合同、摘要和哈希。提交前检查 `git diff --check`、脚本语法和 GitHub 远端分支。

本轮 GitHub 目标：把本文件、E274 脚本、E274 README/状态摘要及上一轮 E273 审查目录一起提交到当前实验分支；不把 Desktop 网页草稿直接当作已验证结论。

## 9. 当前允许写进论文的表述

可以写：

> SafeConf is evaluated as a post-hoc reliability layer over frozen perturbation predictions. We separate prediction evidence, public-history quality/content, and model-specific error memory, and test each increment under the same perturbation-cold and label-budget contract.

暂时不能写：

- 双历史门控网络已经优于简单基线；
- 公共生物历史在所有研究中稳定提高风险排序；
- SafeConf 首次解决单细胞扰动预测置信度；
- E273 的开发增益已经是 Tahoe 外部确认；
- GPU 占用时间本身证明方法有效。

## 10. 本轮最终判定

**主候选保留：** `magnitude-anchored residual risk learning`，但 V0 先用简单、可审计的风险器。

**暂缓：** record-level Deep Sets encoder、learned gates、evidence dropout、跨模型 Error Memory transfer。

**必须继续：** OOF provenance 审计、Tahoe raw history 合同、PertEMA 同合同对照、E274 error-memory budget、sealed test 的一次性外部评价。

**若 H/E 失败：** 收缩为 prediction evidence + measurement-quality-aware risk audit，保留 abstain/support 作为工程输出，不强行保留“双历史”论文叙事。
