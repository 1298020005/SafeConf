# 文献核查档案（GLM 联网核实，2026-08-17）

方法：GLM 直接检索 + WebFetch 原文页 + 三个并行搜索子任务（共 16+ 组查询、
约 25 篇候选）。标注：**[V]** 已验证原文/官方页；**[P]** 仅摘要/元数据级核实；
**[X]** 未找到或不可验证。

## A. 交接包点名的五篇——全部存在，描述基本准确

| 工作 | 引用 | 核实结论 | 与 SafeConf 的边界 |
|---|---|---|---|
| PRESCRIBE | Cheng, Chi, Zhou, Xin, Xia. *NeurIPS 2025* 主会。[原文页][1] | **[V]** 多元 deep evidential 回归（贝叶斯式，模型内置），同时建模 epistemic（未见基因与训练基因的相似度）与 aleatoric（训练数据质量）；给出逐预测置信度，用于**过滤低可信预测** | 模型内置置信度 vs SafeConf 部署后、模型无关、可加装；PRESCRIBE 无跨 setting 的弃用合同。**禁写"首个"** |
| TxPert | *Nature Biotechnology* 2026, s41587-026-03113-4（PubMed 42067667）。代码 valence-labs/TxPert | **[V]** 多知识图（STRING/GO/…）+ latent transfer，评估未见单扰动/双扰动/跨细胞系；作者内部 PxMap/TxMap 与训练入口未公开 | E201 用公开 STRING-GAT 重训练，论文措辞冻结为"公开 STRING-GAT 重训练审计"，不得写成复现作者最强模型 |
| PerturbMap | arXiv:2607.28090（2026-07-30 上线） | **[V]** "Cross-Context Transfer of Single-Cell Perturbation Responses"：train-only reliability-weighted transport，把 source 响应搬到 recipient 坐标——做**补预测** | SafeConf 是**预测已给出之后**决定信不信/查不查。正文必须主动划清；它晚于 E201 冻结，不回头改协议 |
| HyperMap | bioRxiv 2026.04.23.720505 | **[P]** 元学习框架，把现有 atlas 翻译到新 context 预测扰动响应 | 同上：预测器，非事后审计 |
| Ahlmann-Eltze 等 | *Nature Methods* 2025-08-04 Brief Communication, s41592-025-02772-6 | **[V]** 扰动预测不优于线性基线 | magnitude 类简单基线必须保留为强基线的文献依据 |

## B. 交接包遗漏、GLM 新增的近邻（Related Work 必须补）

1. **ConfPert** — Alwani & Wang, 2026-05, ICML 2026 workshop（OpenReview 论坛
   `1uE9rtYYzp`）。**[P]** 对 8 个扰动预测器（0～6×10⁸ 参数）× 5 数据集做事后、
   模型无关的 **conformal 覆盖保证**。这是"事后+模型无关"路线上最近的已发表工作。
   边界：覆盖保证 ≠ 任务级风险排序/复核分诊；conformal 覆盖在分布漂移下**静默退化**，
   无 fail-closed 弃用。
   ⚠️ OpenReview 全文有浏览器验证墙，GLM 未能打开正文；**投稿前必须人工浏览器复核**
   其是否含 abstention/风险排序实验。同一作者对还有 PerturbCausal（ICML 2026，
   虚拟页 77996：审计 GEARS/CPA/Geneformer/STATE 的效应量膨胀，提出模型无关
   quantile-matching 校准）——该小组正在快速圈地"扰动预测可靠性"问题。
2. **GEARS 的湿实验优先级先例** — Roohani et al., *Nature Biotechnology* 2024
   （s41587-023-01905-6）。**[V]** GEARS 自带 GNN 内部不确定性分数，并曾用该分数
   挑选组合扰动做湿实验验证。"不确定性排序 → 湿实验优先级"的思路**不是 SafeConf
   首创**。边界：单模型内置、单任务、无跨 setting 验证、无弃用机制。
3. **Molina & Zhang** — bioRxiv 2026-07-24（10.64898/2026.07.24.740459，AITHYRA）。
   **[P]** 扰动响应分解（conserved template + perturbation-specific），评估用的
   正是 **K562/RPE1/HepG2/Jurkat leave-one-cell-line-out**——与 E201 同款面板。
   含义：面板正在成为社区标准（利好可比性），同时窗口在收窄（别再拖几个月）。
4. **Schäfer et al., scPertEval** — bioRxiv 2026-07-23（10.64898/2026.07.23.740433）。
   **[P]** 评价协议分类学 + scPertEval 包；文中明确写道扰动预测的**不确定性校准
   "目前很少被评价"**——niche 仍开放的直接文献证据，可作 motivation 引语。
5. **Mao et al., VCBench / in-the-wild** — arXiv:2604.27646（2026-04-30）。
   **[V]** 未见 context/未见扰动/跨数据集下性能显著下降、指标不一致改变模型排序。
   E201 的 motivation 引文（仓库已引用，口径一致）。
6. **GPerturb** — *Nature Communications* 2025-07（PMC12215016）。**[P]** 高斯过程
   建模扰动效应，自带 credible intervals——贝叶斯 UQ 路线的预测器代表。
7. **Medea** — bioRxiv 2026-01（Zitnik 组，10.64898/2026.01.16.696667）。**[P]**
   组学 AI agent，含 "calibrated abstention"（LLM 共识驱动）。最近邻的"弃答"先例，
   但非统计风险信号、非扰动预测。
8. **Risk Advisor** — Lahoti et al., *Patterns* 2023（PMID 36910557）。**[V]** 通用
   ML 的模型无关事后风险估计元学习器——SafeConf 思路的通用 ML 祖先，无生物学、
   无 fail-closed 跨 setting 验证。

## C. 名称核对（防混写）

- NeurIPS 2025 Datasets & Benchmarks 的 **PerturBench**（仓库引用的
  proceedings.neurips.cc 哈希 `8aee5372…`）与 *Nature Methods* 2025 的
  **scPerturBench**（s41592-025-02980-0，27 方法 × 29 数据集）是**两个不同基准**，
  Related Work 分开写。
- "Perturb-map"（Dhawan et al., *Cell* 2022，空间 CRISPR 功能基因组学技术，
  GSE193460）与 "PerturbMap"（arXiv:2607.28090）重名但完全无关，引用时注意大小写
  与指向，别引错。

## D. 刮查（scoop check）总裁决

**SafeConf 的确切空位——(a) 事后 (b) 模型无关 (c) 只用部署时可得信号（分歧/幅度/
元数据）(d) 任务级风险排序供人工复核/湿实验分诊 (e) 跨研究/跨细胞系 setting 上
验证过的 fail-closed 弃用规则——截至 2026-08-17 未被占据。** 组件分别存在
（GEARS 内置不确定性+湿实验、ConfPert 事后覆盖、Medea 弃答、基准文献记录 OOD 崩塌），
但没有人把它们组装成"验证过的弃答式分诊层"。最大威胁：**ConfPert**（发布仅 3 个月，
同为事后+模型无关）。最后 6 个月新出现的 2026-07/08 条目里没有做"不确定性弃答/
风险排序"的直接撞车；唯一 7-8 月的 abstention 命中都是医学 LLM 推理类，非扰动生物学。

[1]: https://papers.nips.cc/paper_files/paper/2025/hash/d6383e7643415842b48a5077a1b09c98-Abstract-Conference.html
