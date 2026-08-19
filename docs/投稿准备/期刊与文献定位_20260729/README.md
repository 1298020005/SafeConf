# SafeConf 期刊、近邻工作与实验修正

审计日期：2026-07-29

## 当前判断

SafeConf 有明确的方法学对象，但现在不宜写成“通用置信度分数”，也不能写成
“分歧普遍优于 magnitude”。2025–2026 年已经出现三个会直接影响审稿的变化：

1. PRESCRIBE 已经把单细胞扰动预测的实例级不确定性做成学习式方法；
2. Systema、PerturBench 和大规模泛化基准表明，简单基线经常不输复杂模型，单一
   RMSE 还可能掩盖系统变化、方向错误或预测塌缩；
3. selective conformal risk control 已经成为独立统计方向，SafeConf 若没有对应
   条件和证明，不能把 `ABSTAIN` 写成正式的选择性风险保证。

SafeConf 还能站住的贡献是：

> 对满足同一冻结输出合同的预注册加权单细胞扰动预测家族，提供只依赖预测输出的
> 事后审计。在目标真值不可见时，输出具有严格数学含义的家族平均误差和最坏成员
> 误差下界；经验排序只有在相应 setting 的冻结外部验证条件通过时才开启，否则
> 明确返回 `ABSTAIN`。

这个定位与 PRESCRIBE 的学习式单模型置信度不同，也比“再训练一个风险回归器”更
容易解释。数学零件来自经典平方误差分解、三角不等式和 conformal calibration，
不能包装成全新定理；新意落在问题形式化、证书组合、严格的数据访问合同和单细胞
部署验证。

## “一区、二区”采用什么口径

### 可以公开核验的口径

本文件优先使用期刊官网列出的 2025 Journal Impact Factor 和 JCR 类别排名。表中
Q1/Q2 是按官网公开的“排名/类别总数”推算；并列排名和 JCR 实际计算规则可能造成
边界差异，最终以 Web of Science/JCR 或学校图书馆核验为准。

### 不能混用的口径

- JCR Q1/Q2、SCImago Q1/Q2 和中科院一区/二区是三套不同体系；
- SCImago 排名不能写成 JCR 分区；
- 中国科学院文献情报中心已经声明，自 2026 年起不再更新和发布期刊分区表，其他
  机构后续发布的表与其无关；
- 若河南大学毕业或奖学金规则仍采用“中科院分区”，应由学校图书馆核验最后一版
  官方表及学校认定年份，不能用 LetPub、梅斯等第三方页面代替。

## 候选期刊

### 第一组：方法增强完成后冲刺

| 期刊 | 当前可核验信息 | 与 SafeConf 的匹配 | 直接投稿风险 |
|---|---|---|---|
| **Bioinformatics** | 2025 IF 5.5；Mathematical & Computational Biology 7/67，JCR Q1 | 新计算生物学方法、真实数据、独立测试集，主题最匹配 | PRESCRIBE 正面对照、Systema/PerturBench 指标和更宽预测器家族尚未完全闭合，当前容易被判为增量方法 |
| **Cell Reports Methods** | 官网强调 robust、reproducible 的实验与计算方法；本轮未从可公开官方页取得 JCR 类别排名 | fail-closed、可复现证书协议与方法刊气质匹配 | 不能在未核验分区时先称“一区”；需要更完整的资源与实用性证据 |
| **PLOS Computational Biology** | 官网接受 Research、Methods、Software，明确包含 AI/ML 和组学方法 | 适合把问题上升为“科学预测系统何时复核或拒答” | 要求广泛用途、显著方法或生物学洞见；只做单细胞模型审计不够 |
| **Briefings in Bioinformatics** | 2025 IF 7.3；4/85、6/67，JCR Q1 | 范围含新方法、single-cell、AI/ML | 文章类型和编辑偏好存在不确定性，投前应先发 presubmission inquiry |

Bioinformatics 只能作为完成直接竞品、统一评测、家族扩展和新外部解盲后的冲刺
目标；其官网要求真实生物数据上的 SOTA 比较和显著概念推进。PLOS Computational
Biology 只有在决策效用或生物学理解具有更广泛意义后才合适。Briefings in
Bioinformatics 官网对原创研究的文章类型说明存在口径差异，不能仅凭影响因子直接
投，应先做 presubmission inquiry。

### 第二组：可核验的现实 Q2 路径

| 期刊 | 当前可核验信息 | 与 SafeConf 的匹配 | 判断 |
|---|---|---|---|
| **NAR Genomics and Bioinformatics** | 2025 IF 3.0；22/67、80/192，按排名推算 Q2 | 明确接收 single-cell、AI、benchmark 和大规模方法；要求最高水平原创性/实用性、足量 SOTA、可量化优势、开源与 FAIR | 补齐阻断项后的主要候选；当前尚不具备低风险送审条件 |
| **Bioinformatics Advances** | 2025 IF 2.6；29/67，按排名推算 Q2 | 范围覆盖算法、统计、软件和细胞层面计算生物学；官网同样拒绝简单应用或增量改进 | 可作第二选择；仍须完成真实数据 SOTA 对照和实质性方法论论证 |

如果目标是先形成一份证据严谨、范围匹配的投稿，顺序建议为：

```text
增强完成后：
Bioinformatics
    ↓ 若方法广度或显著性不足
NAR Genomics and Bioinformatics
    ↓
Bioinformatics Advances
```

这是一条基于 scope 和当前证据的投稿路径，不是录用概率承诺。

### 有条件考虑，不列入当前主序列

- **npj Systems Biology and Applications**：官网明确包含 computational modeling、
  single-cell systems biology；2025 IF 4.4，但官网未列 JCR 类别排名。本项目若增加
  系统生物学机制解释可考虑。
- **Communications Biology**：接受创新计算方法，但要求给生物学领域带来新的认识。
  当前证书审计的生物学发现不足。
- **Patterns**：需要更广泛的数据科学贡献。除非把软件扩展到单细胞以外多个科学
  预测任务，否则 desk risk 高。
- **Genome Biology**：计算方法在范围内，但没有显著生物发现或大范围基准时风险很高。
- **GigaScience**：开放、FAIR、可复现和大规模数据是核心标准。SafeConf 的哈希、
  预注册和最小复现与其气质相符，但当前还缺可复用软件包与足够宽的大规模基准，
  故只作条件候选。
- **BMC Bioinformatics / BMC Genomics / CSBJ**：可以作为后续路径；若学校对分区有
  硬性要求，先由图书馆核验学校采用的正式口径。

## 2024–2026 近邻工作

### 单细胞扰动预测与评测

| 工作 | 它解决的问题 | 对 SafeConf 的影响 |
|---|---|---|
| **PRESCRIBE，NeurIPS 2025** | 多变量深度证据回归，同时估计模型与数据不确定性；用置信度过滤低可靠预测 | 最直接竞品。SafeConf 不能再声称“首个实例级可靠性方法”；差异只能写成不改造现有预测器、仅依赖满足同一冻结合同的预测输出，并给声明 family 的预真值确定性下界 |
| **Systema，Nature Biotechnology 2025** | 去除或缓解系统变化的干扰，评价 perturbation-specific effect | 只报告原始 RMSE 不够；必须报告参考空间、方向型或 perturbation-specific 指标，并保留简单 perturbed-mean / matching-mean 基线 |
| **27 方法 × 29 数据集泛化基准，Nature Methods 2025/2026** | 在 6 个指标和多种未见背景、未见扰动 setting 下比较 27 种方法 | 审稿人会要求统一 benchmark 和 context/perturbation generalization，不能用少量自定义 split 代表普适性 |
| **Ahlmann-Eltze 等，Nature Methods 2025** | 比较基础模型、深度模型与简单线性基线 | 强制保留 no-change、linear、mean 等简单基线；模型参数量不再是说服力 |
| **PerturBench，NeurIPS 2025 Datasets & Benchmarks** | 统一遗传和化学扰动任务，讨论 RMSE、rank 和模型塌缩 | 需要多指标并列，不能只用一个平均误差 |
| **PertEval-scFM，ICML 2025** | 基础模型嵌入的 zero-shot 扰动预测评价 | 进一步说明基础模型未必超过 PCA 等简单表示 |
| **scPertEval，bioRxiv 2026-07** | 把评价拆成 representation、transform、metric、reporting 四层 | Methods 中需要明确每层对象，避免“同名 Pearson/RMSE 实际算的不是同一空间” |
| **PertAdapt，Bioinformatics 2026** | 多基础模型、GEARS 与简单基线的统一适配和比较 | 可作为扩展 predictor family 和统一输出合同的候选入口 |

### 不确定性、拒答与证书

| 工作 | 它解决的问题 | SafeConf 必须划清的边界 |
|---|---|---|
| **TISSUE，Nature Methods 2024** | 空间单细胞数据的 conformal uncertainty 和下游分析 | 说明单细胞领域已有 conformal UQ；SafeConf 的差异是扰动预测家族误差对象 |
| **GEARS uncertainty / ensemble** | 原生模型包含逐基因 log-variance 不确定性头，也可用同架构多次训练 | 当前正式 family 运行使用 `uncertainty=False`，必须补原生 GEARS-UQ 同任务对照 |
| **CPA uncertainty** | 测试组合到训练扰动和协变量嵌入的距离型 uncertainty score | 不能只把 CPA 当预测器；该距离是直接经验风险基线 |
| **GPerturb，Nature Communications 2025** | 高斯过程给出基因扰动效应及其后验不确定性 | 不是任意冻结家族证书，但应纳入相关工作；条件允许时作模型相关 UQ 对照 |
| **CAP，JMLR 2025** | 在线选择后的 conformal coverage / FCR 控制 | “先选择再报告”会改变统计条件；SafeConf 当前经验排序不能自动继承这种保证 |
| **Intervention selective conformal，arXiv 2026** | 在 intervention 场景及 Replogle K562 上研究选择后的有效覆盖 | 阻断“首次把 conformal 用于单细胞扰动”的表述；尚未同行评议 |
| **Selective Conformal Risk Control，2025**；**SCoRE，2026** | 对被系统接受的样本控制风险 | SafeConf 的 `ABSTAIN` 目前是冻结经验 gate，不是已经证明的 selective risk control |

## 对方法表述的修正

### 可以写

1. 注册预测家族在指定 Hilbert 几何中的 family RMS error 和 worst-member error
   存在真值无关的确定性下界；
2. 方法可附着在冻结的异构预测器输出上，不要求为每个预测器新增不确定性头；
3. 参考质心上界只能在明确的校准假设和 family-shift 代价下搬移；
4. 经验复核排序具有 setting dependence，未验证 setting 和 double-unseen setting
   可以正式返回 `ABSTAIN`；
5. 输入哈希、预测先冻结、真值后解封和失败保留构成可审计协议。

### 不能写

1. “首次提出单细胞扰动预测不确定性”；
2. “首次发现模型分歧反映误差”；
3. “提出了全新的平方误差分解或 conformal prediction”；
4. “SafeConf 普遍优于 magnitude、PRESCRIBE 或简单基线”；
5. “证书给出单个预测正确的概率”；
6. “只要下界零违例，风险排序就被验证”；
7. “开真值后的指标扩展是新的前瞻确认”；
8. “效应向量 cosine/Pearson 等同于 Systema exact”。

## 投稿前阻断项

### 1. PRESCRIBE 同协议正面对照

仓库已有 PRESCRIBE 复现和失败记录，但目前跨实验、跨预测器的结果不能组成公平
head-to-head。因为 PRESCRIBE、原生 GEARS-UQ 和 CPA uncertainty 与各自预测器
绑定，不能强行要求它们使用 SafeConf 的同一预测输出。正式比较拆成两条轨道：

1. SafeConf、普通 disagreement、magnitude、source magnitude 使用同一冻结
   prediction family 和同一 A0 error target；
2. PRESCRIBE、GEARS-UQ、CPA distance 使用相同 dataset、split、task、gene、
   metric 和复核预算，分别评价各自的 `predictor–uncertainty` 对，并明确它们的
   error target 不同。

两条轨道统一报告：

- PRESCRIBE pseudo E-distance、GEARS 原生 uncertainty、CPA distance；
- 普通 disagreement、predicted magnitude、source magnitude；
- SafeConf 的严格下界、family 治理和 setting gate；
- risk–coverage、AURC、固定 10%/20%/30% 复核收益及基因簇置信区间；
- 训练、推理、显存、内存和是否需要重训。

不同 error target 的单个 AURC 不能直接推导“谁全面更好”。重点是检验 SafeConf
能否提供竞品没有的严格 family-error 语义，并在相同任务合同和预算下形成可复现的
互补价值。

### 2. Systema / PerturBench / scPertEval 指标合同

E193 先检查同一 Hilbert 证书能否扩展到 effect-vector cosine 和 Pearson。完整投稿
仍需要：

- Systema exact expression-state/reference-space 指标；
- perturbed mean、matching mean、no-change、PCA/linear；
- rank 或矩阵层面的塌缩指标；
- 每个 representation、transform、metric 和 aggregation 的明确字段；
- seen、unseen perturbation、unseen context、double unseen、cross-study 分层。

### 3. 扩展注册预测家族

当前主要 family 是三种子 scGPT + 三种子 GEARS。要支撑“异构黑盒预测器家族”，
至少再加入两类不同机制的预测输出，例如：

- CPA 或 CellOT；
- PCA/linear 或 perturbed-mean；
- PRESCRIBE 或 PertAdapt 统一输出中的其他基础模型。

证书恒等式对这些向量仍成立并不构成充分实验结果；还要报告 tightness、排序信号和
退化情况。

### 4. 再做一次真正前瞻的外部解盲

E192 是有效的锁定目标，但 E193 使用的是已经打开的 E190/E192 真值，只能叫
post-truth robustness。若要把方向型指标写进确认性主张，需要新的目标块：

1. 先冻结数据来源、任务、基因面板和评价空间；
2. 生成六成员或扩展家族预测并提交 SHA-256；
3. 冻结风险量、预算、置信区间单位和 PASS/ABSTAIN gate；
4. 推送远程后一次性读取目标真值；
5. 不因结果不理想删除数据集或改门槛。

## 当前实验分工

| 证据 | 回答的问题 | 边界 |
|---|---|---|
| E176–E183 | 多研究双侧 family certificate 与覆盖 | 部分 setting 失败必须保留 |
| E189 | random、row、column、double missing 和小训练子矩阵 | double unseen 中经验排序可为负 |
| E190 | Adamson K562 → Replogle K562 直接迁移 | family 与简单基线差异很小 |
| E191 | 固定复核预算是否有用 | diversity 与 magnitude 总体互有胜负 |
| E192 | Adamson K562 → Replogle RPE1 锁定确认 | 预算收益为正，但相关区间跨 0，按冻结 gate 为 `ABSTAIN` |
| E193 | RMSE、effect cosine、effect Pearson 三种 Hilbert 几何；867 个任务、2,601 个几何任务实例 | 0 个 family/worst 下界违例，恒等式残差 \(6.66\times10^{-16}\)；开真值后分析，不是 Systema exact，也不改变 E192 gate |
| E194 | 55/50/50 个 family 构成场景；重复、失衡、遗漏和合成攻击 | 310 个场景、134,385 条逐任务记录、492/492 治理检查通过；证明证书严格依赖成员与权重，不增加外部泛化证据 |

E193 的排序结果进一步说明了为什么必须分开“证书成立”和“排序可用”：

- E190 K562 的 cosine diversity 与方向误差相关：
  \(\rho=0.568\)，基因簇 95% CI [0.278, 0.783]；20% 复核 utility=0.782；
- E190 Pearson 的相关区间跨 0，20% utility=-0.026；同一 setting 中
  source-to-family-centroid distance 的 utility=0.634；
- E192 RPE1 的 cosine 相关接近 0，Pearson 为负：
  \(\rho=-0.210\)，95% CI [-0.507, 0.039]；两种方向几何的 20% diversity
  utility 区间均跨 0；
- 数学下界在三种几何中始终成立，经验排序却随 target cell line 和 metric 改变。

因此，多几何扩展加强了 `registered-family certificate`，没有得到一个可跨环境通用
的方向型风险路由器。后续新外部解盲必须把 metric-specific `ACTIVATE/ABSTAIN`
写进冻结协议。

E194 又关闭了“复制成员或加入异常成员后仍把证书当成同一对象”的漏洞：

- 六个真实预测数组哈希唯一，A0 固定为 scGPT/GEARS 各占 1/2；
- duplicate-governed 场景逐任务恢复 A0，数值差不超过 \(2.22\times10^{-16}\)；
- K562 的 A0 diversity–固定 A0 family error 为 ρ=0.424，而 scGPT-only /
  GEARS-only 分别为 -0.081 / 0.156，主要信号来自架构间差异；
- absolute RMSE 的对称攻击在质心误差不变时使 mean diversity 增加
  286%–300%，即达到 A0 的 3.86–4.00 倍；其对“自身 family error”的高相关是
  构造结果，对固定 A0 error 的相关基本不变；
- 因此新增模型必须形成新版本 target family，不能把更多成员或更大 disagreement
  当作自动增强的证据。

E194 是已解封真值上的 post-truth stress test，不提高 E192 gate，不扩展真实预测器
机制，也不能替代新的前瞻外部验证。

## 决策

现阶段不把项目标成“稳定二区”。更准确的状态是：

- 方法对象和确定性证书已经形成；
- 周老师追问的难度矩阵、跨研究、跨细胞系和有限复核预算已有实证回答；
- 2025–2026 新文献带来的三项阻断问题仍要闭合：PRESCRIBE 同协议比较、完整
  perturbation-specific/multi-metric 合同、更宽的预测器家族；
- 完成这些项目后，NAR Genomics and Bioinformatics / Bioinformatics Advances 才
  具备正式投稿的基本证据条件；若直接竞品比较、外部解盲、软件 FAIR、统计完整性和
  方法广度均支持，再以 Bioinformatics 为冲刺目标。任何一条都不是录用承诺。

## 官方与原始来源

### 期刊

- Bioinformatics about：<https://academic.oup.com/Bioinformatics/pages/About>
- Bioinformatics scope：<https://academic.oup.com/bioinformatics/pages/scope_guidelines>
- NAR Genomics and Bioinformatics about：<https://academic.oup.com/nargab/pages/about>
- NAR Genomics and Bioinformatics scope：<https://academic.oup.com/nargab/pages/scope_and_criteria>
- Bioinformatics Advances about：<https://academic.oup.com/bioinformaticsadvances/pages/about>
- PLOS Computational Biology journal information：<https://journals.plos.org/ploscompbiol/s/journal-information>
- Briefings in Bioinformatics about：<https://academic.oup.com/bib/pages/About>
- npj Systems Biology and Applications scope：<https://www.nature.com/npjsba/aims>
- npj Systems Biology and Applications metrics：<https://www.nature.com/npjsba/journal-impact>
- GigaScience：<https://academic.oup.com/gigascience>
- GigaScience reporting standards：
  <https://academic.oup.com/gigascience/pages/editorial_policies_and_reporting_standards>
- 中科院文献情报中心停止更新期刊分区表声明：
  <https://cssar.cas.cn/library/dtxx/202604/t20260409_8183275.html>

### 论文与基准

- PRESCRIBE（NeurIPS 2025 正式论文）：
  <https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6383e7643415842b48a5077a1b09c98-Abstract-Conference.html>
- Systema：<https://www.nature.com/articles/s41587-025-02777-8>
- 27 方法 × 29 数据集泛化基准：
  <https://www.nature.com/articles/s41592-025-02980-0>
- 深度模型与线性基线：
  <https://www.nature.com/articles/s41592-025-02772-6>
- PerturBench：
  <https://proceedings.neurips.cc/paper_files/paper/2025/hash/8aee537279a66ced96319dfca3c00002-Abstract-Datasets_and_Benchmarks_Track.html>
- PertEval-scFM：<https://openreview.net/forum?id=t04D9bkKUq>
- TISSUE：<https://pubmed.ncbi.nlm.nih.gov/38347138/>
- GEARS：<https://www.nature.com/articles/s41587-023-01905-6>
- CPA：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/>
- GPerturb：<https://www.nature.com/articles/s41467-025-61165-7>
- CAP：<https://jmlr.org/beta/papers/v26/24-0452.html>
- Intervention selective conformal：<https://arxiv.org/abs/2603.02204>
- Selective Conformal Risk Control：<https://arxiv.org/abs/2512.12844>
- SCoRE：<https://arxiv.org/abs/2603.24704>
- scPertEval：<https://www.biorxiv.org/content/10.64898/2026.07.23.740433v1>
- PertAdapt：<https://academic.oup.com/bioinformatics/article/42/Supplement_1/btag307/8726320>
