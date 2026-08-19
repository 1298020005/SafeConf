# SafeConf 投稿审核报告（GLM / ZCode）

审核时间：2026-08-17 01:20（Asia/Shanghai）
审核人：GLM（ZCode 智能体），受作者委托独立于 Grok 交接包进行复核
审核基础：已读交接包全文 + 仓库七份优先文档 + E202a 对照表 + E201 冻结协议
（`TARGET_RELEASE_AND_EVALUATION_FREEZE.md`）+ E199/E200 正式评价 CSV；
联网核实了 PRESCRIBE、TxPert、PerturbMap、HyperMap、ConfPert、GEARS、scPerturBench
与期刊分区事实（详见 `01_LITERATURE_VERIFY_20260817.md`、`02_JOURNAL_FACTS_20260817.md`）。
现场核对 E201 队列：**15/16 完成**（hepg2/seed_4 于 2026-08-17 01:08:30 完成，
jurkat/seed_4 于 01:08:36 启动，预计 13:00–14:30 完成），target 真值访问 0 行。

---

## 1. 一句话结论（直接回答 A/B/C/D 四问）

**A. 中科院大类一区（Genome Biology / Nature Methods）？——现在不能投。**
缺四样东西：E201 四背景结论（文章脊柱）、可安装软件包、非饱和数据上的 PRESCRIBE
同任务对照（E202b）、说得清的生物学用途/湿实验。而且 2025–2026 这个方向已经拥挤
（PRESCRIBE、TxPert、PerturbMap、HyperMap、ConfPert），大类一区会把它读成增量方法文。

**B. Briefings in Bioinformatics（小类一区 / JCR Q1）？——是现实目标，但有三个前置条件。**
条件：① E201 按冻结流程评完（预计 08-18）；② 主张收窄为 fail-closed 风险合同；
③ E202a 对照表扩成论文表 1（补 ConfPert、GEARS 两条，见第 7 节）。满足后，
无论 E201 三门 outcome 如何都有对应的诚实写法（见第 5 节分叉），Briefings 可送审。
录用不保证；分区口径注意：Briefings 中科院**大类**是 2 区，"一区"只在**小类**
（数学与计算生物学）和 JCR Q1 意义上成立。

**C. Bioinformatics（JCR Q1、中科院大类 3 区）？——方法上最对口、门槛最友好，但它不是大类一区。**
若学院认中科院大类，投它等于放弃"一区"口径。合理定位：Briefings 被拒后的第一备选，
且其 Applications 轨道**强制**投稿时代码公开——软件包反正要做。

**D. "随便二区"现在有没有一篇诚实的方法文骨架？——骨架存在，正文不存在，"随便"不成立。**
旧稿（registered-family certificate 主线）与当前主线不一致，不能小修补投稿；
E202a 只是表格底稿。骨架 = 预注册的 2,008 任务协议 + 五块已解盲证据（含负结果）+
冻结三门。按本报告第 6 节排期，2–3 周可成稿；任何刊都要过审稿，没有"随便"。

---

## 2. 题目裁决

**推荐主投用交接包 4.1：**

> SafeConf: a fail-closed post-prediction reliability contract for single-cell
> perturbation models

理由：与老师三问、E189/E192/E199/E200/E158 证据、以及"预测后、真值前"的定位全部
兼容；"fail-closed（失败即关闭：信号未验证就明确弃用，不假装有把握）"是本文唯一
站得住的差异化主张。

**若 E201 多数背景过路由门**，可换 4.2 的变体（把 leave-one-cell-line-out
放进副标题），主标题仍不建议出现 "outperform magnitude" 类词。

**GLM 的措辞纪律（比交接包多一条）：** 主标题/摘要不同时使用 "certificate（证书）"
与 "audit（审计）"当主关键词——主贡献是**合同/协议**，family 证书只是零件，
混用会把审稿人引向"证书数学是否新颖"的死胡同（见第 3 节对证书门的说明）。

禁用题目清单照交接包 4.3 全部有效。

---

## 3. 主张是否成立：逐实验核对

| 实验 | 论文里**能写**的句子 | **不能写**的句子 |
|---|---|---|
| E189 小矩阵/行列/双未见 | 随机缺格明显偏容易（关联 0.368–0.412，CI 高于 0）；整列 0.210–0.247 仍正；整行 −0.095～−0.013 无稳定正信号；**双未见关联反号为负（Spearman −0.349～−0.241，E191 效用：分歧 −0.127、幅度 −0.080，均低于随机）**；"setting 决定信号可用性"。⚠️ 注意：E202a 表与交接包把 −0.349～−0.241 误标为 "utility"，实为 Spearman 区间（GLM 已核对 E189/E191 正式报告并全量修正） | "风险分在所有缺失模式下好用" |
| E190 Adamson→Replogle K562 | 跨研究流程与防泄漏合同成立；diversity ρ=0.424 与 magnitude ρ=0.420 **相当**（无独特优势） | "跨研究已经明显更好" |
| E192 Adamson→RPE1 | diversity ρ=0.300，95% CI **[−0.040, 0.580] 跨 0** → 事前双 gate 判 `ABSTAIN`（尽管 20% utility 点估计 0.696 为正）。**这是 fail-closed 设计最有力的自家论据** | "跨细胞系迁移成功" |
| E199 公开 TxPert、K562 内未见基因（263 任务） | diversity 下界 ρ=0.3948 [0.2835, 0.4969]，20% utility 0.2084 [0.1033, 0.3755]；**magnitude ρ=0.0955 [−0.0256, 0.2187] 跨 0、utility 跨 0**。目前最干净的正结果 | "在所有细胞上成立"（只有 K562、一种未见方式） |
| E200 K562 整背景留出（566 任务） | transfer risk ρ=0.4240、utility 0.3648；**magnitude ρ=0.8797 [0.8437, 0.9095]、utility 0.9133——magnitude 显著更强**；另 training_delta_dispersion ρ=0.6639 也强于 transfer risk，应一并报告 | "整背景留出证明 SafeConf 更好" |
| E158/E159 PRESCRIBE 官方分数 | 官方 combined/epistemic 在 Norman P3/P4 严格未见基因上**完全饱和**（面板内仅 1 个不同值），主统计不可估计 → 该 setting 系统应 `ABSTAIN`；写成"竞品在此 setting 退化" | "我们打赢了 PRESCRIBE"；重跑 P3/P4 当确认性成功 |
| E194 family 治理 | 证书严格属于预注册加权 family；复制成员灌水、合成成员可放大 diversity → 治理规则必要 | "分歧越大越好" |
| E198 协议校准 | 12 个评价协议过完整性门、指标事前冻结、事后不换 | "这验证了 SafeConf" |
| E84/E87/E89/E118 chemical | 能算；多处 magnitude 更强 → 只能作边界章节 | gene 与 chemical 同一套成功 |

**核心叙事判断（GLM 独立意见）：** E199 与 E200 放在一起，是本文最有价值的科学
现象——**同一个信号的有效性随 setting 翻转**（K562 内未见基因：magnitude 失效、
diversity 有效；整背景留出：magnitude 压倒一切）。这不是尴尬的矛盾，这正是
fail-closed 合同的动机。论文应把它作为"发现"来写，而不是把两个数都往"我们不差"
方向拧。配套地，建议正式化一个概念：**validation footprint（已验证适用域）**——
每个风险信号配一张"在哪些 setting 已验证/未验证/已退化"的足迹表，E202a 就是雏形。
这能把对照表升级为方法学贡献。

**关于证书门的数学性质（交接包未明说，审稿人会打）：** 对等权重的 4-seed family，
`family_RMS² = centroid_RMSE² + disagreement²` 是 bias–variance 分解的**恒等式**，
不是经验发现。论文中证书门只能作为**防篡改/数值完整性检查**报告（残差在数值容差
内、无违反），绝不能写成科学结论，否则会被一句 "this is trivial" 打掉。

---

## 4. E201 没评完意味着什么

**现在可以写的：**
- 协议本身：4 target（K562/RPE1/HepG2/Jurkat）× 4 seed、80 epochs、每 target 整列
  留出、2,008 个 context–perturbation 任务（其中 1,808 个 ≥30 细胞进主分析）、
  三门判据、按 perturbation condition 整簇 bootstrap 5,000 次、物理盲视图
  （扰动表达矩阵置 0）、双远程封存后才开真值的时序证明。
- 训练账：16/16 完成、真值 0 访问（今日下午起可写）。
- 一切已解盲实验（第 3 节）。

**必须等 STAGE_7 之后才能写的：**
- 任何含 E201 误差/相关/utility 数值的句子；
- "跨细胞系/四背景成立（或不成立）"的总括判断；
- 三门通过与否；
- Abstract 的结果句、结论段、图 5 的结果面板。

**红线重申：** 在 STAGE_6（真值释放）之前，任何人、任何工具不得以任何理由读取
E201 target 扰动表达——包括"先看一眼好不好"。16/16 完成本身不解除盲态。

---

## 5. 若不能发，最小补丁

**P1（零成本，立刻做）：主张收窄为三句话。**
① 预测已给出、真值未知时，哪些信息可以合法打分；② 在小矩阵、行列缺失、跨研究、
公开图模型上，哪些 setting 可以路由、哪些必须 abstain；③ E201 用四细胞 × 四种子
检验整背景留出是否同一套结论。全文禁语：首个/普遍优于 magnitude/gene+chemical
统一成功/任何分区承诺。

**P2（1 天，不占 GPU）：把 E202a 扩成论文表 1。**
在现有六行上加三行：
- **ConfPert（2026-05，ICML 2026 workshop）**：模型无关 conformal 覆盖——区分点：
  覆盖保证 ≠ 任务级风险排序复核分诊，且覆盖在分布漂移下静默退化，无 fail-closed；
- **GEARS（Nat Biotech 2024）内置不确定性 → 湿实验优先级先例**：区分点：单模型内置
  vs 模型无关事后可加装、无跨 setting 验证、无 abstain；
- **PerturbMap（arXiv:2607.28090）**：区分点：它做"补预测"（train-only
  reliability-weighted transport），SafeConf 做"预测已给出后决定信不信、查不查"。
饱和分数一律写 `undefined / ABSTAIN`，不写 ρ=0。

**P3（2–3 天）：最小可安装审计包。** `pip install safeconf-audit` + 一条复现命令
（重算 E199/E200 主数字）。Bioinformatics Applications 轨道强制公开代码；Briefings
事实上同等要求；这也是大类一区"别人能否复现"门槛的地基。

**P4（只在冲中科院大类一区时开）：E202b。** 换一套从未解封的数据，预注册
"PRESCRIBE 官方分数必须先非饱和"，再与 SafeConf、magnitude 同任务比。P3/P4 不能重用。

**明确不做（交接包 7.3 全部有效，另加一条）：** 不为赢 magnitude 调 E200/E201 公式；
不把 chemical 拼进 gene 主表；不新开几十个 E 编号；不在老师确认前开工湿实验；
不提前开真值；**不用自动化在无人核验下推进评估链**（本工作区 STATE.md 的逐步
人工核验就是为此设计）。

**E201 出数后的三叉（与交接包一致，GLM 认可）：**
- 多数背景过路由门 → Briefings 主线正文；
- 部分过 → setting 异质性就是文章，abstain 规则是贡献；
- 全不过 / magnitude 全面主导 → 停"优于幅度"，写"何时必须停用经验排序"的失败
  边界文，仍可投方法刊，不冲 Genome Biology。

---

## 6. 一周可执行任务清单（不破坏盲测）

| 日期 | 任务 | GPU |
|---|---|---|
| 08-17 白天 | E201 jurkat/seed_4 自行跑完（预计 13:00–14:30）；GLM 交付审核报告+文献档案（本目录）；论文骨架中英文 + 图 1–4 初稿（全部用已解盲数据） | 不占用 |
| 08-17 午后 | 16/16 达成 → RUNBOOK STAGE_1（family seal）+ STAGE_2（双远程提交），逐步人工核验 | 仅推理 |
| 08-17 夜 | STAGE_3 十六份零真值预测（seed_1 先封共享文件，再 2–4） | GPU1 |
| 08-18 凌晨 | STAGE_4 风险特征 + general baseline + E200 等价性（≤5e-6）+ STAGE_5 双远程 | CPU/GPU1 |
| 08-18 | STAGE_6 开真值 → STAGE_7 冻结评价（三门、四 target 全报） | CPU |
| 08-18–08-19 | 按三叉写正文；替换论文 E201 占位符 | — |
| 08-19–08-21 | 图定稿（白底 300dpi）、软件包、cover letter、请周老师过目 | — |
| 08-22+ | 视目标决定是否开 E202b / 冲大类一区加码项 | — |

周报/组会口径照交接包第 10 节（不写分区、不写 Spearman）。

---

## 7. GLM 对交接包的独立校正（它漏了什么、说错了什么）

交接包主体判断（不能投大类一区、不能说稳定二区、骨架存在）**GLM 同意**。以下为
修正与补充，均有证据路径：

1. **遗漏最大竞品 ConfPert。** Alwani & Wang，2026-05，ICML 2026 workshop
   （OpenReview `1uE9rtYYzp`）：对 8 个扰动预测器做事后、模型无关的 conformal
   覆盖。这是"事后+模型无关"路线上离 SafeConf 最近的已发表工作，Related Work
   不写它会被审稿人抓。注意：OpenReview 全文有验证墙，GLM 仅核实到摘要级，
   **投稿前必须人工浏览器打开全文复核**它是否含 abstention 实验。同一作者对的
   PerturbCausal（ICML 2026）也在做预测器审计——这个小组正在快速圈地。
2. **遗漏 GEARS 的湿实验优先级先例。** GEARS（Nat Biotech 2024）自带不确定性分数
   并用它挑选组合扰动做湿实验验证。"用不确定性给湿实验排优先级"的思路不是 SafeConf
   首创，差异必须写成"单模型内置、无跨 setting 验证、无弃用机制"vs"模型无关、
   事后、fail-closed"。
3. **同款评估面板已出现。** Molina & Zhang（bioRxiv 2026-07-24，AITHYRA）用与 E201
   完全相同的 K562/RPE1/HepG2/Jurkat leave-one-cell-line-out 面板（作为预测器）。
   利好：面板正在成为社区标准，可比性强；风险：窗口在收窄，**不要再拖几个月**。
4. **niche 仍开放的文献证据。** Schäfer et al.（bioRxiv 2026-07-23，scPertEval）
   明确写道扰动预测的**不确定性校准"目前很少被评价"**——这句可直接当 motivation
   引语，也再次确认截至 2026-07 这个空位还在。
5. **名称核对。** 仓库引用的 NeurIPS 2025 D&B "PerturBench"与 Nature Methods 2025
   "scPerturBench（27 方法 × 29 数据集）"是**两个不同基准**，Related Work 都要写、
   不要混称（交接包把它们当成一个了）。
6. **交接包说对、GLM 已逐一证实的：** PRESCRIBE = NeurIPS 2025 模型内置 evidential
   回归（支持过滤低可信预测）；TxPert = Nat Biotech 2026（公开仓库缺作者内部训练
   入口与 PxMap/TxMap，本项目只能称"公开 STRING-GAT 重训练审计"）；PerturbMap =
   arXiv:2607.28090（2026-07-30，train-only reliability-weighted transport）；
   HyperMap = bioRxiv 2026-04；中科院分区表 2026 年起停更、2025 升级版为最后一版
   （fenqubiao.com 官方声明原文核实）。
7. **进度更新。** 交接包写 14/16；GLM 现场核对为 **15/16**（hepg2/seed_4 已完成，
   jurkat/seed_4 运行中），预计今日午后 16/16。

---

## 8. 分区事实（GLM 联网核实版，2026-08-17）

| 期刊 | IF(2025) | JCR | 中科院 2025 升级版（最后一版） | 备注 |
|---|---|---|---|---|
| Nature Methods | 28.3 | Q1 | 大类生物学 **1 区 Top** | 2025 已发线性基线 Brief Comm. 与 27×29 基准 |
| Genome Biology | 9.2 | Q1 | 大类生物学 **1 区** | 方法文要软件+宽验证+生物用途 |
| Cell Systems | 7.5 | Q1 | 大类生物学 **1 区** | 偏生物学洞见 |
| Briefings in Bioinformatics | 7.3 | Q1 | 大类生物学 **2 区**；小类数计 **1 区** | "一区"仅在小类/JCR 口径成立 |
| Bioinformatics | 5.5 | Q1 | 大类生物学 **3 区** | Applications 轨道强制代码公开 |
| NAR Genomics & Bioinformatics | 3.0 | Q2 | 大类生物学 **4 区** | 保底 |
| Bioinformatics Advances | 2.6 | Q2 | 大类生物学 **4 区** | 保底 |

- 中科院文献情报中心官方声明 2026 年起停更分区表；高校主流沿用 2025 升级版；
  第三方"新锐分区"多数 985/211 暂不认可。**行动项：向学院书面确认认哪一年、
  认大类还是小类**——这决定 Briefings 算不算"一区"。
- Patterns（ESCI，无中科院分区）谨慎；Communications Biology 桌拒率高。

---

## 附录：GLM 引用的证据路径

- 交接包：`~/.zcode/tmp/prompt-attachments/.../01-GLM-_SafeConf-_20260817.md`
- E202a 总表：`docs/实验结果/E202_q1_blocker_closure_20260815/tables/E202A_SETTING_COMPARISON.csv`
- E199 正式评价：`docs/实验结果/E199_txpert_public_k562_20260802/formal_evaluation/tables/E199_RISK_ASSOCIATIONS.csv`、`E199_REVIEW_UTILITY.csv`
- E200 正式评价：`docs/实验结果/E200_txpert_cross_context_k562_20260802/formal_evaluation/tables/E200_RISK_ASSOCIATIONS.csv`、`E200_REVIEW_UTILITY.csv`
- E201 冻结协议：`docs/实验结果/E201_txpert_multitarget_retraining_20260802/TARGET_RELEASE_AND_EVALUATION_FREEZE.md`
- E201 队列现场：`/home/yyf/data/txpert_official_20260802/e201/formal/E201_QUEUE_SUPERVISOR.log`（tail）
- 事实总账：`START_HERE_FOR_AGENTS.md` §2；`docs/实验结果/GATE_STATUS_20260729.md`
- 文献与期刊核查：本目录 `01_LITERATURE_VERIFY_20260817.md`、`02_JOURNAL_FACTS_20260817.md`

（完。下一步：按 `04_E201_RUNBOOK.md` 与 `STATE.md` 推进 E201 评估链。）
