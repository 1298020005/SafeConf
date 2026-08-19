# SafeConf · Codex 互审包（对抗审核用）

**作者**：Grok（2026-07-12）  
**用途**：Codex / Claude / 其他 agent **互相挑刺**，禁止互相吹捧。  
**用户硬目标**：按**周老师问题与布置**推进；最终**稳定二区为底线**，一区为冲刺。  
**禁止**：课程论文线；把下载当实验；无脑加 panel；混轻量器与真深度模型主表。

阅读顺序：§0 → §1 → §2 → §3 → §4 → §5 → §6 → §7。  
每条结论必须能指到**路径或实验编号**。无路径 = 无效。

---

## 0. 一句话 idea（锁死，审核用）

**允许的窄 claim（二区底线叙事）：**

> 在单细胞扰动预测中，预测器先给出效应向量；SafeConf 是 **post-hoc 任务风险分诊层**：在**不使用 holdout 真值**的前提下，用可部署线索（含历史支持、背景相似度、**预测幅度**、**多模型分歧**等）对 `(context, perturbation)` 任务排序，使高分任务对应更高真实预测误差，从而优先复核/湿实验。评价必须写清 `predictor_name + error_metric + split_setting`。

**允许的升级 claim（E74 方向，需单独过关）：**

> 不同模型家族预测分歧提供 **pair mean/max 误差的无真值下界**，并在严格未见基因任务上相对固定预测幅度聚合有排序增益；**不能**定位单模型谁错，**不能**当概率校准置信度。

**禁止 claim（一票否决）：**

- SafeConf 普遍优于所有预测器 / 所有数据  
- 固定四项等权分数跨域全面碾压 magnitude  
- seed 分歧 = 可靠性  
- 下载 Tahoe = 完成大规模验证  
- 与 PRESCRIBE 同协议已赢（未跑对照前）

---

## 1. 周老师原话 → 必须交付物（验收表）

来源：`~/.codex/attachments/3d36bc69-c573-44c2-821b-6795c42ac3fc/pasted-text.txt`  
拆解：`workspace/group_meeting_20260709_MAINLINE_WHITE/周老师聊天记录_要求拆解与实验设计_20260709.md`

| ID | 老师要什么 | 必须交付 | 当前状态 | Codex 自检 |
|---|---|---|---|---|
| Z1 | 分数跟什么错误 correlate | 每张主表有 `error_metric` + 与 score 的 Spearman/CI | 双模型线较好；旧 7 主表较模糊 | □ |
| Z2 | 实际错误用的是哪个模型 | 每行有 `predictor_name` | E60–E72 有；旧表/轻量表不足 | □ |
| Z3 | disagreement 用了多模型 → 不是 per-model | 正文写清：task / pair-risk，不写“该选哪个模型” | E74 写清了；对外叙事常混 | □ |
| Z4 | 第四项幅度：没真值怎么算 | `predicted_magnitude` vs `true_magnitude` 拆分；真值只诊断 | 原则有；旧表需审计 | □ |
| Z5 | 输入必须是“没见过 holdout 真值”能算的 | `E33` 类 provenance 表；deployable 标记 | 部分有 | □ |
| Z6 | random pair 太简单 | Setting0 仅作对照，不写“最真实” | 仍常当主结果 | □ |
| Z7 | **小矩阵** | 25/50/75% 可见 + 真预测器误差 | smoke/轻量为主 → **未过** | □ |
| Z8 | **整行+整列 holdout** | 新 context + 新 perturbation 分开报 | 整列未见基因较实；整行弱 → **半过** | □ |
| Z9 | **跨数据集** | 源上校准/冻结，目标只评；同族优先 | 轻量多；真模型 E69 弱 → **未过** | □ |
| Z10 | gene + chemical 都看 | 化学用 CPA 类，不硬套 GEARS/scGPT | 数据有，真模型合同缺 → **未过** | □ |
| Z11 | 三 setting 过了 ≈ 小文章 | 三关齐才允许“可写小文章”措辞 | **未齐** | □ |

**老师原话门槛：**  
> 如果这三个（小矩阵 / 行列表 holdout / 跨数据集）都解决了，感觉写一个小文章应该可以了。

**审核规则：** 任一 Z7–Z9 为未过 → 禁止对外说“已按老师做完 / 可投”。

---

## 2. Codex 当前在想什么（事实画像）

### 2.1 已形成的正确想法

1. 停止疯狂调冻结权重（test-set tuning 风险）。  
2. 误差对象必须绑定具体预测器。  
3. true magnitude 不能进可部署分数。  
4. seed 分歧多次阴性 → 正确放弃当主信号。  
5. 跨模型家族分歧 > 同模型 seed / 廉价 baseline（E60/E64/E65–E72）。  
6. E74 理论下界 + 三数据集分层：科学上最硬的一块。  
7. 阴性要留：McFarland、跨域弱、magnitude 常很强。

### 2.2 错误 / 危险想法

| # | Codex 倾向 | 为什么错 |
|---|---|---|
| C1 | 堆 panel2/3 复现 = 推进发表 | 老师三关未齐时，复现是次优先 |
| C2 | 下载多 = 证据多 | Tahoe/OP 无适配 predictor 不算主证据 |
| C3 | 轻量跨域高 ρ 可进主表 | 与 Z2 冲突：误差模型不是 GEARS/scGPT |
| C4 | E74 可替代四特征 SafeConf 叙事 | 可升级 claim，但必须跟老师说清换题 |
| C5 | pair-risk = 单模型置信度 | 老师已说“不是针对每个 model” |
| C6 | 7 主表 + E74 混写同一方法 | 审稿人打穿协议不一致 |
| C7 | “一定能二区/一区”驱动实验 | 导致结果导向调参（E73 已自觉停，勿复发） |

### 2.3 当前证据分层（审核时不得混层）

| 层 | 内容 | 能支撑什么 | 不能支撑什么 |
|---|---|---|---|
| L0 协议/工程 | PredictionRecord、E33 provenance | 可复现、防泄漏口径 | 发表 claim |
| L1 旧 7 主表 | Cui/Frangieh/Lara×2/sciplex3/Santinha/McFarland + frozen v0.2 | task-risk 有信号、McFarland 边界 | 真深度模型可靠性 |
| L2 轻量难 setting | E34–E57 小矩阵/行列表/跨域 smoke | setting **可行性** | 与 L3 同权主结果 |
| L3 真双模型 | Adamson/Norman/Frangieh × GEARS+scGPT；E74 | pair-risk / 跨家族分歧 | 化学、跨域稳定、三 setting 全过 |
| L4 未完成 | CPA/chemCPA、PRESCRIBE 对照、统一六类 split 主表 | — | 任何“已完成”措辞 |

---

## 3. 按周老师要求的实验设计（执行规格）

**总原则：** 同一字段 schema；真值只用于最后 error；先 gene 同合同，再 chem 适配模型。

### 3.1 统一结果 schema（每行任务）

```text
split_setting          # random_pair | submatrix_p | row_holdout | col_holdout | cross_dataset
coverage_or_holdout_id
dataset_name
perturbation_family    # gene | chemical
predictor_name         # GEARS | scGPT | CPA | ref_A | ref_B | ensemble_...
gene_panel_id
task_key               # context|perturbation
score__predicted_magnitude
score__support
score__context_sim
score__model_disagreement
score__safeconf_frozen   # 若仍报
score__pair_risk         # disagreement 或校准后
error__rmse
true_magnitude_oracle    # 仅诊断，不进 deployable score
fold_id / seed
```

### 3.2 Setting 队列（优先级 = 老师顺序）

| 优先级 | 实验 | 输入 | 输出 | 通过标准（写进报告） |
|---:|---|---|---|---|
| P0 | **S0-audit** 输入与误差来源 | 现有分数代码 | provenance CSV + 泄漏清单 | deployable 分数 0 处使用 holdout true effect |
| P0 | **S1-submatrix** 小矩阵 | 同数据集 context×pert 矩阵；真预测器或冻结预测缓存 | 25/50/75% × ≥3 seed 汇总 | 至少 2 个 gene 数据集：score–error Spearman CI 下界>0 **或** 诚实写“覆盖下降后失效” |
| P0 | **S2-row-col** | 同上 | 整行(context) / 整列(pert) 分表 | 行、列分开；列 holdout 上 support=0 规则写死 |
| P0 | **S3-cross** | 源数据集全部可见，目标只评 | A→B、B→A 矩阵 | 同族 gene→gene 或 chem→chem；增量相对 predicted magnitude；CI 穿 0 必须报 |
| P1 | **S4-chem** | sciplex3 / OpenProblems / Tahoe 子集 | CPA 或 chemCPA（或声明不可比） | 不与 GEARS 主表混排 |
| P1 | **S5-ablation** | 各 setting | magnitude / support / context / disagree / 组合 | 禁止只报最好特征 |
| P2 | **S6-pair-risk-rep** | 不重叠 panel | E74 复现 | 多 panel 合并 CI 仍>0 才写“稳定超过 magnitude” |
| P2 | **S7-competitor** | 同任务合同 | PRESCRIBE 或明确不可比表 | 未跑不得写赢 |
| P3 | 写作 | 过关 setting 表 | 小文章骨架 | 仅用 L3+过关 L2 升级版 |

**明确后置：** 无限 E75–E78 式 panel，除非 S1–S3 主表已出。

### 3.3 成功/失败判据（反结果导向）

- 允许失败：某 setting 上仅 magnitude 有效 → 写边界，不调公式刷正。  
- 禁止：在测试任务上看完 error 再改权重/特征。  
- 预注册：split 清单、predictor 版本、基因面板在跑预测前冻结。

---

## 4. 期刊定位（全网对照，非保证录用）

分区因年/学科/中科院与 JCR 口径不同；下表是**匹配度**，不是“一定中”。

### 4.1 竞品与领域门槛（必须引用/对照）

| 工作 | 要点 | 对 SafeConf 压力 |
|---|---|---|
| scPerturBench (Nat Methods 2025) | 多方法多数据泛化 benchmark | 单 split / 单数据不够 |
| Systema 等 | 效应大小、简单基线、扰动特异性 | 必须 magnitude + 特异检查 |
| PRESCRIBE (NeurIPS 2025) | 预测时联合估计 data/model uncertainty | **集成式 UQ**；SafeConf 必须强调 **post-hoc / 异构预测器 / 不改原模型** |
| OpenProblems / OP3 | 未见扰动与细胞类型 | 化学+未见 setting 有现成赛道 |
| risk–coverage / selective prediction 文献 | 拒绝高风险后误差下降 | 二区叙事应有 RC/AURC，不只 Spearman |

### 4.2 稳定二区：优先匹配（方法/协议/可复现）

| 期刊 | 为何匹配 | 要补什么才敢投 |
|---|---|---|
| **Bioinformatics (OUP)** | 算法+评估协议友好 | Z7–Z9 + 真模型误差绑定 + 软件/复现 |
| **BMC Bioinformatics** | 工具/流程/评估 | 完整 pipeline + 阴性边界 |
| **Briefings in Bioinformatics** | 方法+领域综述型长文也可 | 文献地图 + 与 PRESCRIBE 区分 |
| **GigaScience** | 大数据+可复现+协议 | Tahoe/OP 真合同或降为 data note 辅文 |
| **PLOS Computational Biology** | 计算生物学方法 | 生物解释 + 失败机制分组 |

**二区底线投稿包（最少）：**  
老师三 setting 用**同一真预测器合同**过关或诚实边界 + L3 双模型证据 + magnitude 消融 + 可复现代码 + 禁止 claim 清单清零。

### 4.3 一区冲刺（更难，非当前时态）

| 期刊 | 匹配点 | 额外门槛 |
|---|---|---|
| **Genome Biology** | 基因组方法长文 | 强方法增量 + 跨域 + 双模态 |
| **Nucleic Acids Research** (methods/web) | 工具可见度 | 服务/包+大规模验证 |
| **Cell Reports Methods** | 方法导向 | 清晰 protocol + 实用性 |
| **Nature Methods** 类 | 领域顶刊 | 本质新能力；仅评估协议极难；需碾压基线与竞品 |
| **Patterns** | 数据科学跨学科 | 叙事与可迁移协议 |

**一区额外硬条件（摘要）：**  
predictor-aware 残差风险可迁移；选择性预测全面优于 magnitude 与原生 UQ；gene+chem 真模型；机制分组；最好外部盲测/湿实验。  
**当前全部未齐 → 禁止写“已具备一区”。**

### 4.4 投稿策略（务实）

1. **先小文章 / 二区方法文**：锁 claim = task-risk triage + 难 setting 矩阵（老师线）。  
2. E74 pair-risk 作 **Methods 亮点或扩展**，勿偷换题目。  
3. 一区仅在 S1–S3+S4+S7 过关后再评估改投。  
4. 化学极弱时：gene 主文 + chem 补充/第二篇，勿硬并。

---

## 5. Codex 对抗审核协议（互套）

### 5.1 角色

| 角色 | 动作 |
|---|---|
| **Proposer（执行 Codex）** | 只提交：实验编号、路径、schema 表、通过/失败、下一步 |
| **Attacker（另一 Codex/Claude）** | 只攻击：泄漏、混层、换题、挑结果、期刊夸大 |
| **Arbiter** | 对照 §1 Z 表与 §0 禁止 claim 判 PASS/FAIL |

### 5.2 每轮必须回答的攻击题（拷贝即用）

```text
A1. 本结果的 predictor_name 是什么？若是“reference/light”，为何进主表？
A2. 分数任一输入是否用了 holdout true expression / true effect？指到代码行或 provenance 表。
A3. 本结果属于 Setting 小矩阵 / 行 / 列 / 跨数据集 / random 中的哪一个？
A4. 相对 predicted magnitude 的 Δρ 与 CI 是多少？没有则为何还报“有用”？
A5. 是否与 L1 旧 7 主表或 L2 轻量结果混写为同一方法？
A6. 是否把 pair-risk 写成了单模型置信度？
A7. 阴性结果是否同页展示？
A8. 是否在测试误差可见后改过特征/权重？
A9. 该结果关闭了 Z 表哪一格？未关闭却写“按老师推进”→ FAIL
A10. 期刊句子是否超过 §4 允许强度？
```

### 5.3 一票否决（KILL）

- 用 true magnitude 打分  
- 无 `predictor_name` 的“预测错误”  
- 测试集调参  
- 删除 McFarland / seed 阴性  
- 未跑 PRESCRIBE 却写全面优于集成 UQ  
- 三 setting 未过却写可投 / 稳定二区  
- 课程论文内容进科研主仓叙事  

### 5.4 每轮输出模板（Proposer）

```markdown
## 实验 ID
## 关闭的 Z 编号
## 路径
## schema 表路径
## 主数字（含 CI）
## vs magnitude
## 阴性
## 攻击者预计会打哪里
## 下一步唯一动作（一条）
```

### 5.5 每轮输出模板（Attacker）

```markdown
## FAIL/WARN/PASS
## 命中的 A 题号
## 证据路径（打穿用）
## 要求 Proposer 补的最小实验
## 是否违反用户目标（老师线 / 二区底线）
```

---

## 6. 通往“稳定二区 / 冲一区”的关卡图

```text
[现在] 
  L3 有苗头（E74）
  Z7 Z9 Z10 未过
  claim 有漂移风险
     |
     v
[Gate Q2-A] 锁 claim（§0 二选一主叙事，另一作补充）
     |
     v
[Gate Q2-B] Z4–Z5 provenance 全绿
     |
     v
[Gate Q2-C] Z7+Z8+Z9 真预测器合同 全绿或诚实边界成文
     |
     v
[Gate Q2-D] vs magnitude + 消融 + RC@k + 复现包
     |
     v
[可投二区方法文]  ← 用户“稳定二区”最低线
     |
     v
[Gate Q1] S4 chem 真模型 + S7 竞品 + 跨域稳定增量 + 机制
     |
     v
[再评估一区期刊]
```

**无 Gate Q2-C → 无“稳定二区”。**  
**无 Gate Q1 → 无一区讨论。**

---

## 7. Codex 即日起允许 / 禁止清单

### 允许

- 跑 S1–S3 主合同（真预测器或冻结预测缓存）  
- 写 provenance / 泄漏检查  
- 报阴性  
- 推进 CPA 化学线  
- 用本文件 A1–A10 自审后再 push  

### 禁止

- 新开 panel4/5 除非 Arbiter 批准  
- 轻量高 ρ 进 Abstract/主图  
- “一定二区/一区”话术  
- 混 L1/L2/L3 数字  
- 改老师三 setting 定义为别的实验凑数  

---

## 8. 关键路径索引

| 内容 | 路径 |
|---|---|
| 老师聊天原文 | `~/.codex/attachments/3d36bc69-.../pasted-text.txt` |
| 老师拆解 | `proj/workspace/group_meeting_20260709_MAINLINE_WHITE/周老师聊天记录_要求拆解与实验设计_20260709.md` |
| 执行队列（旧编号） | `.../后续实验执行安排_按周老师要求_20260709.md` |
| 证据总账 | `proj/docs/实验结果/E63_周老师问题_证据总账_20260711.md` |
| 二区/一区门槛 | `proj/docs/实验结果/E68_一区目标_二区底线_证据门槛_20260711.md` |
| 7 主表 | `proj/docs/实验结果/Formal_main_20260604/` |
| E74 | `proj/docs/实验结果/E74_pair_risk_certificate_20260711/` |
| Grok 前序判断 | `proj/agents/grok/2026-07-12_周老师后_Codex主线判断.md` |
| Grok 数据判断 | `proj/agents/grok/2026-07-12_数据集与分区判断.md` |
| 本互审包 | `proj/agents/CODEX_ADVERSARIAL_REVIEW_PACK_20260712.md` |

---

## 9. 给 Codex 的启动指令（可直接粘贴）

```text
你是 SafeConf 的 Proposer 或 Attacker（先声明角色）。
强制阅读：
  /home/yyf/proj/agents/CODEX_ADVERSARIAL_REVIEW_PACK_20260712.md
硬约束：
  1) 只服务周老师 Z1–Z11，优先 Z7/Z8/Z9；
  2) 用户底线=稳定二区，一区仅在 Gate Q2 全过后评估；
  3) 用 §5.2 A1–A10 自审；命中 KILL 则停；
  4) 禁止课程论文、禁止下载冒充实验、禁止无脑 panel；
  5) 每轮只提交 §5.4 模板；不写空话。
当前唯一优先动作：推进能关闭 Z7 或 Z8 或 Z9 的最小实验，使用统一 schema。
```

---

## 10. Grok 终审摘要（供 Attacker 攻击）

1. **idea 可发**：post-hoc 任务风险分诊 + 难 setting 矩阵，在方法类生物信息期刊有位置；不是 Nat Methods 默认档。  
2. **数据集方向对**，合同不统一、化学真模型缺、三 setting 未过 → **现在不是稳定二区**。  
3. **Codex 后半段科学升级（E74）有价值，但对老师清单半对齐**，存在 claim 漂移。  
4. **路径**：先 Gate Q2-A→D，再谈期刊；一区另算。  
5. **互审以本文件为法**；他文与本文冲突时，以周老师原话 + §0 禁止 claim 为准。
