# 论文完成度审核，以及 GPT 论文学习会话里哪些话不能信

审核日期：2026-09-06  
被审核会话：Codex/ChatGPT 论文学习会话 `019f139e`（文件 `~/.codex/sessions/2026/06/29/rollout-2026-06-29T21-42-29-019f139e-0161-7483-aed1-857a3ad57666.jsonl`）  
数字来源：`CLAIM_TABLE.json`，由官方 CSV 生成，不以聊天记录为准。

本文只回答两件事：现在这篇研究做到哪了；GPT 长会话里哪些科学判断已经过期、说重了、或和正式表不一致。

---

## 1. 完成度一句话

**筛错这件事有正式盲测证据。训练加权还没有正式效果。跨架构和新外部确认都还没有。所以现在不能把二区写成一定能发，更不能把一区写成已经能冲。**

当前可对外陈述的主结果来自 E201：四个细胞系整行留出（holdout）、1808 个主任务。SafeConf 风险分与家族均方根误差的合并 Spearman 为 0.4082，四个细胞系为 K562 0.4803、RPE1 0.2868、HepG2 0.5633、Jurkat 0.4453。预测幅度（predicted magnitude）为 0.6189。控制幅度后的偏相关为 0.2503。固定检查 20% 任务时，SafeConf 效用 0.3200，幅度 0.5943。K562 上深度模型质心误差 0.0540，官方简单规则 0.0584，“猜成没变化”0.0523。

证据路径：

- `docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/tables/E201_RISK_ASSOCIATIONS.csv`
- `docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md`
- `docs/实验结果/E199_txpert_public_k562_20260802/formal_evaluation/tables/E199_RISK_ASSOCIATIONS.csv`
- `docs/实验结果/E204_risk_guided_training_20260830/PROFILE_ACCEPTANCE_20260905.md`

---

## 2. 已经闭合 / 尚未闭合

| 工作 | 状态 | 能说什么 |
| --- | --- | --- |
| E199 未见基因 | 正式报告已出 | 三个公开 TxPert 检查点的分歧有用，幅度弱 |
| E200 整个 K562 留出 | 正式报告已出 | 幅度远强于旧风险分 |
| E201 四细胞系盲测 | 正式报告已出 | SafeConf 有用，单独弱于幅度，控制幅度后仍有 0.2503 |
| E202 | 主检验失败 | 负结果，必须保留 |
| E204 | 工程验收通过，profile only | 程序能加权；正式 80 轮没有 |
| E205 | 只有冻结协议 | 0 张结果表 |

E204 的 `PROFILE_ACCEPTANCE_20260905.md` 写明本轮“不判断模型效果”。数据盘上仍是 `profile` / `smoke`。把 E204 profile treated as performance gain，属于误导。

---

## 3. GPT 会话 `019f139e` 的结构问题

这条会话从 2026-06-29 开到 2026-09-05，约 290 MB。它不是一条连续的“当前论文学习”，而是多层旧故事叠在一起：

1. 6 月底：课程论文、考试复习、HTML 讲义。
2. 7 月：组会、按“一区冲刺”下载数据、GEARS/scGPT 七数据集、Tahoe 化学扰动。
3. 7 月中后期：把“稳定二区/一区”拆成门槛并自己加实验编号。
4. 8–9 月：才回到 TxPert 四细胞系和周老师的两个用途。
5. 9 月 5 日：整理手把手学习稿，这一轮和正式 E201 表基本对齐。

因此：**不能把整条 GPT 会话当成当前论文事实。** 7 月说“已经超过幅度”的实验合同，和 9 月 E201 的合同不是同一件事。

下面六类误导在会话或仓库入口里都出现过。类名按审核清单保留英文，便于机器核对。

### 3.1 four seeds treated as four models

GPT 和仓库入口经常把 E201 的四个随机种子说成“四个模型”。正式事实：四个成员都是 TxPert STRING-GAT，只改初始化。这是初始化敏感性，不是跨架构不确定性。E205 协议就是为了补这个空。9 月 5 日最后一轮 GPT 已经改口，但更早的“四个模型”仍会带偏阅读。

对照：`E201_CORE_REPORT.md` 第 1 段，“四种子重训练 family”。

### 3.2 E199 described as four random seeds

仓库首页 README 在 2026-09-06 审核前写成“E199：同一模型四个随机种子的分歧”。这是错的。E199 的注册家族是三个公开检查点：GAT、Exphormer、Exphormer-MG。证据：`E199` 的 `ANALYSIS_FREEZE.md` 与 `E199_RISK_ASSOCIATIONS.csv`。

### 3.3 E204 profile treated as performance gain

用户多次要求“直接安排实验、我只要结果”。GPT 容易把“程序跑通”说成“加权已经改善模型”。正式文件只允许说工程验收 PASS。K562 一轮训练 294,951 条、权重回退 0、目标扰动表达读取 0，这些数字证明实现正确，不证明误差下降。

### 3.4 稳定二区/一区

用户目标一直是“2区一定能发，1区可以冲刺”。GPT 在 2026-07-06 起按“一区优先，二区保底”开工，生成过 `docs/投稿升级/Q1_CCFA_upgrade_20260707/` 等工作台，并把若干历史实验写成冲刺成果。那些材料早于 E201 解封。

2026-07-11 GPT 自己也写过“还达不到稳定二区”。9 月 5 日研究判断再次写：不宜说稳定二区。当前官方数字没有推翻这一条。

### 3.5 SafeConf beating magnitude

会话里至少有两套互相打架的话：

- 7 月 E61/E69：多数方向幅度更强，GPT 当时正确要求保留负结果。
- 7 月 11 日 E74：在另一套 GEARS–scGPT、72 个任务的合同上，声称相对幅度的 Δρ 区间高于 0。
- 9 月 E201：合并效用差 -0.2743，SafeConf 没有打赢幅度。

当前论文主线是 E201，不是 E74。把 E74 的“超过幅度”搬到今天，就是 SafeConf beating magnitude 这类过期主张。

### 3.6 agents/ or July Gate used as current fact

`GATE_STATUS_20260729.md` 截止于 2026-07-29，不含 E201 解封。`agents/` 里 Grok/GLM 原始意见不是事实源。GPT 长会话多次从这些目录推断“现在能不能发”。学习稿 00 已经把权威顺序改对；读 GPT 历史时仍会倒回去。

---

## 4. 锁定数字对照表

这些显示值必须和 `CLAIM_TABLE.json` 一致，也必须出现在教学稿和投稿报告里。

| 锁定项 | 显示值 | 官方表 |
| --- | --- | --- |
| 主任务数 | 1808 | `E201_RISK_ASSOCIATIONS.csv` pooled `n_tasks` |
| SafeConf 合并 Spearman | 0.4082 | 同上 |
| 幅度合并 Spearman | 0.6189 | 同上 |
| 偏 Spearman | 0.2503 | `E201_PARTIAL_ASSOCIATIONS.csv` |
| SafeConf 20% 效用 | 0.3200 | `E201_REVIEW_UTILITY.csv` |
| 幅度 20% 效用 | 0.5943 | 同上 |
| K562 / RPE1 / HepG2 / Jurkat | 0.4803 / 0.2868 / 0.5633 / 0.4453 | 分 target 行 |
| K562 质心 / 对照 / 官方规则 | 0.0540 / 0.0523 / 0.0584 | `E201_TARGET_ERROR_SUMMARY.csv` |

---

## 5. 对用户问题的直接回答

GPT 有没有误解？有。主要不是 9 月 5 日最后一轮算错 E201 数字，而是整条会话把课程、旧七数据、Tahoe、一区冲刺工程和当前 TxPert 主线叠在一起。中立判断：

- 9 月 5 日学习稿里的 E201 数字，和正式 CSV 一致，可以当教材骨架。
- GPT 会话本身不能当教材。
- “二区一定能发”在当前证据下不成立。缺的是训练收益、跨架构和新外部确认，不是再写一份更乐观的说明。
