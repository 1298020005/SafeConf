# Codex: 只读这一份。不要从聊天、七月旧包或 GLM 草稿当事实入口。

路径：`/home/yyf/proj/agents/grok/CODEX_READ_THIS.md`  
副本：`/home/yyf/proj/agents/CODEX_READ_THIS.md`  
CWD：`/home/yyf/proj`  
日期：2026-08-20  
作者：Grok，给 Codex 审核与接着做。

本目录里 `2026-07-12_*`、粘贴/剪贴板脚本、COPY/PASTE 全是旧工具，**不要当当前科研入口**。

---

## 你要对用户做什么

1. 通读本文件。
2. 核对下面“事实”是否与仓库冻结报告一致。发现错误就写证据路径，不要默认同意 Grok 或 GLM。
3. 只从 **Next only** 往下做。不要重开已解盲负结果去调分。
4. **不要打开 E201 target 扰动真值。** 16 个模型已封存，预测未全部完成，真值释放是后面的硬步骤。

---

## 硬目标

周老师（2026-07-09 原话大意）：分数跟谁的误差比、打分有没有看真值、没见过的任务怎么打；再补小矩阵、整行整列、跨数据集。gene 和 chemical 都能算。三个更难 setting 解决了，可以写小文章。那是科学闭环，不是录用保证。

作者后来要“一区，至少二区”。口径必须拆开，禁止把分区写成已达成：

- 中科院**大类**生物学 1 区（Genome Biology / Nature Methods）：现在不能投。
- Briefings in Bioinformatics：JCR Q1，中科院大类 2 区、小类计算生物学 1 区。这是现实主投。学院认大类还是小类，作者自己确认。
- Bioinformatics：方法对口，JCR Q1，中科院大类约 3 区。学校若认大类，这不是一区。
- “随便二区”不成立。负结果还在，E201 没评完。

允许的主张只有一句：

> 预测已经给出、真值未知时，对任务做风险审计。已验证的 setting 可以排序复核；未验证、区间跨 0、分数饱和或会帮倒忙时，明确 ABSTAIN。不宣称普遍优于 predicted magnitude，不宣称首个不确定性方法。

---

## 事实（2026-08-20，以冻结报告和现场为准）

### 已解盲、可写进论文的

| 块 | 能写 | 不能写 |
|---|---|---|
| E189 小矩阵/行列/双未见 | 随机缺格偏容易（Spearman 约 0.37–0.41）；双未见 Spearman **−0.35～−0.24** | 所有缺失模式都好用 |
| E191 预算效用 | 双未见 20% 效用：分歧 **−0.127**，幅度 **−0.080**（低于随机） | 把 −0.35～−0.24 写成 utility（那是 Spearman，交接包曾标错，GLM 纠正了对） |
| E190 跨研究 K562 | 合同成立；分歧 ρ=0.424 与幅度 ρ=0.420 相当 | 跨研究已经明显更好 |
| E192 跨研究 RPE1 | ρ=0.300，CI **跨 0** → 事前规则 **ABSTAIN** | 跨细胞系迁移成功 |
| E199 公开 TxPert，K562 未见基因，n=263 | 分歧 ρ=**0.3948** [0.2835, 0.4969]，20% utility **0.2084** [0.1033, 0.3755]；幅度 ρ=0.0955，CI 跨 0 | 已在所有细胞上成立 |
| E200 公开 TxPert，K562 整背景留出，n=566 | 风险分 ρ=**0.4240**；幅度 ρ=**0.8797**，utility **0.9133**。幅度显著更强 | 整背景留出证明 SafeConf 更好 |
| E158/E159 PRESCRIBE | 官方分数在严格未见基因上饱和，相关不可估计 → ABSTAIN | “打赢 PRESCRIBE”；重跑 Norman P3/P4 当确认成功 |
| E198 | 评价指标事前校准 | 这等于验证了 SafeConf |
| E194 | 证书属于预注册家族，复制成员会灌水 | 分歧越大越好 |
| chemical E84/E87/E89 | 能算；难 setting 上幅度更强 | gene+chemical 同一套成功 |

**科学主线（同意 GLM）：** E199 与 E200 是翻转，不是矛盾。未见基因上分歧有增量（Δρ=+0.299，CI 全正；Δ效用 CI 跨 0）；整背景留出上相对幅度是负增量（Δρ=−0.456，Δ效用 −0.548）。固定加权两边都赢不了。论文写“信号有效域 + 未验证即停用”，不要写“我们的分数更好”。

证书门 `family_RMS² = centroid_RMSE² + disagreement²` 对等权家族是恒等式，只作完整性检查，不当发现。

E201 的“家族”是 **同一 STRING-GAT 的四个种子**，不是 scGPT–GEARS 跨架构分歧。正文必须写清。

### E201 现场

- 训练：**16/16 COMPLETE**（jurkat seed 4 于 2026-08-17 13:04 结束）。target 扰动访问全程 0。
- 封存：`SEALED_16_CHECKPOINTS`，提交 `cd1779e`，GitHub/Gitee 一致。
- 防泄漏检查修订：`28736b7`（dummy-X 不得改掉 matched control）、`1010ea5`（float32 容差 1e-5；实测差 4.8e-7，不是泄漏）。
- 零真值预测：仅 **K562 seed 1 COMPLETE**（2026-08-19 01:49）。seed 2–4 与其余三个细胞 **未做**。2026-08-20 现场无预测进程，GPU1 空。
- 真值：**未释放。禁止打开。**

评估链剩下的唯一主实验：

```text
K562 seed 2–4 → RPE1/HepG2/Jurkat 各 seed 1 再 2–4
→ 风险分 + general baseline + E200 等价性
→ 双远程提交预测前哈希
→ 才释放 target 真值
→ 三门分判，四个细胞全报
```

命令以 `agents/glm/04_E201_RUNBOOK.md` 为准。GPU 只用 GPU1。目录已存在则脚本拒覆盖。K562/seed_1 已完成，从 seed 2 接着，不要重跑 seed 1。

### GLM 产出怎么用

目录：`agents/glm/`（**未进 Git**）。

**采信：** 审核报告 `00_AUDIT_REPORT_20260817.md` 的分区与主张；`06_EVIDENCE_AUDIT.md` 对 E199/E200 复算；`08_SCIENCE_CRITIQUE_AND_FIXES.md` 的翻转与固定分数不可两头赢；E189 Spearman/utility 纠错。

**不采信为投稿图：** `agents/glm/paper/figures/`（标题叠字、框体重叠、E201 仍画着 running 08-17）。不要用任何带 V2 字样的旧图。

**草稿可用、数字未齐：** `agents/glm/paper/SafeConf_manuscript_v1_EN.md` 与中文对照。E201 必须保持 `[[E201-PLACEHOLDER-*]]`。图 7 条件合同是**事后示意**，不能当正式证据。E203（换架构、源侧选权重）**现在不要开**。

**软件包：** `code/safeconf_audit/` 声称 13 项复算通过，未进 Git。投稿前再核一次，不要在论文里写不存在的 pip 包名。

---

## Next only（顺序，不要加戏）

1. **接着 E201 STAGE_3。** `CUDA_VISIBLE_DEVICES=1`。K562 seed 2、3、4，然后 RPE1 / hepg2 / jurkat，每个细胞先 seed 1 再 2–4。每份核验：`E201_PREDICTION_RUN.json` 为 COMPLETE、有 `predictions.npy`、**没有** `truth.npy`、target 表达非零计数为 0。
2. STAGE_4：`run_e201_pretruth_risk_features.py` 与 `build_e201_official_general_baseline.py`。E200 等价性最大绝对残差 ≤ 5e-6。
3. STAGE_5：预测前表和哈希双远程提交。三方 HEAD 一致。
4. STAGE_6：只有上一步完成后才 `release_e201_target_truth.py`。
5. STAGE_7：正式评价，三门分开判，四个 target 全报，负结果保留。
6. **图：** 扔掉 GLM `figures/` 那套。按冻结数字重做白底主图（合同、E199/E200 翻转、足迹、E201 协议、分量）。E201 结果数字出来前，协议图不许画假结果。投稿图英文；给老师的中文图可以另存，不要混进正文。
7. 出数后才填 GLM 文稿占位符。摘要按三叉选一版，不许预先选好听的。

明确不做：重跑 E198–E200 调分；重跑 PRESCRIBE P3/P4；开 E203；加湿实验；把 chemical 塞进 gene 主表；宣称已是稳定一区/二区。

---

## 图与论文（给审核用的结论，不是过程）

现在没有可投稿的图。GLM 图不合格。Grok 写过重绘脚本但没有完成可用成品。Codex 审核时把“无合格主图”记为缺口，不要把坏图送审。

题目用这个，不要改成预测器题目：

> SafeConf: a fail-closed post-prediction reliability contract for single-cell perturbation models

中文对应：单细胞扰动预测后的风险审计——何时可以排序复核、何时必须停止。

---

## 回复格式（每一轮）

1. 做了哪一步  
2. 路径  
3. 数字（CI，且对 magnitude）  
4. 负结果  
5. 有没有踩 Kill  
6. 下一步只一件  

## Kills

看 holdout 真值；改 E201 种子/公式/任务；删负结果或只报好看的细胞；把 GLM 坏图当终稿；把 Fig7 模拟当前瞻证据；提前写 E201 成功/失败；声称稳定 Q1/Q2；把七月 `CODEX_READ_THIS` 旧“Next only”当今天任务。

**现在唯一动作：E201 零真值预测从 K562 seed 2 继续。**
