# 证据独立复核档案（GLM，2026-08-17 凌晨）

目的：不依赖任何既有报告的转述，**直接从任务级明细表独立重算**论文将引用的
头条数字，并与冻结汇总表逐项对照。这是"认真审核"的落点之一，也是
`code/safeconf_audit/`（最小审计包）的验证底稿。

## 1. 独立重算结果（pandas + scipy，逐任务行）

### E199（K562 未见基因，公开 TxPert STRING-GAT）

明细表：`docs/实验结果/E199_txpert_public_k562_20260802/formal_evaluation/tables/E199_TASK_CERTIFICATE.csv`（272 行）
主任务筛选：`n_cells >= 30` → **263 行**（与冻结报告 263 一致；9 行为小细胞数敏感性层）

| 量 | 冻结报告 | 独立重算 | 一致 |
|---|---|---|---|
| ρ(diversity 下界, family_rms_error) | 0.3948 | 0.3948 | ✅ |
| ρ(predicted_magnitude, family_rms_error) | 0.0955 | 0.0955 | ✅ |
| 恒等式残差 max abs | 报告口径 ≤1e-10 | 2.60e-18 | ✅ |
| 下界违反数（RMS+worst） | 0 | 0 | ✅ |

### E200（K562 整背景留出）

明细表：`docs/实验结果/E200_txpert_cross_context_k562_20260802/formal_evaluation/tables/E200_TASK_METRICS.csv`（580 行）
主任务筛选：`analysis_stratum == 'primary_ge30'` → **566 行**（一致；14 行敏感性层）

| 量 | 冻结报告 | 独立重算 | 一致 |
|---|---|---|---|
| ρ(transfer_risk, gat_centroid_rmse) | 0.4240 | 0.4240 | ✅ |
| ρ(predicted_magnitude, gat_centroid_rmse) | 0.8797 | 0.8797 | ✅ |
| ρ(training_delta_dispersion, gat_centroid_rmse) | 0.6639 | 0.6639 | ✅ |

**结论：论文正文将引用的五个头条相关系数与两个结构不变量，全部通过独立复算。**
E158 饱和、E192 ABSTAIN、E189 负效用的原始表未在本轮逐行重算（其结论为
"不可估计/门槛裁决"，机制上不依赖浮点复算），引用时以冻结报告为准。

## 2. 审计包（safeconf-audit v0.1.0）

- 位置：`code/safeconf_audit/`（pip 可安装：`pip install -e .`；命令 `safeconf-audit --repo /home/yyf/proj`）
- 检查项：上述全部点估计（容差 5e-4）+ 任务数 + 恒等式 + 违反数 + 簇 bootstrap
  CI 的**符号一致性**（分歧 CI 排除 0、幅度 CI 跨 0、transfer CI 排除 0）
- 诚实性口径：CI 数值不做逐位相等（原始 RNG 种子不在冻结内容内），只要求
  "是否排除 0"的结论一致——该口径已写入包 README，供论文 Methods 引用
- 运行结果：**2026-08-17 02:2x，`ALL PASS`（12.6 秒，13 项检查）**。
  独立重算的簇 bootstrap CI（E199 分歧 [0.283,0.495]、E200 transfer [0.347,0.496]）
  与冻结值（[0.2835,0.4969]、[0.3506,0.4953]）数值上几乎重合——强于符号一致性
  要求。优化记录：首版逐簇 pandas 过滤太慢（>6 分钟未完成），改为 numpy 预分组后
  12.6 秒完成。

## 3. 审核抓获的标签错误（重要，已全量修正）

**E202a 对照表与交接包 §9 把 E189 双未见的 −0.349～−0.241 标成了 "utility"
（复核效用）。经核对正式报告，这是 Spearman 相关区间：**

- `E189_INTERPRETATION.md` §4 明确列为"分歧与 family error 的关系"：
  随机缺格 0.368–0.412（CI 高于 0）、整列 0.210–0.247、整行 −0.095～−0.013、
  **双未见 −0.349～−0.241（Spearman）**；
- `E191_INTERPRETATION.md` 的 20% 预算效用另有其数：
  **双未见 diversity −0.127、magnitude −0.080（均低于随机期望）**。

已修正的交付物：英文正文、中文正文（结果段+表1）、审核报告 E189 行、
图 2 森林图行、图 1c 足迹行、补充图 S1 面板 b（全部重生成）。
**后续 AI 引用双未见负结果时，必须区分 Spearman（E189）与 utility（E191）。**

## 4. 给后续 AI 的复核提示

1. E199 的 263 行筛选条件是 `n_cells>=30`（表内无 stratum 列，E200 才有）；
2. 簇 bootstrap 的簇键用 `condition_label`；
3. 若重跑包出现 FAIL：先查 pandas/scipy 版本差异，再查表是否被改动——
   表被改动属于严重事件，须立即冻结并报告作者。
