# SafeConf 当前研究状态

更新时间：2026-08-15 23:55（Asia/Shanghai）

## 一句话

E201 盲训练已到 **12/16**（K562 seed 4 运行中，真值仍为 0）。
一区缺口不在“再多跑几个模型”，而在主张收窄、竞品对照表、以及 E201 开真值后
按门判决。详细审核见
[`../投稿准备/Q1_PATH_SEMANTIC_AUDIT_20260815.md`](../投稿准备/Q1_PATH_SEMANTIC_AUDIT_20260815.md)。

## 投稿口径（先写在这里，避免再混）

- 主投：Briefings in Bioinformatics（小类一区 / JCR Q1）
- 冲刺：Genome Biology（大类一区，需软件包和更宽验证）
- 主主张：预测后 fail-closed 风险合同，不是“永远优于 magnitude”，也不是分区口号

## 正在跑

E201 HepG2/Jurkat/RPE1 的 seed 3 已完成；当前 K562 seed 4。
随后还有 RPE1 / HepG2 / Jurkat 的 seed 4。不要并行第二份占卡训练。

## 下一件立刻做的科学工作

E202a：用已解盲的 E158/E159、E189、E192、E199、E200 做 setting × 方法对照表。
见 [`E202_q1_blocker_closure_20260815/README_先看这个.md`](E202_q1_blocker_closure_20260815/README_先看这个.md)。
