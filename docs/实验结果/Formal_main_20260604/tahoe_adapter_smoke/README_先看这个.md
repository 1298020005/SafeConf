# Tahoe adapter smoke 下一步说明

更新时间：2026-06-05

## 一句话

Tahoe-100M 已经下够第一版需要的数据，不继续下 337G 原始矩阵。下一步不是“再下载”，而是写一个 pseudobulk adapter（聚合差异表达适配器），先证明 Tahoe 能不能变成 SafeConf 的 PredictionRecord（预测记录）。

## 为什么做这个

当前 7 主表已经够支撑一篇稳健的方向，但如果想冲更高目标，需要一个更大的 external validation（外部验证）。

Tahoe 的优势：

- 约 100M 个 single-cell profiles（单细胞表达记录）。
- 50 个 cell line（细胞系）。
- 约 380 个 drug（药物）。
- 约 19,000 个 observed drug × cell_line pair（药物×细胞系组合）。
- 已有 pseudobulk differential expression（聚合差异表达），包含 `log2FoldChange`（基因变化幅度），可能不需要 337G 原始表达矩阵。

## 这一步只做什么

1. 读取少量 Tahoe pseudobulk parquet 分片。
2. 构造 task（任务）：`cell_line（细胞系） × drug + concentration（药物+剂量）`。
3. 构造 held-out pair split（留出背景×扰动组合切分）。
4. 用最简单的 V0 baseline（同 drug 在其他 cell line 的平均 effect）生成 smoke PredictionRecord。
5. 输出 leakage audit（泄漏审计）和样本量统计。

## 当前已完成的小 smoke

Codex 已经先跑了一个很小的 smoke。这个顺序应该先让 Claude 复核再扩大，后续会按复核结果执行。

当前 smoke 结果：

- 扫描 10 / 1026 个 pseudobulk 分片。
- 选出 500 个 task。
- 覆盖 8 个 cell line 和 480 个 drug+dose。
- 生成 30 条 PredictionRecord。
- test pair leakage = 0。
- 结论：adapter 方向初步可行，但这不是正式结果。

## smoke v2

根据 Claude 的复核意见，又补了一个更严格的 smoke v2：

- 扫描 40 / 1026 个 pseudobulk 分片。
- 选出 2577 个 task。
- 覆盖 41 个 cell line 和 1136 个 drug+dose。
- 要求 `min_exact_support >= 3`。
- 生成 488 条 PredictionRecord。
- predictor（预测器）两个：
  - `V0ExactDoseMean`：同 drug+concentration，在其他 cell line 下的平均 logFC。
  - `V0DrugMeanAcrossDose`：同 drug，跨 concentration，在其他 cell line 下的平均 logFC。
- test pair leakage = 0。
- test context missing = 0。
- test perturbation missing = 0。
- true_error_rmse CV = 0.709，说明误差有足够变异，不是所有任务都差不多。

但也有两个风险：

- `same_drug_other_concentration_in_train` 比例接近 1，说明同一个 drug 的其他 concentration 几乎总在 train 中出现。这个不是 pair leakage，但属于 dose-response information leakage（剂量反应信息泄漏）风险。
- `test_plate_seen_in_train_ratio = 1.0`，说明 test 的 plate 在 train 中都出现过。因为 cell line 的 plate 中位数是 2，plate 和 cell line 可能混杂，后续不能简单说 plate 风险已消失。

所以当前判断：

> Tahoe adapter 技术上可行，但 Tahoe formal validation 必须先设计 dose/plate sensitivity analysis，不能直接扩大成论文结果。

详细复核文本：

`发给Claude_Tahoe_adapter复核.md`

## 这一步不做什么

- 不把 Tahoe 直接写进正式主表。
- 不训练 GEARS / CPA / scGPT。
- 不下载 337G 原始 expression_data。
- 不拿 smoke 数字当论文结论。

## 成功标准

可以继续 Tahoe formal external validation 的最低条件：

- 至少能构造 500 个以上 task。
- test pair 不泄漏到 train。
- test 的 drug 和 cell line 在 train 中都有支持。
- V0 smoke 能生成 PredictionRecord，而不是只有 metadata。

如果 smoke 失败，Tahoe 只保留为数据储备，不进入论文实验。
