# 请复核 Tahoe adapter smoke 设计

角色：请你当客观审稿人和方案设计顾问，不要默认同意 Codex，也不要顺着用户“想冲一区”的愿望。

## 当前事实

SafeConf 当前主线是：single-cell perturbation prediction output confidence scoring（单细胞扰动预测结果可信度打分）。

7 主表 formal audit 已完成。Tahoe-100M 现在只作为 external mega-scale validation candidate（超大外部验证候选），不是当前主表结果。

Tahoe 已下载：

- metadata + pseudobulk differential expression（聚合差异表达）
- pseudobulk 分片：1026/1026
- 总大小约 85G
- 未下载 337G 全量 expression_data（原始表达矩阵）

Tahoe 当前审计显示：

- obs metadata rows: 100,648,790
- drug × cell_line pairs: 19,000
- pairs with at least 20 cells: 18,999
- pseudobulk 包含 `log2FoldChange`, `n_cells_trt`, `n_cells_ctrl`, `drug`, `concentration`, `Cell_ID_DepMap`, `Cell_Name_Vevo`

## Codex 准备做的 smoke

不是正式实验，只是 adapter 可行性测试：

1. 读取少量 Tahoe pseudobulk parquet 分片。
2. 以 `(cell_line, drug, concentration)` 构造 task。
3. 以 `log2FoldChange` 作为 true_effect（真实效应向量）。
4. 做 held-out pair split，检查 test pair 不在 train。
5. 用 V0 baseline：同 drug + concentration 在其他 cell line 下的平均 effect。
6. 生成 smoke PredictionRecord 和 leakage audit。

## Codex 已经误提前跑了一个很小的 smoke，请你也复核这个

说明：这不是正式实验，也不是论文结果。Codex 原本应该先把本文件发给你复核，再决定是否启动；它提前跑了一个很小的 smoke，现在需要你客观判断这个方向能不能继续。

已跑 smoke 参数：

- scanned shards（扫描分片）：10 / 1026
- selected tasks（选中任务）：500
- contexts（背景/cell line）：8
- perturbations（扰动/drug+dose）：480
- genes（基因）：1000
- PredictionRecords（预测记录）：30
- test pair leakage（测试组合泄漏）：0
- test context missing（测试背景在训练中缺失）：0
- test perturbation missing（测试扰动在训练中缺失）：0
- effect definition（效应定义）：`logFC`，来自 Tahoe pseudobulk `log2FoldChange`
- predictor（预测器）：`V0DrugMeanPseudobulk`，即同 drug+dose 在其他 cell line 的平均 logFC

输出位置：

```text
code/20260426_154505_perturb_transport_final_push/outputs/safeconf_tahoe_pseudobulk_smoke_20260605/
```

当前只说明：

> Tahoe pseudobulk 可以被转成 SafeConf 风格的 PredictionRecord，并且 smoke split 没有 pair leakage。

当前不能说明：

> SafeConf 已经在 Tahoe 上有效，或者 Tahoe 可以直接进入论文主表。

## smoke v2 更新

根据你的回复，Codex 补了 smoke v2：

- scanned shards：40 / 1026
- selected tasks：2577
- contexts：41
- perturbations：1136
- genes：1000
- min exact support：3
- PredictionRecords：488
- predictors：
  - `V0ExactDoseMean`
  - `V0DrugMeanAcrossDose`
- test pair leakage：0
- test context missing：0
- test perturbation missing：0
- true_error_rmse CV：0.709

但出现两个需要你重点判断的问题：

1. concentration leakage audit：
   - `same_drug_other_concentration_in_train_ratio = 0.9995`
   - 也就是说 test 里的 drug+dose 虽然 pair 没泄漏，但同 drug 的其他 dose 几乎总在 train 里。

2. plate audit：
   - `test_plate_seen_in_train_ratio = 1.0`
   - `cell_line_plate_median = 2.0`
   - 这说明 plate 不是完全被 pseudobulk 消掉的自由变量，可能和 cell line 混杂。

输出小表：

```text
docs/实验结果/Formal_main_20260604/tahoe_adapter_smoke/tables/RUN_STATUS_V2.json
docs/实验结果/Formal_main_20260604/tahoe_adapter_smoke/tables/TAHOE_PREDICTION_RECORDS_SMOKE_V2.csv
docs/实验结果/Formal_main_20260604/tahoe_adapter_smoke/tables/TAHOE_CONCENTRATION_LEAKAGE_AUDIT_SMOKE_V2.csv
docs/实验结果/Formal_main_20260604/tahoe_adapter_smoke/tables/TAHOE_TEST_PLATE_AUDIT_SMOKE_V2.csv
docs/实验结果/Formal_main_20260604/tahoe_adapter_smoke/tables/TAHOE_PREDICTOR_DISAGREEMENT_SMOKE_V2.csv
```

## 请你回答

1. `log2FoldChange` 能不能作为 SafeConf 的 true_effect？它和主表的 `mean_diff` 定义不一致，这个问题该怎么写？
2. task key 应该是 `(cell_line, drug)`，还是必须是 `(cell_line, drug, concentration)`？
3. pseudobulk 中有 plate（板）字段，是否应该把 plate 纳入 task 或 split，防止 plate leakage？
4. 如果 smoke 只读 2-5 个分片，能不能说明“adapter 可行”？还是必须读全量 1026 个分片？
5. V0 baseline 用“同 drug 在其他 cell line 的平均 log2FoldChange”是否合理？
6. 上面这个 30 条 PredictionRecord 的 smoke 是否足够证明 adapter 可行，还是必须先扩大到 500-1000 条 smoke records 再讨论？
7. Tahoe 如果 smoke 通过，下一步应该是：
   - A. 全量 Tahoe external validation；
   - B. 先抽样 formal validation；
   - C. 只放 supplement；
   - D. 暂时不放论文。
8. 有没有必要现在下载 337G expression_data？请给明确 yes/no 和理由。

## 请追加回答 smoke v2 的 4 个问题

9. `same_drug_other_concentration_in_train_ratio ≈ 1.0` 是否意味着 Tahoe 的 held-out pair split 对 drug-dose 任务仍然太容易？是否应改成 held-out drug 或 held-out drug-family split？
10. `test_plate_seen_in_train_ratio = 1.0` 且 cell_line plate 中位数只有 2，这是否足以要求 plate sensitivity analysis？
11. 488 条 PredictionRecord 距离 500 条很近，但一次 50-shard 扩大运行触发底层 parquet/内存崩溃。你是否认为 488 条已经足够作为 smoke v2，而不是继续强行扩大？
12. 在这种 dose/plate 风险下，Tahoe 下一步是：
    - A. 继续 sampled formal validation；
    - B. 先重设 split；
    - C. 只保留 adapter feasibility；
    - D. 暂停 Tahoe，转回 7 主表 + supplement。

## 注意

- 不要把 smoke 数字当论文主结果。
- 不要为了追求大数据而牺牲 effect definition（效应定义）一致性。
- 如果你认为 Tahoe 不适合 SafeConf，请直接说，不要因为它很大就默认支持。
