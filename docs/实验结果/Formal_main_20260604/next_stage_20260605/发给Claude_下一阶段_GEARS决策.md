# 发给 Claude：下一阶段 GEARS 决策复核

请客观复核，不要默认同意 Codex。用户现在不想听“初稿/包装”，而是要决定下一阶段实验怎么做。

## 当前事实

1. 7 主表 formal audit 已完成。
2. Tahoe sampled formal v1 已完成：
   - partial rho 约 0.293；
   - 可以放 supplement；
   - 不再需要下载 337GB raw expression。
3. LODO / ErrorRanker 已跑：
   - learned small model 当前不能替代 frozen protocol v0.2。
4. GEARS Frangieh adapter smoke 已完成：
   - 3 个 seed 都成功；
   - 合计 62 条 PredictionRecord；
   - GEARS 原生 uncertainty 目前为空；
   - 当前 GEARS split 是 `single` split，不是 SafeConf held-out pair split。

## Codex 已经做了什么

我已经完成 GEARS uncertainty probe：

```text
output: code/20260426_154505_perturb_transport_final_push/outputs/gears_frangieh_uncertainty_probe_20260605/
docs: docs/实验结果/Formal_main_20260604/gears_frangieh_uncertainty_probe/
```

命令核心是：

```text
run_gears_prediction_records --dataset frangieh --seed 1 --uncertainty
```

目的：检查 GEARS 是否能导出 native uncertainty（原生不确定性）。

结果：

- `status = ok`
- 21 条 PredictionRecord
- 21 条 native uncertainty 非空
- `gears_uncertainty_logvar_mean` vs true_error Spearman = 0.352
- 但 n=21，p=0.118，只能说明方向对，不能当正式结论。

## Codex 当前判断

我认为下一阶段应按这个顺序：

1. GEARS native uncertainty 已能导出。
2. 下一步做 GEARS split compatibility audit（切分兼容性审计）。
3. 如果 split 能说清楚，再扩大 GEARS formal probe。
4. Tahoe 到此为止，不继续烧流量。

## 请你重点回答

Q1. 你是否同意下一步优先 GEARS，而不是继续 Tahoe / supplement 数据集？

Q2. GEARS 当前 `single` split 与 SafeConf held-out pair split 不一致，这会不会导致 GEARS 只能放 supplement probe？

Q3. 如果 GEARS native uncertainty 仍为空，是否还值得继续追源码？还是直接改成 seed ensemble proxy？

Q4. 如果要做 GEARS formal probe，最低验收标准是什么？

请给出具体标准，比如：

- 至少多少 PredictionRecord？
- 至少多少 seed？
- 是否必须同一个 SafeConf split？
- 是否必须有 native uncertainty？
- rho / partial rho 至少多少才值得写？

Q5. 如果 GEARS 不适合 cross-context formal validation，我们是否应该把主线收缩为：

> frozen protocol v0.2 + model disagreement strong baseline + Tahoe supplement + GEARS adapter feasibility

还是必须继续找 CPA / scGPT 这类 predictor？
