# 发给 Claude：GEARS native uncertainty probe 复核

请客观复核，不要默认同意 Codex。重点判断：现在已经证明 GEARS 能导出 native uncertainty，下一步是否值得扩大成 GEARS formal probe。

## Codex 做了什么

我在 Frangieh 上重新跑了 GEARS，打开 `--uncertainty`：

```text
run_gears_prediction_records --dataset frangieh --seed 1 --uncertainty
```

输出目录：

```text
code/20260426_154505_perturb_transport_final_push/outputs/gears_frangieh_uncertainty_probe_20260605/
```

文档目录：

```text
docs/实验结果/Formal_main_20260604/gears_frangieh_uncertainty_probe/
```

## 结果摘要

| 指标 | 数字 |
|---|---:|
| status | ok |
| PredictionRecord | 21 |
| native uncertainty 非空记录 | 21 |
| test MSE | 0.00150 |
| test Pearson | 0.99565 |
| top20 DE MSE | 0.00491 |
| top20 DE Pearson | 0.93528 |
| `gears_uncertainty_logvar_mean` vs true_error Spearman | 0.352 |
| p value | 0.118 |

## Codex 当前判断

1. 这证明 GEARS native uncertainty 可以导出。
2. 方向上是对的：logvar 越高，误差倾向越高。
3. 但只有 21 条 test record，所以不能作为正式结果。
4. 下一步如果要增强论文，应做 GEARS formal probe，而不是继续堆 Tahoe 或写初稿。

## 请你回答

Q1. 你是否同意：现在 GEARS native uncertainty 已经“可导出”，但还没有“正式验证有效”？

Q2. 下一步 GEARS formal probe 的最低设计是什么？

请明确：

- 是否必须改成 SafeConf held-out pair split？
- 是否保留 GEARS single split 作为 supplement？
- 至少要多少 test PredictionRecord？
- 至少要几个 seed？
- 是否需要同时导出 seed ensemble proxy？

Q3. 如果 GEARS formal probe 只能做 supplement，不适合主表，是否仍然足以支撑“predictor-agnostic（不绑定预测器）”这个说法？

Q4. 下一步优先级是否应为：

1. GEARS formal probe；
2. GEARS uncertainty vs SafeConf score 公平比较；
3. supplement 数据集补齐；
4. 其他模型 CPA / scGPT 后置。

如果不同意，请给出你认为最稳的排序。

