# E206 浮点并列修正记录

发现时间：2026-09-12

## 原因

E207 独立复算发现，直接执行
`alpha × rank_magnitude + (1-alpha) × rank_safeconf` 时，二进制浮点会把
`1-0.8` 表示为 `0.199999...`。两项加权秩在数学上完全相同时，运算顺序可能产生
约 `1e-16` 的差异，使 Spearman 把并列任务误当成有先后。

E206 使用相同线性融合方式，因此必须复核。修正采用数学等价实现：对
`alpha=j/20`，先计算

```text
j × average_rank(magnitude) + (20-j) × average_rank(SafeConf)
```

再统一除以 `20 × 任务数`。平均秩只含整数或半整数，整数加权后相同理论分数保持
严格相等。没有改变输入、权重网格、评价端点、任务或随机种子。

## 修正前文件留痕

| 文件 | SHA-256 |
|---|---|
| `E206_REPORT.md` | `4b7a0691bfd2d245400640c868753f2f7cbc8f603e45b0bff5a5c6921a024afa` |
| `E206_STATUS.json` | `3b4939a3ae63b397c3d64c366ea099583b4d5632efb92081d6b404c34b611515` |
| `E206_WEIGHT_SWEEP.csv` | `b267789a6887dd51eddc003a39d162b139aedd2141c2486af2fbaa3d3f461588` |
| `E206_LOTO_RESULTS.csv` | `37e9d807a0df79124b7750b2185374fd78b266323a7de9e875f673f171ff08bb` |
| `E206_BOOTSTRAP_SUMMARY.csv` | `455dc58ea20694dbdebe2b8bfbea1bfe944cef1b1d68e248376f3bfda907fb36` |

修正后必须重新执行 5,000 次簇自举，并在本文件追加新旧主结果差值。修正不保证原
0.80 候选仍是最优；若发生变化，E205 已冻结的 0.80/0.20 仍保持不变，作为事前
登记候选完整报告。

## 重算结果

5,000 次 perturbation-cluster bootstrap 已于 2026-09-12 完成：

- 留一背景选择仍为 K562=0.80、RPE1=0.75、HepG2=0.80、Jurkat=0.80；固定候选
  仍为 0.80/0.20；
- 四背景宏平均 Spearman 仍为 0.7604，幅度为 0.7421，差值点估计为 0.0182944；
- Spearman 差值 95% 区间仍为 `[0.0091, 0.0271]`；
- 20% 复核效用差值点估计仍为 0.0133937，区间仍跨 0：
  `[-0.0139, 0.0520]`；
- 变化只出现在少量并列的精确排序和末位小数，四位小数主结论不变。

修正后关键文件：

| 文件 | SHA-256 |
|---|---|
| `E206_WEIGHT_SWEEP.csv` | `fbf44c38350ecc09e2f684e725db58a80b0aa4fb3757f131bf8c1c338ef1ee2f` |
| `E206_LOTO_RESULTS.csv` | `8c2c767e4f561002c13079786bac51b25313925bcc7d78493b60686d41157059` |
| `E206_BOOTSTRAP_SUMMARY.csv` | `8081de482cd40ea36386afc0f047c4118619b6251c06fce5d8c708977c38e99b` |

报告和状态文件含生成时间，不在这里自引用哈希；它们由最终
`E206_STATUS.json.outputs` 清单登记。
