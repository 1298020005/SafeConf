# Formal Main Decision

## 一句话结论

`CONTINUE_WITH_FOCUSED_CLAIMS`：继续做，但论文主张必须收缩为“基因扰动线稳定，化学扰动线有强例子也有失败边界”。

## 当前正式主表

- datasets: 7
- aligned rho > 0.20: 6/7
- partial rho > 0.10: 6/7
- RC@80% positive: 7/7
- gene_main partial rho range: 0.328 to 0.474
- chem_robust partial rho range: -0.061 to 0.629

## 能写的 claim

1. 在 4 个 gene_main 数据集上，冻结协议控制 effect magnitude（效应大小）后仍保持正相关。
2. 在 Srivatsan 化学数据集上，化学线也有强信号。
3. 7/7 数据集 RC@80% 为正，说明“保留高可信预测”在应用层面有价值。

## 不能写的 claim

1. 不能写 SafeConf 对所有化学扰动都稳定有效。
2. 不能写信号完全独立于 effect magnitude；正式表显示 magnitude-only baseline 很强。
3. 不能因为 McFarland 失败而临时修改冻结公式。

## McFarland 定位

- v0.2 main aligned rho: -0.086
- v0.2 main partial rho: -0.061
- best observed score on McFarland: `learned_risk_score` (0.587)
- observed non-control cell_line × drug pairs in h5ad metadata: 1,175
- formal held-out test cell_line × drug pairs: 1,163

Decision: keep McFarland in the main table as a failure boundary. If it is revisited, first redefine tasks with dose/time, not formula tuning.

## Tahoe 定位

- downloaded: 84.9 GB
- obs rows: 100,648,790
- observed drug × cell_line pairs: 19,000
- pairs with at least 20 cells: 18,999
- role: external mega-scale validation candidate

Tahoe should not enter the current formal main table yet. The next useful step is a pseudobulk adapter and leakage audit.

## 下一步

1. Give Claude this diagnostics package for critique.
2. Do not change protocol v0.2 formula.
3. If time allows, build Tahoe pseudobulk adapter as external mega-scale validation.
4. For paper draft, write McFarland as an honest failure boundary.
