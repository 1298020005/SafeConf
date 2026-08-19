# E9 强基线统一审计报告

生成时间：2026-07-07 05:55

## 核心结论

E9 的结果给论文定位划了边界：Frozen v0.2 不能被包装成稳定超过 magnitude 的方法。七主数据集中，Frozen v0.2 直接按 aligned rho 对比 magnitude 只赢 2/7，按 AURC reduction 只赢 2/7。

这并不否定 SafeConf。E2 显示 7/7 个数据集在控制 magnitude 后仍有正的 residual signal。更稳的论文主张是：SafeConf 提供 magnitude 之外的风险信息，并可用于 risk triage / selective verification。

Tahoe chemical 是必须保留的边界：top-10 enrichment 中，magnitude = 6.49，SafeConf full = 4.88，combined 75% magnitude = 6.26。这说明 chemical 场景中 magnitude 是主导强基线，SafeConf 更适合作为补充信号或失败边界解释。

## 对投稿叙事的影响

1. 摘要中不要写“SafeConf outperforms magnitude across datasets”。
2. 可以写“SafeConf adds residual risk information beyond magnitude in seven benchmark datasets”。
3. 主图中要同时放 magnitude、SafeConf、combined 和 oracle/reference。
4. Tahoe chemical 放入 Results 的 boundary subsection，语气诚实；这会比藏结果更抗审稿。
5. 下一轮实验要围绕 selective prediction / risk budget 展开，让任务从“谁相关更高”转到“有限复核预算下谁更有用”。

## 自动生成图

- `figures/E9_fig1_frozen_vs_magnitude_aligned_delta.svg`
- `figures/E9_fig2_incremental_value_after_magnitude_control.svg`
- `figures/E9_fig3_tahoe_top10_boundary.svg`

## 自动生成表

- `tables/E9_DEPLOYABLE_BASELINE_LADDER.csv`
- `tables/E9_FROZEN_VS_MAGNITUDE_HEAD_TO_HEAD.csv`
- `tables/E9_DEPLOYABLE_WINNER_TABLE.csv`
- `tables/E9_E2_INCREMENTAL_VALUE.csv`
- `tables/E9_TAHOE_CHEMICAL_BOUNDARY.csv`
