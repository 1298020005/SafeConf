# E64｜GEARS 与 source-only 全局效应基线

## 设计

E60 的三 seed GEARS 分歧没有风险信号。这里保留完全相同的 24 个固定未见基因任务和 GEARS ensemble，将第二预测器改为训练条件中所有非 control 扰动 effect 的平均值。这个 baseline 的计算时删除了全部 24 个 held-out conditions。

因此 `risk_gears_globalmean_disagreement` 在打分时只使用 GEARS ensemble 向量和训练域全局 effect；真实 held-out effect 只在最后计算 GEARS RMSE。

## 结果

| score | 可部署 | ρ(score, GEARS RMSE) | bootstrap 95% CI | top20 高误差富集 |
|---|---|---:|---:|---:|
| risk_gears_globalmean_disagreement | 是 | -0.243 | [-0.649, 0.214] | 0.608 |
| risk_gears_predicted_magnitude | 是 | 0.095 | [-0.275, 0.424] | 0.929 |
| true_l2_diagnostic | 否（oracle） | 0.943 | — | — |

## 边界

这不是第二个深度预测器，也不代表 scGPT/CPA 的结果。它是一个明确、可复算的独立基线，用来检测“GEARS 偏离训练域平均效应”是否与 GEARS 自身错误相关。

## 文件

- 任务表：`tables/E64_TASK_TABLE.csv`
- 汇总：`tables/E64_RISK_ERROR_SUMMARY.csv`
- 图：`figures/F1_gears_baseline_disagreement_vs_error.svg`
