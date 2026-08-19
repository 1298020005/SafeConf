# E11 Selective prediction / risk-coverage 审计报告

生成时间：2026-07-07 05:55

## 核心结论

现有 SafeConf risk-coverage 曲线显示：当只保留低风险预测、拒绝最高风险的 20% 预测时，7/7 个数据集的平均 RMSE 下降；宏平均 improvement = 16.67%，中位数 improvement = 12.83%。

保留 50% 低风险预测时，宏平均 improvement = 21.72%，但这更像高强度分诊，实际 wet-lab 场景要结合复核成本。

McFarland 是边界：80% coverage improvement = 3.95%，50% coverage improvement = -2.16%。论文中需要把它保留为 failure/boundary case。

## 投稿价值

E11 把问题从“风险分数与误差是否相关”推进到“模型在有限覆盖率下是否能降低平均错误”。这更接近 selective prediction，也更适合作为 CCF-A 方法升级的入口。

当前仍缺 formal risk guarantee。下一步如果要冲 CCF-A，需要引入 calibration split，对每个 coverage / risk budget 给出可复现阈值，并报告 held-out risk control 是否满足目标。

## 自动生成图

- `figures/E11_fig1_risk_coverage_curves.svg`
- `figures/E11_fig2_improvement_at_80pct_coverage.svg`
- `figures/E11_fig3_macro_selective_gain.svg`
