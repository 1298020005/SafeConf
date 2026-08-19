# E29 GEARS–scGPT shared Adamson risk audit

生成时间：2026-07-08T12:56:33

## 先看结论

E29 把 E25 中 Adamson fold-1 的 7 个单基因任务全部放进 GEARS–scGPT 同任务合同中。

- PredictionRecords：14
- Tasks：7
- Genes：512
- strict issue_count：0

最重要的变化：E29 不只检查格式，还计算了 GEARS 与 scGPT 在同一任务上的预测分歧，并把这个分歧作为可部署风险信号，与真实误差做小样本相关性检查。

边界也很清楚：n=7，只能作为严格合同下的风险审计 smoke，不能作为正式模型优劣结论。
