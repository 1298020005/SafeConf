# E29 GEARS–scGPT shared Adamson risk audit

先看结论：E29 将 Adamson fold-1 的全部 7 个可用 GEARS 单基因任务扩展成 GEARS/scGPT 双预测器 strict 合同，并新增任务级风险排序。

- PredictionRecords：14
- Tasks：7
- Genes：512
- strict issue：0

核心表：

- `tables/E29_TASK_RISK_SCORES.csv`：每个扰动的 GEARS/scGPT 误差、模型分歧、风险排序。
- `tables/E29_RISK_AUDIT_SUMMARY.csv`：风险信号与任务平均误差的相关性和 top-risk enrichment。
- `reports/E29_GEARS_SCGPT_SHARED_ADAMSON_RISK_AUDIT.html`：可视化报告。

边界：n=7，只能说明同合同下的风险审计流程跑通，不能作为正式性能 benchmark。
