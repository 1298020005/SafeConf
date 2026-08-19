# E23 shared benchmark adapter workbench

先看结论：

- E23 不是新的模型结果，而是 model-specific validation 的 shared benchmark 地基。
- 已生成 `input/SHARED_BENCHMARK_TASK_MANIFEST.csv`，共 120 个 task groups。
- manifest 检查：5/5 pass。
- 现在可以开始写 GEARS/scGPT/CPA adapter，但不能说三模型统一验证已经完成。

入口：

- HTML 报告：`reports/E23_SHARED_BENCHMARK_ADAPTER_WORKBENCH.html`
- Markdown 报告：`reports/E23_SHARED_BENCHMARK_ADAPTER_WORKBENCH_REPORT.md`
- 任务 manifest：`input/SHARED_BENCHMARK_TASK_MANIFEST.csv`
- 输出 schema：`tables/ADAPTER_REQUIRED_OUTPUT_SCHEMA.csv`
