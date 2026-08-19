# E22 generator strict smoke

先看结论：

- 修改后的 `run_confidence_mvp_v2_1.py` 新生成 Haber 200-gene smoke。
- 输出 240 条 PredictionRecord，120 个任务组。
- strict validator 状态：`pass`，issue_count = 0。
- 这不是生物学性能新结论，只是生成器合同修复验证。

入口：

- HTML 报告：`reports/E22_GENERATOR_STRICT_SMOKE.html`
- Markdown 报告：`reports/E22_GENERATOR_STRICT_SMOKE_REPORT.md`
- 流程图：`figures/generator_strict_smoke.svg`
