# E21 strict contract remediation smoke

先看结论：

- E21 不替换 E17 正式结果，只做一个小型 strict contract 修法验证。
- 从 E17 抽样 60 条记录、30 个任务组。
- 将 `true_effect_key` 改为 task-scoped 后，strict validator 状态：`pass`。
- 这说明下一轮 full rerun / shared benchmark adapter 应采用 task-scoped true effect。

入口：

- HTML 报告：`reports/E21_STRICT_CONTRACT_REMEDIATION.html`
- Markdown 报告：`reports/E21_STRICT_CONTRACT_REMEDIATION_REPORT.md`
- 流程图：`figures/strict_contract_remediation_smoke.svg`
- 小型 strict bundle：`input/PREDICTION_RECORDS.csv`
