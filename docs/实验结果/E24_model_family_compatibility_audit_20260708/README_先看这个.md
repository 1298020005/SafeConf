# E24 model-family compatibility audit

先看结论：

- E23 manifest 有 120 个 task groups，但 perturbation 是 Hpoly/Salmonella 这类 stimulus/timecourse。
- 因此 E23 适合作为 PredictionRecord 合同 smoke，不适合直接作为 GEARS biological benchmark。
- `scgpt_env` 可以 import GEARS；本地有 Norman/Adamson/Dixit/Frangieh processed GEARS assets。
- 下一步应做 GEARS gene-perturbation strict smoke，而不是 GEARS-on-E23。

入口：

- HTML 报告：`reports/E24_MODEL_FAMILY_COMPATIBILITY_AUDIT.html`
- Markdown 报告：`reports/E24_MODEL_FAMILY_COMPATIBILITY_AUDIT_REPORT.md`
- GEARS candidate manifest：`tables/GEARS_COMPATIBLE_CANDIDATE_TASKS.csv`
