# E30 GEARS seed-overlap feasibility audit

先看结论：E25 的 GEARS strict 包不能直接当作 seed-ensemble uncertainty benchmark，因为 47 个 unique task groups 中只有 5 个出现至少 2 次，只有 2 个出现 3 次。

可用信息：

- 重复任务内 true effect 最大差异 = 0.0
- repeated-task exploratory 表：`tables/E30_REPEATED_TASK_SEED_DIAGNOSTICS.csv`
- 全任务覆盖表：`tables/E30_TASK_SEED_COVERAGE.csv`
- 建议重跑清单：`tables/E30_RECOMMENDED_FIXED_SEED_RERUN_MANIFEST.csv`

正确口径：E30 是 feasibility / claim-control audit，不是性能提升结果。
