# E31 GEARS fixed-test split smoke

生成时间：2026-07-08T13:21:47

## 结论

E31 验证 GEARS exporter 的 `--test-perturbations-file` 能把 Adamson test split 固定到 E29 的 7 个任务。

- GEARS returncode：0
- GEARS status：ok
- PredictionRecords：7
- strict issue_count：0
- all expected observed：True

边界：这是 1-epoch smoke，不是 GEARS 性能，也不是 seed-uncertainty 正式结果。它只证明 E32/E33 可以在固定任务上做三 seed 重跑。
