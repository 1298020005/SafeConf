# E31 GEARS fixed-test split smoke

先看结论：GEARS exporter 已能通过 `--test-perturbations-file` 固定 Adamson test perturbations。E31 使用 E29 的 7 个任务跑 1 epoch smoke，并导出 strict PredictionRecord。

这不是性能 benchmark。它是为后续固定任务三 seed 重跑铺路。
