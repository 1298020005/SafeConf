# E65｜scGPT 正式微调与 E60 固定任务对齐

先读 `reports/E65_REPORT.md`。

本实验淘汰了旧的 scGPT forward-only smoke：scGPT 在 E60 同一批 24 个 held-out genes 上做了官方 tutorial 风格的 MSE fine-tuning，并和 GEARS ensemble 使用同一 512-gene order 与同一 true effect。
