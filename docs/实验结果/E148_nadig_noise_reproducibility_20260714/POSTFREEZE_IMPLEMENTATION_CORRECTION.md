# E148 冻结后实现纠正

第一次执行后复核发现，模型任务表除 held-out cell-line 任务外，还含 opposite fold 中 source cell-line 的 `random_seen_pair` 和 `perturbation_unseen` 诊断行。它们与已出现的 `context × perturbation` 生物任务重复；若纳入主估计，会把同一实验噪声记录计数两次。

按照冻结合同中的“背景×扰动任务”和按扰动整簇 bootstrap 原则，主分析限定为 `context_unseen` 与 `context_and_perturbation_unseen`，使每个 `context × perturbation` 只保留一次。source-context 行继续完整落盘到 `tables/E148_ALL_MODEL_AND_NOISE_DIAGNOSTICS.csv`，但不进入主相关或置信区间。

本次纠正没有改动风险分数、表达噪声指标、任务选择、阈值、方向模型或通过规则；纠正原因和第一次执行的行数均保留在运行状态中。
