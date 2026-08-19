# Cross-context perturbation prediction confidence scoring 可行性判断

生成时间：2026-05-21 16:43:37

## 1. 当前服务器资源是否足够跑一个最小 MVP？

足够。CPU、内存、磁盘都支持。MVP 不需要 GPU。

建议第一版只做 CPU：生成 PredictionRecord、计算 confidence feature、跑相关性和 risk-coverage。这样更贴合导师建议，也避免继续陷入“换模型但问题不清楚”。

## 2. 当前已有数据是否足够做 confidence scoring？

足够做第一版 MVP。

优先顺序：

1. `KaggleCrossCell`：最适合第一个版本。
2. `Haber`：小，便于快速复核。
3. `Parekh`：小，便于快速复核。
4. `KaggleCrossPatient`：可作为外部或跨 patient/context 检查。

不建议一开始用 Replogle/Tian/TCDD 这类大数据硬冲，因为现在要先证明 confidence task 的定义成立。

## 3. 当前已有结果是否能直接复用？

能部分复用。

- 已有 `SAFETY_TASK_METRICS.csv`：有 `model`、`rmse`、`confidence`、`unsafe_flag`。
- 已有 `RISK_COVERAGE.csv`：可直接看 coverage-RMSE。
- 已有 `SAFE_UNSAFE_CONTRAST.csv`：可直接看 safe vs unsafe。

但它们缺少完整 `PredictionRecord` 所需字段：`context`、`perturbation`、`true_effect`、`predicted_effect`、`confidence_feature` 明细。因此只能做“初步验证”，还不能作为新任务最终证据。

## 4. 如果只用 V0 predictor，能不能快速生成 PredictionRecord？

能。V0 已在 `03_code/transport_models.py:37-62` 实现，任务构造在 `03_code/build_context_splits.py:76-130`。

## 5. held-out context-perturbation pair split 是否已有？

没找到。

当前 `build_context_splits.py:133-160` 只有：

- `leave_context`
- `heldout_perturbation`

没有发现正式的 `heldout_pair` 或 `heldout_context_perturbation_pair` split。

## 6. 如果没有，如何最小代价新增？

新增一个轻量 split 生成逻辑即可，不需要训练深度模型：

- 每个 observed pair 是一格 `(context, perturbation)`。
- test 选一部分 pair。
- train 保留同一个 context 的其他 perturbation，也保留同一个 perturbation 的其他 context。
- 这样模型不能直接见过这个 pair，但能分别见过这个 context 和 perturbation。
- 这个 split 正好对应问题：“这个扰动在这个新细胞背景里能不能信？”

## 7. 第一版应该怎么选？

- 数据集：`KaggleCrossCell`。
- split：优先新增 `heldout_context_perturbation_pair`；暂时不改代码时可先用 `leave_context`。
- predictor：先用 V0 和 ContextSimBaseline，不建议一开始上深度模型。
- confidence feature：support_count、context_similarity、perturbation_stability、expert_disagreement、effect_norm、nearest_context_distance。

## 8. 第一版应该输出哪些图和表？

表：`prediction_record.csv`、`confidence_baseline_comparison.csv`、`risk_error_correlation.csv`、`risk_coverage_summary.csv`、`safe_unsafe_rmse.csv`、`dataset_split_summary.csv`。

图：confidence vs error scatter、risk-coverage curve、high/medium/low confidence RMSE barplot、context × perturbation matrix、support_count vs RMSE、safe vs unsafe RMSE。

## 9. 结论

这个新方向是可行的，而且比“继续换模型”更清楚：核心不再是我能不能预测得最好，而是我能不能告诉老师/审稿人：什么时候预测结果可信，什么时候应该谨慎。
