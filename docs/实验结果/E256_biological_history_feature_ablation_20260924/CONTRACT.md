# E256：真实生物历史特征的分组消融

日期：2026-09-24。基于已公开的 E1–E4 矩阵作回顾性开发审计，不是新的盲测。

## 固定问题

同一折中仅用 V0StrongBaseline 与 ContextSimBaseline 的 train/val 错误作为监督标签；只在该折从未给模型看的 PertMeanPredictor test 任务评估。同一 `(dataset, task_key)` 不得在来源训练和目标测试交叉。七个原始数据集全部纳入，不按结果筛数据集。

在相同来源误差标签、相同小模型、相同超参数和相同测试任务下，固定四组输入：

- A：当前预测的四个幅度/形态量、两个预测器分歧量，加任务背景的 context/support/OOD 五项。其中 `prediction_norm_ratio` 和 `prediction_magnitude_deviation` 需用训练折**全局真实效应长度中位数**作尺度，A 不是完全不接触任何训练历史；它只不含扰动专属的 B/C 生物历史摘要；
- A+B：A 加 `historical_residual_risk`。源码定义是：用训练来源背景的真实效应均值预测另一个训练背景的真实效应，再取误差中位数。因此 B 是**真实生物历史的跨背景可迁移性**，绝不是过去预测模型的误差；
- A+C：A 加 `perturbation_effect_stability` 和 `perturbation_effect_variance`，这是训练来源的**真实扰动效应**相似性/离散度；
- A+B+C：以上合并。

所有组都用过去两个参考预测器的错误作为**监督标签**。B 和 C 均为真实效应历史的不同摘要；本矩阵没有单独的“过去预测模型错误”输入特征。本实验不能分离“过去模型错误标签训练”的贡献，只能检验 B/C 这三种纯生物历史输入是否提供增量。C 只检验这两个原始特征，不能外推到所有可能的生物历史表示。

固定模型：`HistGradientBoostingRegressor(max_iter=60, max_depth=3, learning_rate=0.05, min_samples_leaf=20, l2_regularization=0.1, early_stopping=False, random_state=256)`。原始数值列输入，NaN 交由模型处理；不使用真值派生的 `true_effect_l2_norm`。主要量：PertMean test 错误的 Spearman 与 top-20% 归一化复核效用；按数据集报告五折配对均值和正向折数。缺失率同时报告。不得把后验最强数据集改称预注册适用域。

本实验之后，如 B/C 有局部好处，要在新的独立研究/预先固定子场景上确认；若没有，停止“历史真实生物证据已证明”的主张。源码依据：`safetrans_confidence/cli/run_fast_feature_scoring.py` 中 `_historical_residual` 和 `compute_evaluation_diagnostics_from_records`。
