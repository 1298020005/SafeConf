# LOPO 稳健性实验：先看这个

## 1. 本次实验回答什么

本实验检查：只用现有两个预测器
`V0StrongBaseline` 和 `ContextSimBaseline` 的训练误差学习风险模型后，
能否给一个没有参加风险模型训练的第三预测器排序误差。

第三预测器包括：

- `PertMeanPredictor`：同一 perturbation 在训练 contexts 中的平均 effect。
- `Control1NNPredictor`：同一 perturbation 下，选择 control profile 最相似的一个训练 context。

这只能证明检索型预测器家族内部的任务风险迁移，不能证明对 GEARS、CPA
等任意深度模型都成立。

## 2. 运行信息

- 代码提交：`7de1358`
- 数据集：7 个正式数据集
- test score rows：82,512
- bootstrap：1,000 次，按 `task_key` 聚类抽样
- runtime 大表：`/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/`
- Git 证据包：当前目录
- frozen v0.2：未修改

## 3. 首要结果

### PertMean：LOPO pre_model_task_only

该分数不使用目标预测器的输出向量，也不使用 V0/ContextSim disagreement；
但仍使用 fold-local 历史 perturbation effects 和 target context control profile。

| dataset | partial rho | 95% CI |
|---|---:|---:|
| Cui | 0.566 | [0.523, 0.605] |
| Frangieh | -0.013 | [-0.113, 0.091] |
| Lara ex vivo | 0.864 | [0.835, 0.888] |
| Lara in vivo | 0.695 | [0.610, 0.760] |
| McFarland | 0.339 | [0.290, 0.392] |
| Santinha | 0.357 | [0.229, 0.466] |
| Srivatsan | 0.298 | [0.225, 0.373] |

结论：6/7 为正且 CI 下界大于 0；Frangieh 接近零，不应写成 7/7 成功。

### PertMean：LODO x LOPO full

风险模型同时看不到目标预测器误差和 held-out 数据集误差。

| dataset | partial rho | 95% CI |
|---|---:|---:|
| Cui | 0.370 | [0.321, 0.419] |
| Frangieh | 0.399 | [0.326, 0.471] |
| Lara ex vivo | 0.731 | [0.668, 0.779] |
| Lara in vivo | 0.600 | [0.516, 0.671] |
| McFarland | 0.201 | [0.145, 0.258] |
| Santinha | 0.272 | [0.157, 0.386] |
| Srivatsan | 0.644 | [0.604, 0.682] |

结论：7/7 partial rho 的 CI 下界大于 0，支持跨数据集、跨检索预测器的排序迁移。

## 4. 不能忽略的限制

1. `PertMeanPredictor` 与 ContextSim 的误差仍高度相关。
   Frangieh 和 Srivatsan 中，预测向量完全相同的比例分别为 54.66% 和 51.77%。
   因此不能宣传为“任意独立预测器”证据。
2. LODO x LOPO 的 aligned rho CI 为正是 6/7；Santinha aligned rho CI 跨 0，
   但其 magnitude-controlled partial rho CI 为正。
3. LODO x LOPO 的 AURC reduction CI 为正是 6/7；Santinha 跨 0。
4. LODO x LOPO 的 top-10 enrichment CI 下界大于 1 只有 4/7。
   Lara ex vivo、Lara in vivo、Santinha 的极端坏预测检索不稳定。
5. `pre_model_task_only` 是“目标预测器输出无关”，不是“零历史数据”。
   其中 historical residual、effect stability 和 effect variance 使用 fold-train 真效应。
6. Cui 的两个 OOD 特征全缺失，归一化后按中性值处理。

## 5. 泄漏审计

- normalization groups：140/140 为 `ok`
- test rows 用作 normalization reference：0
- provenance rows：48
- third predictor error 用于训练：0
- held-out dataset error 用于训练：0
- LODO held-out dataset 出现在 training datasets：0
- 训练预测器始终为：`V0StrongBaseline;ContextSimBaseline`

## 6. 文件入口

- [正式报告](LOPO_ROBUSTNESS_REPORT.md)
- [主结果与基线](LOPO_BASELINE_LADDER.csv)
- [1000 次 bootstrap CI](LOPO_BOOTSTRAP_CI.csv)
- [特征阶梯](LOPO_FEATURE_ABLATION.csv)
- [预测器相似性](LOPO_PREDICTOR_DIVERSITY.csv)
- [坏预测检索](LOPO_BAD_RETRIEVAL.csv)
- [归一化审计](LOPO_NORMALIZATION_AUDIT.csv)
- [训练来源审计](LOPO_TRAINING_PROVENANCE.csv)
- [运行状态](RUN_STATUS.json)
