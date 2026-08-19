# 发给 Claude / Qoder：SafeConf 特征消融复核

请客观复核，不要默认同意 Codex。

## 背景

Qoder 指出：现有 ElasticNet（线性模型）系数、LODO（留一数据集组外部验证）、magnitude-only（只看效应大小）基线都不等价于经典 feature ablation（特征消融）。这个批评是对的。

Codex 已补跑 feature ablation audit（特征消融审计），路径：

```text
proj/docs/实验结果/Formal_main_20260604/feature_ablation_20260605/
```

核心文件：

```text
tables/FEATURE_ABLATION_SUMMARY.csv
tables/FEATURE_ABLATION_DELTA.csv
reports/FEATURE_ABLATION_AUDIT.md
```

## 请你重点看 4 个问题

### Q1. 这张表是否足以回答“每个特征有没有贡献”？

当前做法是：

- full：完整 v0.2 公式；
- loo_no_context：去掉 context_similarity（背景相似度）；
- loo_no_support：去掉 support_count（支持次数）；
- loo_no_disagreement：去掉 model_disagreement（模型分歧）；
- single-feature：单独使用每个核心特征。

请判断：这是否算合格的经典特征消融？还缺不缺 bootstrap CI（置信区间）或统计检验？

### Q2. model_disagreement 能不能作为主文最核心特征？

结果显示：

- 去掉 model_disagreement 后，7 个数据集 partial rho 平均下降约 0.157；
- `single_negative_disagreement_confidence` 的平均 partial rho 约 0.35，是最强单特征；
- McFarland 失败数据集上，negative disagreement 仍有弱正信号。

请判断：论文能否主张“model disagreement 是最稳定的 SafeConf 信号”？这个说法有没有过头？

### Q3. context_similarity / support_count 的不稳定应该怎么写？

结果显示：

- context_similarity 在 Cui、Lara exvivo 重要，但 Frangieh 上去掉后 partial rho 反而更高；
- support_count 在 Frangieh 重要，但 McFarland、Srivatsan 上去掉后更好。

请判断：这应该写成“辅助特征具有 dataset-specific（数据集特异性）贡献”，还是说明 v0.2 公式仍然太粗糙？

### Q4. 对稳二区 / 冲一区意味着什么？

请按审稿人视角判断：

- 这轮 feature ablation 是否补齐了一个关键短板？
- 它能否明显提高 Q2（二区）把握？
- 对 Q1（一区）还缺什么硬证据？

请不要只说“继续做”。如果你认为还缺最关键一步，请明确排优先级。

## Codex 当前初步判断

Codex 的初步判断如下，请你可以反驳：

1. Qoder 说缺经典 feature ablation 是对的，现在已补。
2. model_disagreement 是当前最稳的核心信号。
3. context_similarity 和 support_count 有贡献，但更依赖数据集结构。
4. McFarland 不应被删除，也不应为它改 frozen protocol（冻结协议）；它适合作 failure boundary（失败边界）。
5. 这轮结果增强 Q2 说服力，但 Q1 仍主要取决于 GEARS formal probe（GEARS 正式探测）和 predictor-agnostic（不绑定预测器）证据。

