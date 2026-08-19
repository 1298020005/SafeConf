# SafeConf 特征消融审计

更新时间：2026-06-05

## 一句话结论

Qoder 说得对：之前已有 ElasticNet（线性模型）系数、LODO（留一数据集组外部验证）和 magnitude-only（只看效应大小）基线，但还缺一张经典 feature ablation（特征消融）表。

这次已经补上：对 7 个主表数据集重新计算 frozen protocol v0.2（冻结的 v0.2 公式），并分别去掉一个特征，看 partial rho（控制效应大小后的相关）掉不掉。

## 这次到底测了什么

每个数据集都测 7 类分数：

| 分数 | 中文意思 | 用途 |
|---|---|---|
| `protocol_full_confidence` | 完整 v0.2 可信度分数 | 主方法原始结果 |
| `loo_no_context_confidence` | 去掉 context_similarity（背景相似度） | 看背景相似度有没有贡献 |
| `loo_no_support_confidence` | 去掉 support_count（支持次数） | 看训练里类似例子数量有没有贡献 |
| `loo_no_disagreement_confidence` | 去掉 model_disagreement（模型分歧） | 看两个预测器意见不一致是否关键 |
| `single_context_similarity_confidence` | 只用背景相似度 | 单特征基线 |
| `single_support_count_confidence` | 只用支持次数 | 单特征基线 |
| `single_negative_disagreement_confidence` | 只用负模型分歧 | 单特征基线 |

`loo` 是 leave-one-out（拿掉一个）的意思。  
这里没有训练新 perturbation predictor（扰动预测器），只是重算 confidence score（可信度分数）。

## 最重要结果

### 1. model_disagreement 是最稳的核心特征

去掉 `model_disagreement（模型分歧）` 后，7 个数据集的 partial rho 平均下降约 `0.157`。

最明显的例子：

| 数据集 | 完整 partial rho | 去掉 disagreement 后 | 下降 |
|---|---:|---:|---:|
| Srivatsan sciplex3 | 0.629 | 0.286 | -0.343 |
| Lara exvivo | 0.430 | 0.217 | -0.213 |
| Frangieh | 0.474 | 0.310 | -0.164 |
| Lara invivo | 0.358 | 0.221 | -0.137 |

这说明 SafeConf 最硬的信号不是“公式凑出来的”，而是很直观的一件事：

> 两个 predictor（预测器）对同一道题意见差得越大，这道题越可能不靠谱。

### 2. context_similarity 和 support_count 是数据集依赖的

`context_similarity（背景相似度）` 在 Cui 和 Lara exvivo 上很重要，但在 Frangieh 上拿掉后 partial rho 反而变高。

`support_count（支持次数）` 在 Frangieh 上很重要，但在 McFarland 和 Srivatsan 上拿掉后反而更好。

这说明：

> v0.2 公式不是每个特征在每个数据集都同向有效。论文里应该说“核心稳定信号是 model disagreement；context/support 是辅助信号，会随数据结构变化”。

### 3. McFarland 的失败更清楚了

McFarland 完整 v0.2 partial rho 是 `-0.061`。  
去掉 support_count 后变成 `0.084`，只用 negative disagreement 也有 `0.084`。

这支持之前判断：

> McFarland 不是完全没有 confidence signal（可信度信号），而是 v0.2 的 chem_robust（药物稳健线）特征组合不适合它。它应保留为 failure boundary（失败边界），不要为了救它临时改公式。

## 对 Qoder 问题的回答

Qoder 说“缺经典 feature ablation 表”，这个批评成立。  
现在这张表已经补了。

但这不等于论文已经稳一区。它解决的是一个具体审稿问题：

> 如果审稿人问“每个特征到底有没有独立贡献”，现在可以回答。

## 关键文件

- 审计报告：`reports/FEATURE_ABLATION_AUDIT.md`
- 主表：`tables/FEATURE_ABLATION_SUMMARY.csv`
- 去特征后的变化：`tables/FEATURE_ABLATION_DELTA.csv`
- 按 predictor（预测器）拆分：`tables/FEATURE_ABLATION_PER_PREDICTOR.csv`
- 输入状态：`tables/FEATURE_ABLATION_INPUT_STATUS.csv`
- 公式说明：`tables/FEATURE_ABLATION_FORMULAS.csv`

## 当前判断

- 对二区：这是重要补强，能回答“特征是否真的有用”的审稿问题。
- 对一区：还不够。一区更需要 GEARS（主流深度预测器）完整结果、更多 predictor（预测器）泛化证据，或者更强的生物学解释。

