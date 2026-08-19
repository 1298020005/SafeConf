# E130｜Tian 第六数据确认性高错误路由

风险分数在读取 Tian 任务真值前写入 `tables/E130_RISK_SCORES_BEFORE_TRUTH.csv`。模型只用前五套历史数据。

| score | Spearman↑ | normalized AURC↓ | top-20% capture↑ | top-20% enrichment↑ |
|---|---:|---:|---:|---:|
| HighErrorRouter | 0.0449 | 0.9933 | 0.2115 | 1.0462 |
| SafeConf | 0.1342 | 0.9871 | 0.2109 | 1.0429 |
| model_disagreement | -0.0179 | 0.9977 | 0.2062 | 1.0199 |
| predicted_magnitude | 0.0667 | 0.9932 | 0.2064 | 1.0206 |

## Fold × perturbation-cluster bootstrap

正的 favorable delta 表示 HighErrorRouter 更好。

| metric | comparator | favorable Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---:|
| normalized_aurc_50_100 | SafeConf | -0.0062 | [-0.0190, 0.0071] | 0.199 |
| normalized_aurc_50_100 | predicted_magnitude | -0.0001 | [-0.0072, 0.0081] | 0.445 |
| normalized_aurc_50_100 | model_disagreement | 0.0044 | [-0.0039, 0.0137] | 0.844 |
| top20_total_error_capture | SafeConf | 0.0007 | [-0.0055, 0.0082] | 0.554 |
| top20_total_error_capture | predicted_magnitude | 0.0052 | [-0.0050, 0.0116] | 0.725 |
| top20_total_error_capture | model_disagreement | 0.0053 | [-0.0042, 0.0142] | 0.854 |

## 预设判定

- 通过：**否**。
- Tian 的 context 是技术批次，因此这里验证的是批次域偏移，不是新细胞类型泛化。
