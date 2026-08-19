# E115｜三正式数据集的实际分诊效用

E115 不重训预测器、不修改风险分数。它把 E108/E112 已冻结的三套 gene 数据测试任务转换为 risk–coverage 和 top-risk 资源分诊指标。

## 三数据集等权宏平均

| score | normalized AURC↓ | top-20% error enrichment↑ | reject-20% remaining error reduction↑ | top-20% total error capture↑ |
|---|---:|---:|---:|---:|
| SafeConf | 0.9746 | 1.1051 | 0.0269 | 0.2279 |
| model_disagreement | 0.9834 | 1.0917 | 0.0231 | 0.2249 |
| predicted_magnitude | 0.9846 | 1.0883 | 0.0223 | 0.2242 |

## 成对 bootstrap

正的 favorable delta 表示 SafeConf 更好。

| metric | comparator | favorable Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---:|
| normalized_aurc_50_100 | predicted_magnitude | 0.0100 | [-0.0003, 0.0216] | 0.967 |
| normalized_aurc_50_100 | model_disagreement | 0.0087 | [-0.0006, 0.0199] | 0.963 |
| top20_error_enrichment | predicted_magnitude | 0.0167 | [-0.0400, 0.0786] | 0.699 |
| top20_error_enrichment | model_disagreement | 0.0133 | [-0.0349, 0.0633] | 0.695 |
| reject20_remaining_error_reduction | predicted_magnitude | 0.0046 | [-0.0102, 0.0213] | 0.707 |
| reject20_remaining_error_reduction | model_disagreement | 0.0038 | [-0.0087, 0.0173] | 0.705 |
| top20_total_error_capture | predicted_magnitude | 0.0036 | [-0.0081, 0.0167] | 0.705 |
| top20_total_error_capture | model_disagreement | 0.0030 | [-0.0070, 0.0136] | 0.703 |

## 预设判定

- 通过：**否**。
- 点估计只有在 AURC 与 top-20% error capture 同时超过两个基线时才算方向一致。
- 还要求相对至少一个强基线的两个主效用指标区间均不跨 0。未通过时只保留描述性趋势。
