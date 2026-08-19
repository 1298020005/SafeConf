# E125｜五正式数据集的实际分诊效用

Shifrut 与 Liang 均在 E115 三数据集效用分析后加入。风险分数未重拟合，五数据集和 folds 等权。

| score | normalized AURC↓ | top-20% enrichment↑ | reject-20% reduction↑ | top-20% error capture↑ |
|---|---:|---:|---:|---:|
| SafeConf | 0.9803 | 1.0746 | 0.0192 | 0.2221 |
| model_disagreement | 0.9891 | 1.0578 | 0.0146 | 0.2186 |
| predicted_magnitude | 0.9881 | 1.0614 | 0.0156 | 0.2193 |

| metric | comparator | favorable Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---:|
| normalized_aurc_50_100 | predicted_magnitude | 0.0078 | [-0.0019, 0.0177] | 0.938 |
| normalized_aurc_50_100 | model_disagreement | 0.0089 | [0.0016, 0.0167] | 0.993 |
| top20_error_enrichment | predicted_magnitude | 0.0132 | [-0.0252, 0.0545] | 0.727 |
| top20_error_enrichment | model_disagreement | 0.0168 | [-0.0178, 0.0506] | 0.828 |
| reject20_remaining_error_reduction | predicted_magnitude | 0.0036 | [-0.0064, 0.0146] | 0.733 |
| reject20_remaining_error_reduction | model_disagreement | 0.0046 | [-0.0044, 0.0136] | 0.837 |
| top20_total_error_capture | predicted_magnitude | 0.0028 | [-0.0051, 0.0115] | 0.731 |
| top20_total_error_capture | model_disagreement | 0.0036 | [-0.0035, 0.0107] | 0.834 |

## 预设判定

- 通过：**否**。
- E125 沿用 E115 的双主指标通过标准，不因新增数据改变阈值。
