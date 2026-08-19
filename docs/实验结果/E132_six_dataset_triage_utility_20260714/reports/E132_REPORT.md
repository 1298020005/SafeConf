# E132｜六正式数据集的分诊效用

E132 在 Tian 解封后追加，属于完整性描述；E127/E130 才是 Tian 事前冻结的路由器检验。

| score | normalized AURC↓ | top-20% enrichment↑ | reject-20% reduction↑ | top-20% capture↑ |
|---|---:|---:|---:|---:|
| SafeConf | 0.9814 | 1.0693 | 0.0178 | 0.2203 |
| model_disagreement | 0.9906 | 1.0515 | 0.0130 | 0.2165 |
| predicted_magnitude | 0.9889 | 1.0546 | 0.0139 | 0.2172 |

| metric | comparator | favorable Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---:|
| normalized_aurc_50_100 | predicted_magnitude | 0.0075 | [-0.0006, 0.0158] | 0.965 |
| normalized_aurc_50_100 | model_disagreement | 0.0091 | [0.0029, 0.0159] | 0.998 |
| top20_error_enrichment | predicted_magnitude | 0.0147 | [-0.0184, 0.0494] | 0.796 |
| top20_error_enrichment | model_disagreement | 0.0178 | [-0.0120, 0.0463] | 0.883 |
| reject20_remaining_error_reduction | predicted_magnitude | 0.0039 | [-0.0047, 0.0132] | 0.800 |
| reject20_remaining_error_reduction | model_disagreement | 0.0048 | [-0.0029, 0.0125] | 0.889 |
| top20_total_error_capture | predicted_magnitude | 0.0031 | [-0.0038, 0.0104] | 0.799 |
| top20_total_error_capture | model_disagreement | 0.0038 | [-0.0024, 0.0098] | 0.887 |
