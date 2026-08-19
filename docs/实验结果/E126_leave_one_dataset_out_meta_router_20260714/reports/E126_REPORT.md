# E126｜跨数据集风险路由器

每次整套留出一个数据集，只用其余四套历史数据拟合正系数 Ridge。留出数据集的任务真值不参与拟合、特征秩变换或阈值选择。

## 五数据集等权结果

| score | Spearman↑ | normalized AURC↓ | top-20% enrichment↑ | reject-20% reduction↑ | top-20% capture↑ |
|---|---:|---:|---:|---:|---:|
| MetaSafeConf_LODO | 0.2073 | 0.9811 | 1.0698 | 0.0179 | 0.2211 |
| SafeConf | 0.2180 | 0.9803 | 1.0746 | 0.0192 | 0.2221 |
| model_disagreement | 0.0625 | 0.9891 | 1.0578 | 0.0146 | 0.2186 |
| predicted_magnitude | 0.0978 | 0.9881 | 1.0614 | 0.0156 | 0.2193 |

## 聚类 bootstrap

正的 favorable delta 表示 MetaSafeConf 更好。

| metric | comparator | favorable Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---:|
| spearman | SafeConf | -0.0106 | [-0.0368, 0.0106] | 0.204 |
| spearman | predicted_magnitude | 0.1095 | [0.0156, 0.1921] | 0.988 |
| spearman | model_disagreement | 0.1448 | [0.0827, 0.2089] | 1.000 |
| normalized_aurc_50_100 | SafeConf | -0.0008 | [-0.0034, 0.0011] | 0.236 |
| normalized_aurc_50_100 | predicted_magnitude | 0.0069 | [-0.0017, 0.0161] | 0.933 |
| normalized_aurc_50_100 | model_disagreement | 0.0080 | [0.0010, 0.0155] | 0.987 |
| top20_total_error_capture | SafeConf | -0.0010 | [-0.0051, 0.0022] | 0.313 |
| top20_total_error_capture | predicted_magnitude | 0.0018 | [-0.0042, 0.0083] | 0.704 |
| top20_total_error_capture | model_disagreement | 0.0026 | [-0.0031, 0.0082] | 0.814 |

## 定位

E126 是在 E125 后冻结的方法改进；它证明的是跨项目历史校准是否可行，不是未来数据上的事前确认。无论结果方向如何，下一步均需用冻结后的同一实现测试第六套未见数据。
