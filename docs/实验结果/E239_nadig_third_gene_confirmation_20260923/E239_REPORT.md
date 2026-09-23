# E239｜第三批未用基因：强简单基线确认

预定主判定：**通过**；两折、332 行、128 个此前未用基因。

| 分数 | 混合任务折宏平均 Spearman |
|---|---:|
| directional_risk_frozen | 0.759 |
| magnitude | 0.367 |
| novelty | 0.445 |
| disagreement | 0.292 |
| magnitude_plus_novelty | 0.533 |
| source_seen | 0.728 |
| source_plus_magnitude_plus_novelty | 0.712 |

冻结分数 − 背景见过+幅度+新颖度：+0.047，基因簇95%区间 [+0.026, +0.070]。

## 必须同时看的适用边界

目标细胞系留出内部以及未中心化 cosine、绝对 RMSE 的完整数字见 `E239_MANDATORY_SENSITIVITY.csv`。即便混合队列主门通过，也不可称为整细胞系留出内部优于幅度或所有错误定义都有效。

本轮与 E136/E237 基因完全不重叠，但仍来自 Nadig 同一研究。E112 图构建器提前物化测试目标，故不宣称首次读取真值前严格盲测。
