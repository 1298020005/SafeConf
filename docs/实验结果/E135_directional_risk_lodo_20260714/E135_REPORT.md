# E135｜方向误差风险的留一数据集探索

## 结果

Ridge(alpha=10) 每次只在其余五个数据集拟合，在完全留出的第六个数据集打分。六数据集等权宏平均：centered Pearson **ρ=0.346**，centered cosine **ρ=0.336**；六个留出数据集的 fold 宏平均均为正。

按 perturbation 整簇重采样，Pearson 方向风险的 95% CI 为 **[0.306, 0.380]**，cosine 为 **[0.297, 0.370]**。

## 留出数据集结果

| held-out dataset | Pearson ρ | cosine ρ | combined rank ρ |
|---|---:|---:|---:|
| Frangieh | 0.632 | 0.625 | 0.628 |
| Lara_exvivo | 0.517 | 0.516 | 0.515 |
| Liang | 0.115 | 0.116 | 0.116 |
| Santinha | 0.315 | 0.334 | 0.324 |
| Shifrut | 0.231 | 0.256 | 0.244 |
| Tian_CRISPRi | 0.265 | 0.172 | 0.236 |

## 解释

原 SafeConf 的固定结构项针对 absolute RMSE。方向误差需要单独的轻量风险头；四个输入仍全部是部署时可见量，没有使用目标任务真值。Ridge 系数允许结构项改变方向，避免把一种误差定义的先验硬套到另一种误差定义。

## 证据边界

本轮先看过五类候选模型，属于探索性模型选择。完整候选结果全部保存在表中，没有只保留最好模型。冻结文件 `E135_FROZEN_DIRECTION_MODEL.json` 只能在新的第七数据集上确认，不能把 LODO 结果伪装成未见数据确认。

## 冻结系数

`intercept=0.485562`；`risk_disagreement_z=0.015638`；`predicted_magnitude_z=0.023463`；`context_novelty_scaled=-0.194864`；`perturbation_novelty=0.131729`
