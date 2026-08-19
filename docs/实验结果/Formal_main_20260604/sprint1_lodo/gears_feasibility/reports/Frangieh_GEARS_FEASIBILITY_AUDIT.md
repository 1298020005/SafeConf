# Frangieh GEARS Feasibility Audit

## 结论

Frangieh（黑色素瘤 CRISPR 扰动数据集）**可以作为 GEARS（图神经网络扰动预测模型）的小型适配器探针候选**，但现在还不能把它写成正式 GEARS 主结果。

原因很直接：

- perturbation（扰动）基本是 gene symbol（基因名），比 Cui 的 cytokine（细胞因子刺激）更适合 GEARS。
- 现有 SafeConf split（切分）已有 `633` 个 context × perturbation（背景×扰动）task（任务）。
- 但是目前还没有在 Frangieh 上训练并导出 GEARS 的 per-prediction records（逐条预测记录），所以只能说“格式上可尝试”，不能说“GEARS 证据已完成”。

## 关键数字

| 项目 | 数值 |
|---|---:|
| cells（细胞数） | 110188 |
| genes in panel（表达矩阵基因数） | 5124 |
| contexts（背景数） | 3 |
| non-control perturbations（非对照扰动数） | 211 |
| SafeConf tasks（已有任务数） | 633 |
| PredictionRecords（预测记录数） | 6330 |
| perturbation 与 var_names 精确重合数 | 211 |
| 看起来像基因名的 perturbation 数 | 209 |
| 最小非对照扰动细胞数 | 57 |

## 泄漏检查

- test pair seen in train（测试组合出现在训练中）: `0`
- test perturbation not seen in train（测试扰动训练中没见过）: `0`
- test context not seen in train（测试背景训练中没见过）: `0`

## 推荐下一步

1. 可以做 GEARS adapter smoke run（适配器冒烟测试），目标只是验证能导出 `PredictionRecord`。
2. 不要直接训练大规模 GEARS，也不要把它当主表结果。
3. 如果 smoke run 能导出 per-prediction GEARS predicted_effect（逐条 GEARS 预测效应），再把它接入 SafeConf 的 confidence scoring（可信度打分）。
