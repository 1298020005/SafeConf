# 发给 Claude / Qoder：特征消融 1000 Bootstrap 复核

请客观复核，不要默认同意 Codex。

## 背景

之前 Qoder 指出：SafeConf 缺经典 feature ablation（特征消融）表。  
Codex 先补了 200 bootstrap 版，现在已升级到 1000 bootstrap。

路径：

```text
proj/docs/实验结果/Formal_main_20260604/feature_ablation_1000_20260606/
```

核心文件：

```text
tables/FEATURE_ABLATION_SUMMARY.csv
tables/FEATURE_ABLATION_DELTA.csv
reports/FEATURE_ABLATION_AUDIT.md
```

## 关键结果

完整 v0.2 partial rho + 95% CI：

| 数据集 | partial rho | 95% CI |
|---|---:|---:|
| CuiHacohen2023 | 0.328 | [0.291, 0.364] |
| Frangieh | 0.474 | [0.437, 0.512] |
| Lara exvivo | 0.430 | [0.366, 0.491] |
| Lara invivo | 0.358 | [0.296, 0.419] |
| McFarland | -0.061 | [-0.102, -0.021] |
| Santinha | 0.224 | [0.141, 0.298] |
| Srivatsan | 0.629 | [0.594, 0.660] |

特征消融平均变化：

| ablation（消融） | mean delta partial |
|---|---:|
| 去掉 model_disagreement | -0.157 |
| 去掉 context_similarity | -0.024 |
| 去掉 support_count | -0.015 |

## 请重点回答

1. 1000 bootstrap 后，是否足以回答审稿人“每个特征有没有贡献”的问题？
2. 是否可以把 model_disagreement（模型分歧）写成 SafeConf 最稳定核心信号？
3. context_similarity（背景相似度）和 support_count（支持次数）是否只能写成辅助信号？
4. McFarland 的负 CI 是否足以把它定为 failure boundary（失败边界）？
5. 对稳二区/冲一区，下一步最缺的是不是 predictor-agnostic（不绑定预测器）证据，而不是继续堆 feature ablation？

## Codex 当前判断

1. Qoder 之前指出的特征消融短板已经补上。
2. model_disagreement 是当前最稳的核心特征。
3. McFarland 失败边界已经被 1000 bootstrap 固定，不应硬救。
4. 下一步更该考虑：主表图、claim 收缩、以及真正统一契约下的第三 predictor，而不是继续硬接当前 GEARS。

