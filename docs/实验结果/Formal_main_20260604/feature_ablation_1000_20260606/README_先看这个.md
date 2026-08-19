# SafeConf 特征消融 1000 Bootstrap

更新时间：2026-06-06

## 一句话结论

feature ablation（特征消融）已经从 200 bootstrap 升级到 1000 bootstrap。  
结论和 200 版一致，没有翻车：

> `model_disagreement（模型分歧）` 仍然是最稳定、最关键的核心信号。

## 为什么要做这一步

Qoder 指出得对：  
ElasticNet（线性模型）系数、LODO（留一数据集组外部验证）和 magnitude-only（只看效应大小）不能替代经典 feature ablation（特征消融）。

审稿人可能会问：

> 你说 context similarity（背景相似度）、support count（支持次数）、model disagreement（模型分歧）都有用，那拿掉其中一个会怎样？

这次就是正式回答这个问题。

## 1000 bootstrap 主结果

完整 v0.2 公式的 partial rho（控制效应大小后的相关）：

| 数据集 | partial rho | 95% CI |
|---|---:|---:|
| CuiHacohen2023 | 0.328 | [0.291, 0.364] |
| Frangieh | 0.474 | [0.437, 0.512] |
| Lara exvivo | 0.430 | [0.366, 0.491] |
| Lara invivo | 0.358 | [0.296, 0.419] |
| McFarland drug-only | -0.061 | [-0.102, -0.021] |
| SantinhaPlatt2023 | 0.224 | [0.141, 0.298] |
| Srivatsan sciplex3 | 0.629 | [0.594, 0.660] |

## 消融结论

平均来看：

| 去掉什么 | partial rho 平均变化 | 解释 |
|---|---:|---|
| 去掉 model_disagreement（模型分歧） | -0.157 | 最伤，说明它是核心特征 |
| 去掉 context_similarity（背景相似度） | -0.024 | 有用但不稳定 |
| 去掉 support_count（支持次数） | -0.015 | 有用但数据集依赖很强 |

单特征来看：

| 单独使用什么 | 平均表现 |
|---|---|
| negative model disagreement（负模型分歧） | 最强，甚至平均略高于完整公式 |
| support_count（支持次数） | 中等 |
| context_similarity（背景相似度） | 最弱且不稳定 |

## 对论文意味着什么

可以更稳地说：

> SafeConf 最核心的可解释信号是 model disagreement（模型分歧）：当两个 predictor（预测器）对同一个 perturbation effect（扰动效应）预测差得越大，这次预测越可能不可靠。

但不能说：

> 三个特征在所有数据集上都同样有效。

更准确的说法是：

> model disagreement 是最稳定核心；context similarity 和 support count 是辅助信号，会受数据结构影响。

## McFarland 怎么处理

McFarland 的 1000 bootstrap CI 仍然为负：

```text
partial rho = -0.061, CI = [-0.102, -0.021]
```

所以它不是“随机没跑好”，而是真正的 failure boundary（失败边界）。  
不要删除，也不要为了它改冻结公式。

## 关键文件

- 主表：`tables/FEATURE_ABLATION_SUMMARY.csv`
- 去特征变化：`tables/FEATURE_ABLATION_DELTA.csv`
- 按 predictor（预测器）拆分：`tables/FEATURE_ABLATION_PER_PREDICTOR.csv`
- 报告：`reports/FEATURE_ABLATION_AUDIT.md`
- 状态：`RUN_STATUS.json`

