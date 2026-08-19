# 发给 Claude / Qoder：GEARS 正式探测复核

请客观复核，不要默认同意 Codex。

## 背景

GEARS（图神经网络扰动预测模型）已经在 Frangieh（黑色素瘤 CRISPR 扰动数据集）上完成正式探测：

```text
proj/docs/实验结果/Formal_main_20260604/gears_frangieh_formal_probe_20260606/
```

核心文件：

```text
tables/GEARS_PREDICTION_RECORD_STATUS.csv
tables/GEARS_CONFIDENCE_EVAL_SUMMARY.csv
tables/GEARS_RISK_COVERAGE.csv
reports/GEARS_CONFIDENCE_EVAL_REPORT.md
```

## 当前结果

GEARS 预测本身：

- seed 1/2/3 都跑完；
- 每个 seed 20 epoch；
- 共 62 条 PredictionRecords（逐条预测记录）；
- test Pearson 大约 0.996；
- DE Pearson 大约 0.927 到 0.945。

confidence scoring（可信度打分）：

| score | aligned rho | RC@80% |
|---|---:|---:|
| `gears_prediction_magnitude_risk` | 0.894 | 14.81% |
| `gears_uncertainty_confidence` | 0.096 | -0.82% |

## 请重点回答

### Q1. GEARS 这条证据应该怎么定位？

它能否支持 predictor-agnostic（不绑定预测器）叙事？

候选口径：

1. 主文证据；
2. supplement（补充证据）；
3. 只作为 adapter feasibility（适配器可行性），暂不当强结果。

请明确选一个。

### Q2. GEARS native uncertainty 很弱，这是不是坏消息？

`gears_uncertainty_confidence` 的 aligned rho 只有 0.096，RC@80% 还是负的。

这是否说明：

- GEARS 原生 uncertainty 不适合作 SafeConf 主结果；
- 还是因为 n=62 太小；
- 或者 Frangieh split 不适合判断 uncertainty？

### Q3. prediction magnitude risk 的 0.894 能不能用？

`gears_prediction_magnitude_risk` 很强，但它可能只是 magnitude confounding（效应大小混杂）。

请判断：

- 能不能作为主文结果；
- 是否只能放成 diagnostic（诊断）；
- 是否必须补 partial rho（控制效应大小后的相关）。

### Q4. 下一步 GEARS 应该怎么做？

候选：

1. 不再扩 GEARS，把它写成“原生 uncertainty 弱，SafeConf 需要外部特征”的动机；
2. 增加 Frangieh GEARS records 数量；
3. 接 GEARS predicted_effect 与 V0/ContextSim 的 model_disagreement（模型分歧）；
4. 换 Norman/Adamson/Dixit 这类更 GEARS 原生的数据；
5. 暂停 GEARS，先补论文主表和特征消融。

请按优先级排序。

## Codex 当前初步判断

Codex 的初步判断如下，可以反驳：

1. GEARS 接入成功，但还不能说 GEARS uncertainty 成功。
2. GEARS 原生 uncertainty 很弱，应作为负结果/动机，不适合作主文强证据。
3. prediction magnitude risk 很强，但可能被 effect magnitude confounding 攻击，需要谨慎。
4. 真正更有价值的下一步，是把 GEARS predicted_effect 接入 SafeConf 的 model_disagreement，而不是继续只看 GEARS 自带 uncertainty。

