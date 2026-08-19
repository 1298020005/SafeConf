# GEARS Frangieh 正式探测结果

更新时间：2026-06-06

## 一句话结论

GEARS（图神经网络扰动预测模型）已经在 Frangieh（黑色素瘤 CRISPR 扰动数据集）上完成 3 个 seed、每个 20 epoch 的正式探测。

结论不是“GEARS uncertainty 很强”。  
更准确地说：

> GEARS 能导出逐条 prediction（预测记录）和原生 uncertainty（不确定性），但这个 uncertainty 对真实误差的区分能力很弱；prediction magnitude（预测幅度）反而和误差高度相关。

## 跑了什么

| 项目 | 结果 |
|---|---:|
| dataset（数据集） | Frangieh |
| predictor（预测器） | GEARS |
| seeds（随机种子） | 1, 2, 3 |
| epochs（训练轮数） | 20 / seed |
| PredictionRecords（逐条预测记录） | 62 |
| native uncertainty（原生不确定性） | 已导出 |

## GEARS 预测本身怎么样

三个 seed 都正常完成：

| seed | test Pearson | DE Pearson | records |
|---:|---:|---:|---:|
| 1 | 0.9963 | 0.9428 | 21 |
| 2 | 0.9957 | 0.9451 | 20 |
| 3 | 0.9959 | 0.9273 | 21 |

这里的 Pearson（相关系数）说明 GEARS 预测本身是能跑通的。

## confidence scoring（可信度打分）结果

| score（分数） | aligned rho | RC@80% | 解释 |
|---|---:|---:|---|
| GEARS prediction magnitude risk | 0.894 | 14.81% | 预测幅度越大，误差越大；信号很强，但可能是 magnitude confounding（效应大小混杂） |
| GEARS native uncertainty confidence | 0.096 | -0.82% | 原生 uncertainty 区分误差能力很弱 |

`RC@80%` 是 risk-coverage@80%（只保留低风险/高可信 80% 后，平均误差下降多少）。

## 现在能说什么

- 可以说：GEARS 已经接入 SafeConf 的 PredictionRecord（预测记录）格式。
- 可以说：GEARS 原生 uncertainty 能导出，但在这个 Frangieh 小探测里不强。
- 可以说：GEARS prediction magnitude 是强信号，但它更像 effect magnitude（效应大小）相关信号，不能直接吹成“可靠 uncertainty”。

## 现在不能说什么

- 不能说：GEARS uncertainty 已经验证成功。
- 不能说：SafeConf 已经证明对 GEARS 也稳定有效。
- 不能只报 prediction magnitude risk 的 0.894，因为这可能被审稿人攻击为“只是效应大小”。

## 下一步建议

1. 把 GEARS 结果给 Claude/Qoder 复核，重点问：这个 GEARS 证据应该放主文、补充，还是只放方法可行性。
2. 如果要增强 predictor-agnostic（不绑定预测器）说服力，下一步不应只看 GEARS native uncertainty，而要把 GEARS predicted_effect 和 V0/ContextSim 的 disagreement（模型分歧）接起来。
3. 如果继续 GEARS，优先增加 records 数量或换更适合 GEARS 的 split，不要只重复 20 epoch。

## 关键文件

- 评估表：`tables/GEARS_CONFIDENCE_EVAL_SUMMARY.csv`
- risk-coverage：`tables/GEARS_RISK_COVERAGE.csv`
- prediction records：`tables/GEARS_PREDICTION_RECORDS_COMBINED.csv`
- 训练状态：`tables/GEARS_PREDICTION_RECORD_STATUS.csv`
- 报告：`reports/GEARS_CONFIDENCE_EVAL_REPORT.md`

