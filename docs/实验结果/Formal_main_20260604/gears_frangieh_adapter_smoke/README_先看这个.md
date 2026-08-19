# GEARS Frangieh adapter smoke 结果说明

这一步做的是 GEARS（一个常用图神经网络扰动预测模型）接入测试。

一句话：

> GEARS 已经能在 Frangieh（黑色素瘤扰动数据集）上训练、预测，并导出 PredictionRecord（预测记录）。但这只是 smoke（小规模连通测试），不是正式 SafeConf 结论。

## 这次跑了什么

- 数据：Frangieh（黑色素瘤相关单细胞扰动数据）。
- 模型：GEARS。
- 设置：3 个 seed（随机种子），每个 seed 只跑 1 epoch（训练 1 轮）。
- 目标：确认 GEARS 预测结果能不能变成 SafeConf 需要的 PredictionRecord。

## 关键结果

| 项目 | 结果 |
|---|---:|
| seed 数 | 3 |
| 是否都成功 | 是 |
| 合计 PredictionRecord | 62 |
| unique perturbation（不同扰动） | 58 |
| 平均 test MSE | 0.00144 |
| 平均 test Pearson | 0.9958 |
| 平均 top20 DE MSE | 0.00542 |
| 平均 top20 DE Pearson | 0.9382 |
| GEARS 原生 uncertainty（不确定性） | 未导出，当前为空 |

## 这说明什么

能说：

- GEARS 依赖和环境能跑通。
- Frangieh 数据可以被 GEARS adapter（适配器）读入。
- GEARS 的 predicted effect（预测效应）和 true effect（真实效应）已经能导出为 SafeConf 的 PredictionRecord。

不能说：

- 不能说 GEARS uncertainty 已经完成，因为 uncertainty 两列还是空。
- 不能说 SafeConf 已经在 GEARS 上正式验证，因为当前只有 62 条 test records。
- 不能把这个和 7 主表 formal audit（正式审计）混为一谈。

## 当前定位

这是 GEARS adapter smoke：

> 证明“能接上”，不是证明“已经发表级稳定”。

下一步如果要冲更高目标，应把 GEARS 从 smoke 推到 formal：

1. 固定 SafeConf split（切分）或明确 GEARS split 与 SafeConf split 的差异。
2. 增加 GEARS test records 数量。
3. 如果能导出 GEARS 原生 uncertainty，就和 SafeConf score 做公平对比。
4. 如果导不出原生 uncertainty，就只能写成 seed ensemble / prediction-record adapter，而不是 GEARS native uncertainty。

