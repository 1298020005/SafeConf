# Sprint 1 LODO 结果速读

## 一句话结论

这轮把已有 ErrorRanker（误差排序器，小模型）和 ElasticNet（可解释线性模型）真正接到了 7 个正式数据集上。结果说明：**可训练小模型目前不能替代 frozen protocol v0.2（冻结打分协议）当主方法**。

## 现在完成了什么

- 跑完 group-LODO（leave-one-dataset-group-out，留一数据集组外部验证）。
- 数据集分组已显式写入，避免同源数据泄漏。
- 对比了 4 类 score（分数）：
  - `protocol_v0_2_family_confidence`：当前主协议。
  - `model_disagreement_risk`：两个预测器意见不一致时风险更高。
  - `lodo_error_ranker_risk`：HistGBT（梯度提升树）训练出来的风险排序器。
  - `safeconf_linear_ranker_risk`：ElasticNet（可解释线性头）。
- 输出了 bootstrap CI（自助法置信区间）和 risk-coverage（风险-覆盖曲线）。

## 最重要的结果

按 7 个数据集、两个 predictor（预测器）平均：

| score | partial rho（控制效应大小后的相关） | 判断 |
|---|---:|---|
| `model_disagreement_risk` | 0.423 | 很强，但更像重要 baseline（基线） |
| `protocol_v0_2_family_confidence` | 0.369 | 仍然最适合做主方法 |
| `safeconf_linear_ranker_risk` | 0.093 | 可解释，但不够稳 |
| `lodo_error_ranker_risk` | 0.045 | 当前不适合做主方法 |

风险覆盖方面，80% coverage（保留 80% 高可信预测）平均改善：

| score | 80% coverage 平均 RMSE 改善 |
|---|---:|
| `protocol_v0_2_family_confidence` | 16.81% |
| `model_disagreement_risk` | 9.94% |
| `safeconf_linear_ranker_risk` | 6.91% |
| `lodo_error_ranker_risk` | 4.63% |

## 怎么理解

这不是坏消息，反而是一个很重要的方向判断：

1. 说明“加一个可训练小模型”并不会自动更好。
2. 说明 frozen protocol v0.2（冻结协议）作为主方法更稳，也更容易防止审稿人质疑“你是不是调参刷结果”。
3. ErrorRanker 和 ElasticNet 适合放在 ablation（消融对照）里，不适合现在抢主角。
4. `model_disagreement_risk` 很强，后面要认真研究它为什么强，但不能只拿它包装成新方法。

## 文件说明

- `tables/LODO_MAIN_TABLE.csv`：主结果表。
- `tables/LODO_RISK_COVERAGE.csv`：risk-coverage 曲线数据。
- `tables/LODO_BOOTSTRAP_CI.csv`：bootstrap 95% CI。
- `tables/SAFECONF_LINEAR_RANKER_COEFFICIENTS.csv`：ElasticNet 系数表。
- `tables/LODO_ERROR_RANKER_STATUS.csv`：训练状态和使用特征。
- `gears_feasibility/`：Frangieh GEARS 可行性审计。

## 给 Claude / Qwen 的复核重点

请重点判断：

1. 是否同意 `protocol_v0_2_family_confidence` 继续做主方法？
2. `model_disagreement_risk` 很强，应该作为强 baseline，还是可以发展成主方法的一部分？
3. ErrorRanker / ElasticNet 是否只放 ablation？
4. 下一步是优先做 Tahoe adapter（超大外部验证），还是优先做 GEARS adapter smoke run？
