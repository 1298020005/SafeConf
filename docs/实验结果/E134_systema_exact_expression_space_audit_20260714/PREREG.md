# E134｜Systema 表达空间精确定义审计（冻结方案）

冻结时间：2026-07-14，E133 结果解封后、E134 计算前。  
触发原因：复核 Systema 方法公式后确认，E133 的 `training mean effect` 是效应空间压力测试；Systema 的 `perturbed centroid` 实际位于受扰动表达空间。跨背景任务中，训练背景 control 均值与测试背景 control 均值不同，两种定义不等价。

## 不变项

- 沿用六数据集、30 个外层 fold、2,953 个测试任务、既有 scGPT/GEARS 预测和 SafeConf 分数。
- 不重训模型，不修改 SafeConf，不筛选任务，不使用验证/测试真值构造参考量。
- 聚类 bootstrap 仍按 perturbation 整簇重采样 3,000 次，数据集等权汇总。

## 精确定义

每个 fold 只用训练任务构造受扰动表达质心：

`O_pert_train = mean(control_context + true_effect)`。

对测试任务：

- `predicted_state = heldout_context_control + predicted_effect`；
- `true_state = heldout_context_control + true_effect`；
- 方向误差在 `predicted_state - O_pert_train` 与 `true_state - O_pert_train` 之间计算；
- 简单预测器恒预测 `O_pert_train`，其 RMSE 与正式模型在同一受扰动表达空间比较。

## 报告规则

- E134 与 E133 并列报告。E134 不能覆盖 E133 的效应空间负结果。
- 若 E134 对 Pearson/cosine 方向误差为正，只能写“在 Systema 表达空间参考下支持”；不能写成跨所有误差定义支持。
- 若 scGPT/GEARS ensemble 对 `O_pert_train` 的 RMSE 优势 95% CI 跨 0，必须说明上游复杂预测器对简单均值的总体优势不稳定。
- 若 E134 仍为负，SafeConf 主张收缩到“control-referenced absolute RMSE risk routing”。
