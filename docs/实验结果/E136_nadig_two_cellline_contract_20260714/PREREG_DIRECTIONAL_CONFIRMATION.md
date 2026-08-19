# 第七数据方向风险确认规则

冻结顺序：E135 方向模型系数 → 本文件 → E136 Nadig 标签合同 → 表达预处理与双模型预测 → 风险分数落盘 → 测试真值评价。

## 唯一确认模型

使用 `E135_FROZEN_DIRECTION_MODEL.json` 中 Ridge(alpha=10) 的四个特征、顺序、系数和截距。Nadig 不参与拟合、归一化参数学习或模型选择。若文件哈希与 E136 状态记录不一致，确认无效。

## 主要终点

按 Systema 表达空间训练扰动质心计算：

1. 两预测器平均 centered Pearson error；
2. 两预测器平均 centered cosine error。

每个 cell-line holdout fold 先计算 Spearman，再对两个 fold 等权平均。按 perturbation 整簇重采样 3,000 次。

## 通过标准

- 两个主要终点的 fold 宏平均 Spearman 都必须大于 0；
- Pearson/cosine 的平均 rank 复合终点，perturbation-cluster bootstrap 95% CI 下界必须大于 0；
- 方向风险不得在读取 Nadig 测试真值后修改；
- 同时报告 predicted magnitude、原 SafeConf 和方向风险，不能隐去更强基线。

只有全部满足，才把方向风险头写为“第七数据确认通过”。任一失败均原样保留，并将论文主张限制为 absolute RMSE routing。
