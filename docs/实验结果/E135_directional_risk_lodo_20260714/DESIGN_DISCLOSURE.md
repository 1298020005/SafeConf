# E135 设计披露

E133/E134 解封后发现原 SafeConf 只对 absolute RMSE 稳定，对方向误差不稳定。随后使用四个既有部署特征探索了五个低复杂度候选：Ridge、正系数 Ridge、HistGradientBoosting、RandomForest、ExtraTrees。候选结构与首轮诊断结果均已暴露，因此 E135 定位为探索性模型选择。

最终选择 Ridge(alpha=10) 的理由是结构最简单、六个 LODO 数据集均为正，并非把其 LODO 结果当成独立确认。模型、特征顺序、系数和 target rank 定义在任何第七数据集表达值、预测或误差被读取前冻结。第七数据集若失败，不再调系数后重复宣称确认。
