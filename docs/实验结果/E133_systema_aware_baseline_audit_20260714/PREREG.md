# E133｜Systema-aware 简单基线与方向误差审计（冻结方案）

冻结时间：2026-07-14（运行结果前）  
性质：文献触发的次要稳健性审计，不替换 E131 的六数据集 RMSE 主终点，不重新拟合 SafeConf。

## 问题

1. 在每个外层 fold 中，只用训练任务真实扰动效应构造 `training perturbed-mean` 简单预测器。scGPT、GEARS 与二者均值预测是否优于该简单基线？
2. 将同一个训练扰动均值从预测效应和真实效应中扣除后，计算 Pearson 与 cosine 方向误差。SafeConf 对这两类误差是否仍有正向排序能力？
3. SafeConf 是否能够排序“正式模型相对 training perturbed-mean 的超额 RMSE”？

## 冻结数据与单位

- 数据：Frangieh、Lara ex vivo、Santinha、Shifrut、Liang、Tian CRISPRi。
- 外层划分：沿用 E97/E99/E119/E122/E128 的冻结 fold，不改变训练、验证、测试归属。
- 基本单位：`dataset × fold × context × perturbation` 测试任务。
- 预测器：已落盘的 scGPT 与 GEARS；不重训，不筛任务。

## 严格信息边界

- `training perturbed-mean` 只使用当前 fold 中 `split=train` 且纳入 100% 训练清单的任务真值。
- 不使用验证或测试任务真值构造参考量、分数或阈值。
- SafeConf、disagreement、predicted magnitude 均直接读取既有正式表，不重新拟合。
- 若一个 fold 无法从训练任务构造参考量，则整 fold 明示缺失，不用测试数据补齐。

## 终点

对每个测试任务计算：

- `rmse_scgpt`、`rmse_gears`、`rmse_model_mean`；
- `rmse_training_perturbed_mean`；
- `excess_rmse = rmse_model_mean - rmse_training_perturbed_mean`；
- 训练扰动均值中心化后的两模型平均 `1 - Pearson`；
- 训练扰动均值中心化后的两模型平均 `1 - cosine`。

每个数据集先在 fold 内计算 Spearman，再对 fold 取宏平均。六数据集结论以数据集等权宏平均和“按 perturbation 聚类、在数据集内重采样”的 3,000 次 bootstrap 置信区间报告，避免把大量相关任务当成独立样本。

## 预先规定的解释

- 该审计是压力测试，不设置新的调参门槛，也不以单个 p 值决定论文成败。
- 若 SafeConf 对 Pearson/cosine 方向误差的六数据集宏平均 Spearman 为正，且至少一种方向误差的 dataset-bootstrap 95% CI 下界不低于 0，记为“跨误差定义支持”。
- 若正式模型在多数数据集的平均 RMSE 不优于 training perturbed-mean，必须在论文中把方法定位限定为“现有预测器的风险路由”，不能暗示上游预测器本身具有领先性能。
- 若 SafeConf 对 excess RMSE 无稳定正相关，则不得声称它能识别“模型何时不如简单预测器”；只保留其对绝对错误的排序结论。

## 数学说明

RMSE 对预测和真值同时减去同一参考向量不变，因此不把“中心化 RMSE”伪装成新终点。训练扰动质心只用于参考敏感的 Pearson/cosine 方向指标，以及作为独立的简单预测器基线。
