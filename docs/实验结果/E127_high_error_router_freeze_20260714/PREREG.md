# E127 设计冻结｜高错误任务路由器与第六数据验证

冻结日期：2026-07-14；冻结基线提交：`f40d099ca53da937603e3918436523d1a17c1f6f`。

## 背景

E126 的正系数 Ridge 能迁移，但没有超过原 SafeConf。它拟合连续误差秩，与实际“把有限复核预算投给高错误任务”的目标并不完全一致。E127 改为直接估计任务进入本 fold 误差最高 20% 的概率，并在尚未运行的 Tian–Kampmann 2021 CRISPRi 数据上验证。

## 冻结方法

- 历史训练库：Frangieh、Lara ex vivo、Santinha、Shifrut、Liang 的 2,221 个正式测试任务。
- 输入：SafeConf、模型分歧、预测幅度、背景新颖度、扰动新颖度、低支持度；每个特征在 dataset × outer-fold 内变为百分位秩，不读取目标误差。
- 标签：历史来源数据各 fold 内误差最高 20%。
- 唯一模型：`HistGradientBoostingClassifier`；100 棵迭代树、学习率 0.05、最多 7 个叶结点、最大深度 3、最小叶样本 40、L2=10，六个特征均施加单调递增约束。不搜索超参数。
- 样本权重：五个来源数据集等权，各数据集内 folds 等权。
- 确认数据：Tian–Kampmann 2021 CRISPRi。其任务真值不得参与拟合、特征变换、阈值或超参数选择。

## 主判定

1. Tian 四 folds 等权的 top-20% total error capture 高于原 SafeConf 和 predicted magnitude；
2. normalized AURC 低于原 SafeConf 和 predicted magnitude；
3. 对至少一个强基线，上述两个指标的成对 fold × perturbation-cluster bootstrap 95% CI 均在有利方向。

未通过时，E127 作为负结果保留，不能继续在 Tian 真值上调参后冒充确认性结果。
