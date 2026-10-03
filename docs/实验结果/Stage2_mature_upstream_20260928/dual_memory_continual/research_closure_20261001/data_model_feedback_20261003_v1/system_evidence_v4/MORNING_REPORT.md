# SafeConf 当前执行回执

## 已完成

- 实现统一评分入口，但保留规则型公共评分与监督型Ridge候选的不同信息合同。
- 完成E201 Source错误监督的折外Shared分数和Source-supervised Ridge候选。
- 完成McFaline五个反馈预算的Target-supervised Ridge候选。
- 完成5000次按基因簇配对bootstrap。
- 检出旧矩阵/PertEMA结果与当前冻结特征的真实错误合同不一致，已排除出同任务指标。

## 采用决定

在同一当前真值合同的212个任务上，PublicRule U20=0.6732；无目标反馈的SourceRidge U20=0.5099，5000次基因簇bootstrap相对PublicRule的差值区间为[-0.3914,-0.0519]，明确不采用。50%反馈TargetRidge U20=0.6914，点增量为+0.0182，但bootstrap差值区间为[-0.1128,0.0897]，暂不替换PublicRule。当前默认配置因此是原始强公共规则；Ridge保留为条件候选。

## 未完成

McFaline当前212任务的完整统一排序已经闭合。SAMS跨模型后处理仍按原合同继续；论文阶段保持暂停。
