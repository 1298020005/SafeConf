# E141 预注册｜七数据 PROGENy 通路忠实度

冻结顺序：PROGENy 资源快照与部署分数 → 本合同 → 打开预测/真值向量 → 通路误差评价。

- PROGENy 每条通路按资源 p value 固定取前 500 个响应基因；与 512 基因面板重叠不少于 5 个才纳入。
- 权重在每个数据集面板内做 L2 归一化，通路活性为 signed weighted projection。
- absolute 主终点：原 SafeConf 与两预测器平均通路活性 RMSE 的 fold→dataset 等权 Spearman。
- direction 主终点：冻结 Directional-SafeConf 与两预测器平均通路活性 cosine error 的同层级 Spearman。
- 以 perturbation 为整簇、dataset 为总体层做 3,000 次 bootstrap，同时报告 predicted magnitude 与 disagreement。
- 通过标准：两个主相关方向均为正，至少一个 95% CI 下界大于 0；不通过也不得改分数后重报。
