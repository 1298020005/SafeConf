# E115 预设分析：三数据集实际分诊效用

## 问题

E113 已回答风险分数能否排序误差。E115 回答排序能否转化为有限复核预算下的实际收益。

## 冻结输入

- E108 Frangieh 正式测试任务；
- E112 Lara ex vivo、Santinha 正式测试任务；
- 风险分数不重拟合；测试真值不改变排序。

## 冻结指标

每个数据集、每个外层 fold、每个分数分别计算：

1. coverage 50%–100% 的 normalized AURC，越低越好；
2. top-20% 高风险任务的平均误差富集，越高越好；
3. 拒绝最高风险 20% 后的剩余误差下降，越高越好；
4. top-20% 风险任务捕获的总错误比例，越高越好。

SafeConf 与 predicted magnitude、model disagreement 成对比较。三数据集等权宏平均为主结果，数据集和 fold 分层 bootstrap 95% CI 为不确定性结果。

## 通过标准

SafeConf 必须在 normalized AURC 与 top-20% error capture 上同时优于两个基线，且相对至少一个强基线的 bootstrap 95% CI 不跨 0，才记为“实际分诊效用得到增量证据”。否则只记描述性趋势或负结果。

