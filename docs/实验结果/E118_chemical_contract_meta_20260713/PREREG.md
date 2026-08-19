# E118 预设分析：化学扰动统一合同元审计

## 问题

把 E84、E87、E89 的 CPA 双预测器化学扰动结果放进同一风险评价合同，判断 gene 主线以外是否存在独立的跨模态增量。

## 冻结输入与分数

- E84：CPA–ridge，sciPlex3 四象限、8 个正式 manifest；
- E87：CPA–ridge，sciPlex3 → OpenProblems；
- E89：CPA–dose interpolation，sciPlex3 → sciPlex4；
- 所有输入必须 strict issue count=0，且 target truth 只用于评价；
- 主比较为 model disagreement 与 predicted magnitude；不重新使用测试真值学习权重。

## 通过标准

只有当跨来源等权宏平均中 disagreement 相对 magnitude 的 Spearman 增量和 top-20% error capture 增量的 bootstrap 95% CI 均不跨 0，才写“chemical 模态中存在独立增量”。否则写为统一正式负边界，不能与 gene 主结果混合放大。

