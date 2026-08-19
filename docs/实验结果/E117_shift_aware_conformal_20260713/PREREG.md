# E117 预设分析：困难设置匹配的 conformal 误差上界

## 问题

E114 的 90% 上界经验覆盖充分但偏宽。E117 检验 E109 的内层 row/column/double 任务能否在不读取外层测试真值的情况下，构造更贴近分布偏移的上界。

## 冻结方法

对每个 E108 外层 fold：

1. 只使用对应 E109 内层任务；
2. 每个 setting 按任务哈希拆成 risk-fit 与 conformal-calibration 两半；
3. 用 disagreement、predicted magnitude、context novelty、perturbation novelty、log-support 拟合正系数 Ridge 基础误差；
4. 按 setting 计算 90% one-sided conformal residual quantile；没有匹配内层 setting 的 random-pair 使用全部内层 calibration residual；
5. 外层测试真值只用于最后覆盖率与宽度评价。

## 通过标准

- pooled empirical coverage 不低于 0.90；
- 四种 setting 的经验覆盖均不低于 0.85；
- pooled mean upper bound 低于 E114；
- 测试真值没有进入拟合、候选选择或分位数。

若任一条件失败，E117 作为负结果保留，不替换 E114。

