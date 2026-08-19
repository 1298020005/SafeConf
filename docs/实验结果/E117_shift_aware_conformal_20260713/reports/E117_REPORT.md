# E117｜困难设置匹配的 conformal 误差上界

E117 只使用 E109 的内层 row/column/double 任务拟合基础误差并校准残差。E108 外层测试真值没有进入模型、候选选择或分位数。random-pair 没有匹配的内层 setting，按预设使用全部内层 calibration residual。

| setting | n | E117 coverage | E114 coverage | E117 mean upper | E114 mean upper | upper reduction |
|---|---:|---:|---:|---:|---:|---:|
| context_and_perturbation_unseen | 90 | 0.800 | 0.978 | 0.0596 | 0.1073 | 0.0477 |
| context_unseen_row | 477 | 0.704 | 0.985 | 0.0605 | 0.1031 | 0.0425 |
| perturbation_unseen_column | 180 | 0.778 | 0.961 | 0.0757 | 0.0978 | 0.0221 |
| random_missing_pair | 90 | 0.667 | 0.989 | 0.0548 | 0.0943 | 0.0395 |
| all_test_settings_pooled | 837 | 0.726 | 0.980 | 0.0631 | 0.1014 | 0.0384 |

## 预设判定

- 通过：**否**。
- pooled coverage ≥0.90：否。
- 每个 setting coverage ≥0.85：否。
- mean upper 低于 E114：是。

只有全部条件通过，E117 才能替换 E114。即使通过，理论保证仍依赖内层困难任务与未来任务的条件可交换性；这里不能写成任意分布偏移下的无条件覆盖。
