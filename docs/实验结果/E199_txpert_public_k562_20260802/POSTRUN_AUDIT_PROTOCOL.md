# E199 运行后稳健性审计口径

记录时间：2026-08-02

## 性质

这不是预注册验证，也不改变 `FORMAL_EVALUATION_FREEZE.md` 中的 gate。正式结果生成后，
复核发现两个需要单独拆开的解释问题，因此追加探索性审计，并在所有输出中标记
`POST HOC / EXPLORATORY`。

1. `family_rms_error² = centroid_error² + diversity_lower_bound²`，所以用 diversity
   解释 family RMS error 同时含有数学结构和经验排序两部分，不能只凭该相关声称
   对真实误差有独立预测作用；
2. 目标基因方向只在扰动基因位于 5,000 基因评价面板时可计算，报告命中率时必须
   同时报分子和有效分母。

## 固定输入

- `formal_evaluation/E199_OUTPUT_HASHES.csv` 中封存的正式输出；
- 主分析仍只使用不少于 30 个真实细胞的 263 个任务；
- 风险量使用结果打开前封存的 `diversity_lower_bound_pretruth`；
- 简单对照仍为 `predicted_magnitude`。

## 追加分析

对以下不由 diversity 代数定义的误差分别报告 Spearman、20% 固定复核预算效用，
以及 diversity 相对 predicted magnitude 的配对差值：

- 三模型等权均值的 `centroid_error`；
- 三个成员中最差的 `family_worst_error`；
- GAT、Exphormer、Exphormer-MG 各自的 centroid RMSE。

沿用 5,000 次任务 bootstrap 和正式脚本中的确定性种子函数。五个 scPertEval 端点的
方向一致性只作描述，不追加成功门。目标基因方向单列“可计算数、方向命中数、命中率”。

## 解释规则

- 正式 gate 仍按原冻结结果报告；
- 新表只能支持稳健性或发现性陈述，不能改写成事前验证；
- 若不同端点方向不一致，正文必须保留这种异质性；
- 整个 context 留出和跨数据集迁移仍为未回答问题。
