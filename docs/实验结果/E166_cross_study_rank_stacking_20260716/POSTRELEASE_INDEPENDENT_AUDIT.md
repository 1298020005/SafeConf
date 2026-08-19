# E166 发布后独立复核

复核日期：2026-07-16。复核只读取已经完成的 `release/` 表格，没有重拟合权重、修改任务或覆盖正式结果。

## 文件完整性

- `RESULTS_SHA256.csv` 记录 14 个发布工件；逐文件重算字节数与 SHA-256，错误数为 0。
- 发布目录共有 16 个文件，另外两个是 `RUN_STATUS.json` 与 manifest 本身，符合生成合同。
- manifest SHA-256 为 `945f28dc75ea92fe84b10d3f2d99c699a624763ed3b3046b0881796d2a4d5a27`，与 `RUN_STATUS.json` 完全一致。
- 正式运行对应冻结提交 `7de0b5c74832d1f611fac2a999bc9bd2566a0b3c`。

## 数值复核

从 `E166_TASK_SCORES.csv` 独立按 `(dataset, fold_id)` 重算四种分数与 RMSE 的 Spearman，再对每个研究的 folds 等权平均：与 `E166_STUDY_RESULTS.csv` 的最大绝对差为 `1.11e-16`。

- 八研究等权 `Delta rho(stack - magnitude) = 0.03237903478474495`。
- 点估计为正的研究为 5/8。
- 正式两层 bootstrap 95% CI 为 `[-0.028279988637739263, 0.087714852012629282]`。
- 八组 LODO 权重均非负，逐行权重和与 1 的最大偏差小于 `2.22e-16`。
- 8 个 leakage-audit 记录均确认留出研究不在训练集合，测试真值用于权重拟合的行数为 0。

因此严格 gate 的 `FAIL` 判定正确，不能写成稳定超过 predicted magnitude。组合分数相对 disagreement 的等研究平均增量为 `0.1179`，两层区间 `[0.0727, 0.1665]`；相对原 SafeConf 的平均差为 `-0.0245`，区间跨 0。

## 冻结哈希

- runner：`3effbd6734f99fb0e95a795674afffde6859e19e46629f5dc80ad3b910a5f8d0`
- contract：`945aab2393da3caf6e5e07d24f772deb0369380fd3f69488bb3fbd933f2fb4fc`
- task scores：`a352b79909289839768275ad434b1a2194559f421a025d4be079b060a7ffba7b`
- run status：`7361ca4431dbd882a795c34c0bebe60bfdede5ef9c748dcb9ab82e492860de13`
