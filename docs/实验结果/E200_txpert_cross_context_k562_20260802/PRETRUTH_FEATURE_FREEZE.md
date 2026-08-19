# E200 跨背景风险特征冻结

冻结日期：2026-08-02

## 输入边界

特征阶段只能读取：

1. TxPert 官方 K562 cross-cell GAT 预测；
2. 同一 GAT 运行保存的 K562 batch-matched control；
3. 官方 cross-cell general baseline 预测；
4. RPE1、HepG2 和 Jurkat 的训练扰动与对照表达；
5. 官方 train/validation/test 拆分和 E200 准备阶段封存的任务支持表。

K562 扰动表达不得进入特征计算。原始 H5AD 同时包含多个背景，程序必须在每次读取 `X` 前验证所有行都来自 RPE1、HepG2 或 Jurkat，并生成读取审计表。

## 任务集

仅使用 580 个严格 context-only 任务：扰动在实际训练条件表中出现，K562 扰动细胞不进入训练。566 个任务含不少于 30 个 K562 细胞，作为主分析；14 个 10–29 细胞任务单列敏感性。

## 原始特征

- `model_baseline_gap`：GAT 任务 centroid 与 general baseline 任务 centroid 的每基因 RMSE；
- `training_delta_dispersion_observed`：对每个扰动，先在每个可用训练细胞系中计算 batch-matched 扰动 delta centroid，再计算这些背景 delta 相对等权 centroid 的 RMS 离散度；
- `negative_log_train_cells = -log(1+n_train_cells)`；
- `support_context_deficit = 3-n_train_contexts`；
- `predicted_magnitude`：GAT 预测 centroid 与对应 K562 batch-matched control centroid 的每基因 RMSE，只作简单对照。

训练 delta 按 TxPert general baseline 的口径计算：每个扰动细胞减去同细胞系、同 batch 的对照均值，然后在该细胞系内取均值。

75 个任务只在一个训练细胞系中有支持，跨背景离散度不可识别。它们的 `training_delta_dispersion` 固定填入其余任务观测离散度的中位数，并保留 `training_delta_dispersion_imputed=true`；不把单背景离散度写为 0。

## 组合风险分数

在 580 个冻结任务上，四个主分量分别用总体标准差（`ddof=0`）做 Z-score：

`transfer_risk = mean[z(gap), z(dispersion), z(-log(1+n_train)), z(3-n_contexts)]`。

不拟合监督权重，不根据 K562 扰动真值选择特征、填补方式或方向。

## 封存门槛

- 580 个任务与 prepare 支持表逐一一致；
- 预测行、基因、任务和 batch 顺序与 prediction seal 一致；
- 训练细胞数和支持背景数与 prepare 表一致；
- K562 扰动表达读取行数必须为 0；
- 四个 Z-score 的均值绝对值小于 `1e-12`，标准差与 1 的差小于 `1e-12`；
- 组合分数代数恒等式的最大残差小于 `1e-12`；
- 特征表、读取审计和输入哈希先提交并双推，然后才允许运行结果评价。
