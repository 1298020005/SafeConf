# E200 TxPert K562 整体背景留出冻结

冻结日期：2026-08-02

## 周老师问题对应关系

E200 专门回答“整行留出”：训练扰动来自 RPE1、HepG2 和 Jurkat，目标 K562 的
扰动细胞全部不进入训练。与 TxPert 论文一致，K562 对照细胞可以用于构造目标背景
基态和 batch-matched control。因此本实验是 **zero-shot perturbation transfer to a new
cell line**，不是目标背景完全没有任何观测。

## 拆分口径

官方 `unseen_cell` 子组有 815 个 K562 扰动，但其中：

- 580 个扰动在实际梯度训练条件表中出现：严格 `context-only`；
- 202 个只在 validation 条件表出现；
- 33 个在 train/validation 都没有；
- 另有 272 个官方 `unseen_cell_pert`，同时是新背景和新扰动。

主分析只使用第一组，避免把“整行”和“整列”混在一起。580 个严格任务中，566 个
不少于 30 个 K562 真实细胞，作为主分析；14 个 10–29 细胞任务单列敏感性。其余
202、33 和 272 个任务只作分层外部压力测试，不回填主分析。

## 公开预测器

- TxPert 官方 `K562_unseen_cell_gat.ckpt`；
- 官方 cross-cell general baseline；
- batch-matched K562 control 作为负对照。

公开记录只提供一个跨细胞 GAT checkpoint。E200 不把 GAT 与 general baseline 的
差异包装成“多模型家族不确定性”，也不声称完成多架构验证。若单模型审计通过，
下一阶段再训练多种子/多架构跨背景模型。

## 结果打开前的风险量

目标 K562 扰动表达不得用于以下特征：

1. `model_baseline_gap`：GAT 与 general baseline 的任务 centroid RMSE；
2. `training_delta_dispersion`：同一扰动在训练细胞系中的 batch-matched delta
   centroid 跨背景离散度；仅一个训练背景时用全体中位数填补，并保留背景缺失项；
3. `negative_log_train_cells = -log(1 + n_train_cells)`；
4. `support_context_deficit = 3 - n_train_contexts`；
5. `predicted_magnitude`：GAT 预测与 K562 batch-matched control 的 centroid RMSE，
   作为简单对照，不纳入主组合。

四个主分量在 580 个严格任务的封存特征上分别作 Z-score，等权平均得到
`transfer_risk`：

`mean[z(gap), z(dispersion), z(-log(1+n_train)), z(3-n_contexts)]`。

标准化只使用预测、训练背景数据和目标对照，不读取 K562 扰动真值。

## 评价与判据

- 主误差：GAT task centroid RMSE；
- 补充端点：scPertEval 的 `mse`、`pearson_pert`、`rank`、
  `energy_distance_pca_k=50`、`de_auprc`；
- 基线：general baseline 与 batch-matched control；
- 置信区间：5,000 次任务 bootstrap；固定复核预算 20%。

三项分开裁决：

1. 经验路由：`transfer_risk` 对 GAT RMSE 的 Spearman 和 20% review utility 的
   95% CI 下限均大于 0；
2. 新增价值：`transfer_risk - predicted_magnitude` 的配对 ΔSpearman 或 Δutility
   至少一个 95% CI 下限大于 0；
3. 模型归属：分别报告 GAT 与 general baseline 的任务误差排名一致性，并分别对
   两种误差做风险关联；不因相关就直接改称模型无关。

## 不能外推的结论

E200 即使通过，也只覆盖 K562 作为目标背景、CRISPRi、公开 GAT 和 3,352 基因面板。
它不能替代其他目标细胞系、其他模型家族、跨独立数据集迁移或湿实验验证。
