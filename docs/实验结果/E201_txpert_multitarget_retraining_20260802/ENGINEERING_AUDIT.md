# E201 工程审计：TxPert 多目标公开代码重训练

日期：2026-08-02

## 公开材料能确定什么

- TxPert 论文的跨细胞系实验依次留出 K562、RPE1、HepG2 和 Jurkat，目标背景的
  扰动细胞全部留出，目标背景对照允许参与训练；每个模型报告 4 次训练的均值和
  标准差。
- 公开 `config-x-cell-gat.yaml` 固定为 STRING top-20 图、4 层 GATv2、128 维
  隐层、2 个注意力头、batch-matched 且取均值的对照、64 推理批量和无 basal
  encoder。
- 上游 `PertPredictor` 固定使用 AdamW，默认权重衰减为 0。
- 公开 K562 checkpoint 的 Lightning 状态显示：第 80 个 epoch 保存、初始学习率
  `3e-4`、5 epoch 线性预热、随后 75 epoch 余弦衰减；EarlyStopping 监控
  `val_pearson_delta`，patience 为 100；ModelCheckpoint 每 5 epoch 检查一次，
  元数据中的最佳验证点为 epoch 9。

依据：[TxPert 论文](https://www.nature.com/articles/s41587-026-03113-4)、
[补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41587-026-03113-4/MediaObjects/41587_2026_3113_MOESM1_ESM.pdf)、
[官方代码](https://github.com/valence-labs/TxPert)、
[官方 Zenodo 记录](https://zenodo.org/records/15420279)。

## 公开材料不能确定什么

- 官方仓库没有 `Trainer.fit()` 入口，只公开 baseline、checkpoint inference 和
  prediction；Git 历史中也没有训练入口。
- 论文和补充材料没有列出 4 个随机种子的具体数值。
- checkpoint 不含 `hyper_parameters`。它记录的每 epoch 5,976 个训练 batch 与
  当前 Zenodo 数据和公开 batch size 无法对应，因此作者内部训练数据装配不能从
  公开产物唯一还原。
- checkpoint 原始路径含 `K562_adamson_train_together`，但当前公开 datamodule
  明确从 `all` 中排除 `K562_adamson`。E201 不把路径字符串猜测成事实，也不把
  K562_adamson 扰动加入主训练。
- 论文最强配置使用未公开的 PxMap 和 TxMap；E201 只能检验公开 STRING-GAT。

所以 E201 的准确名称是“固定公开代码、公开数据和可恢复训练设置的重训练审计”，
不是作者内部流水线的逐字节复现。

## 盲训练隔离

每个目标细胞系建立独立 H5AD 训练视图：

1. 保留四个公开细胞系的全部对照；
2. 只保留另外三个细胞系中属于官方 train 或 validation 条件的扰动细胞；
3. 删除目标细胞系的全部扰动细胞；
4. 删除 K562_adamson；
5. 清空原始 `uns`，防止目标差异表达排名等结果元数据进入训练文件；
6. 复制官方 condition split 和 subgroup 标签，它们只含任务名称，不含表达真值；
7. 对完整输入、训练视图和拆分文件记录 SHA-256。

训练 adapter 的 datamodule 只建立 train 和 validation dataset，不建立 test dataset；
训练期间只计算 source-context validation Pearson Δ。完整目标 H5AD 只在模型和
checkpoint 封存后的独立预测阶段使用。

## 工程阶段与正式阶段

先对 RPE1、seed 1 运行 20 个训练 batch 的工程 smoke，验证显存、单步时间、
梯度、checkpoint 和目标隔离。smoke 不计算目标预测或目标误差，也不参与任何科学
结论。根据实测资源冻结正式运行数量；正式协议冻结并双远程推送前，不打开其他目标
结果。

20-batch smoke 通过后，再运行一个 RPE1、seed 1 的完整 1-epoch 资源门：

- 使用全部 4,265 个训练 batch 和 1,203 个 source validation batch；
- 模型、数据、batch size 64、AdamW 和学习率 `3e-4` 与正式方案相同；
- 因为只运行 1 epoch，使用恒定学习率，不冒充 80-epoch 结果；
- 只记录总时间、峰值显存、训练吞吐、验证吞吐和 source validation 指标；
- 不加载目标扰动真值，不作为科学结果。

这一资源门用于决定正式作业的排队和并行数，不会根据其验证数值修改
风险特征、误差定义或后续科学假设。
