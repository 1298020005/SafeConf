# E201 TxPert 四目标多种子正式训练冻结

冻结日期：2026-08-02

## 要回答的问题

周老师问的是：留出一整个细胞背景后，没有见过目标 `(context, perturbation)`
pair 的模型能否预测，SafeConf 在不读取目标扰动表达时能否给这些任务
分层。E200 只有官方 K562 单 checkpoint。E201 用公开代码把 K562、RPE1、
HepG2 和 Jurkat 分别作为目标背景，建立同一架构的多种子结果。

TxPert 论文明确报告四个 leave-one-cell-line-out 实验，目标背景的扰动细胞
全部留出，四个细胞系对照可用，结果为 4 个训练种子的均值和标准差。
依据：[TxPert 论文](https://www.nature.com/articles/s41587-026-03113-4)、
[官方代码](https://github.com/valence-labs/TxPert)。

## 实验定位

准确名称是：**固定 TxPert 公开代码、公开数据和可恢复训练设置的
STRING-GAT 重训练审计**。

不写成作者内部流水线的完全复现，原因是：

- 公开仓库没有 `Trainer.fit()` 入口；
- 论文没有公开 4 个随机种子的具体数值；
- 论文的最强模型使用未公开 PxMap/TxMap；
- 公开 checkpoint 的内部训练批次数无法用当前 Zenodo 数据唯一还原。

## 目标、数据和种子

| 目标 | source 背景 | train dataset 行 | train batches | validation 行 | validation batches |
|---|---|---:|---:|---:|---:|
| K562 | RPE1, HepG2, Jurkat | 294,951 | 4,608 | 80,340 | 1,256 |
| RPE1 | K562, HepG2, Jurkat | 273,003 | 4,265 | 76,950 | 1,203 |
| HepG2 | K562, RPE1, Jurkat | 314,391 | 4,912 | 89,117 | 1,393 |
| Jurkat | K562, RPE1, HepG2 | 282,132 | 4,408 | 78,682 | 1,230 |

每个 target 固定种子 `{1, 2, 3, 4}`。种子 1 来自公开配置，2–4 是在任何
目标扰动真值打开前事先登记的连续整数。不会根据某个 target 的真值结果
删除或补选种子。

## 模型与优化

- TxPert commit：`08d82eea86746b044cf7531f4ec8c5f60e1cb73f`；
- 图：STRING top-20，`reduce2perts=true`；
- 扰动编码器：4 层 GATv2，hidden 128，2 heads，`skip_cat`；
- 预测器：TxPert 公开 cross-cell GAT，无 basal encoder；
- batch-matched control：启用，并在可用对照内取均值；
- batch size：64；
- optimizer：AdamW，初始学习率 `3e-4`，weight decay 0；
- 训练：80 epochs；前 5 epochs 线性 warmup，随后 75 epochs cosine decay 到 0；
- validation：只用 source-context validation，监控 `val_pearson_delta`；
- checkpoint：每 5 epochs 评估一次 best，同时保存 last；
- EarlyStopping：patience 100，大于总训练轮数，因此不会因一个 target 早停；
- 主 checkpoint：epoch 80 的 `last.ckpt`，与官方释放的 epoch-79 Lightning
  checkpoint 口径一致；source-validation best 只作敏感性分析。

## 真值隔离

1. 训练进程只接收对应 target 的物理盲 H5AD；
2. 目标扰动细胞为 0，目标 control 允许进入训练；
3. H5AD `uns` 在训练前为空，datamodule 只加载公开 Pharos 等级；
4. datamodule 只构造 train/validation dataset，不构造 target test dataset；
5. 每次正式运行必须记录 blind manifest、Git HEAD、TxPert HEAD、命令、
   数据行数、实际 step、最高显存和 checkpoint 哈希；
6. 16 个正式 checkpoint 全部完成和封存前，不打开任何新的目标扰动结果。

## 排队和停止规则

按实测 RPE1 资源门，单模型约 11.9 GPU 小时，16 个模型约 190 GPU 小时。

- 第一阶段：四个 target 各训练 seed 1，只看 source validation 和工程状态；
- 第二阶段：四个 target 固定补齐 seeds 2–4；
- 不因 source validation 分数不漂亮、不同 target 之间有差异而丢弃已登记作业；
- 只有非有限损失、显存溢出、文件哈希改变、目标真值访问或 checkpoint 无法读取
  等工程失败才停止对应作业，原始失败目录保留。

## 后续评价的事前边界

正式评价另行冻结代码，但以下边界现在固定：

- 主层是 strict context-only：扰动在 source train 出现，整个 target context 扰动留出；
- source-validation-only、source-unsupported 和 `unseen_cell_pert` 单独作压力层，
  不与整行主层混合；
- 主误差为每任务 batch-matched centroid RMSE；
- 必须同时报告每个 seed、四种子 centroid、family RMS 和 worst-seed error；
- 强基线至少包含 official general baseline、batch-matched control 和 predicted magnitude；
- 补充评价按 E198 事前选中的 `mse`、`pearson_pert`、`rank`、
  `energy_distance_pca_k=50` 和 `de_auprc`；
- 分歧先解释为四种子 family-error 的确定性 lower certificate；
  它与单个 seed 误差的经验关系、相对 magnitude 的增量和 20% 复核效用分开裁决。

PerturBench 显示简单模型常有竞争力，rank 能暴露仅看 RMSE 时的模式塌缩；
scPertEval 则要求先用技术重复和负参考检查协议动态范围。
依据：[PerturBench](https://papers.nips.cc/paper_files/paper/2025/file/8aee537279a66ced96319dfca3c00002-Paper-Datasets_and_Benchmarks_Track.pdf)、
[scPertEval](https://github.com/Virtual-Cell-Research-Community/scPertEval)。
