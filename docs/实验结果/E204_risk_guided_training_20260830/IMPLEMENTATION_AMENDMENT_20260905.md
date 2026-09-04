# E204 权重覆盖修正记录

日期：2026-09-05

## 发现的问题

第一版 `build_e204_source_weight_manifest.py` 从 E201 的 target 任务表生成权重。该表只包含目标细胞系中可评价的 467–580 个条件，不是 TxPert 实际用于 source 训练的 1,366 个条件。K562 单 epoch profile 虽然完成，但 304,158 个训练样本因条件不在清单中退回单位权重，因此该作业只证明加权代码能运行，不能作为 E204 科学结果。

## 修正

新增 `build_e204_training_weight_manifest.py`，直接读取四个物理盲训练视图和冻结的 `train_test_split.pkl`。split 每个 target 有 1,366 个标签，其中 `ctrl` 是未扰动对照，固定使用单位权重；其余 1,365 个 source 扰动条件计算支持数、背景覆盖和 source effect dispersion。四个 target 合计应有 5,460 行扰动权重。目标扰动表达在这些物理视图中不存在；程序继续审计访问行为。

`run_e204_weighted_training.py` 现在拒绝每个 target 少于或多于 1,365 条扰动权重的清单，防止旧 preview 被误用于正式训练。

## 时间边界

E204 的公式、λ=0.5、权重截断和 `dispersion_only` 对照在 E201 解盲前已于提交 `5bb3550` 登记。本次是权重覆盖的实现修正，不改公式。由于修正发生在 E201 解盲后，E204 在 E201 同一数据上的结果按预登记二级实验解释；最终确认需要再使用一个没有参与本次排错的新数据集或新 target。
