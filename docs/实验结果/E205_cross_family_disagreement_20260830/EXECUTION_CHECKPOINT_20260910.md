# E205 执行检查点

记录时间：2026-09-10 12:56（Asia/Shanghai）

## 已完成

- 冻结 E205 执行方案：四目标、四种子、整细胞背景留出，只更换 TxPert 的扰动传播结构；
- 冻结 SafeConf-M 为 `0.80 × 幅度百分位 + 0.20 × SafeConf 百分位`，禁止在 E205 结果上重新调权重；
- 新增 Exphormer 训练适配器并通过 Python 语法、命令行参数和 Hydra 配置组合检查；
- 确认合成后的配置保留 `config-x-cell-gat` 的整细胞留出数据合同，同时将 `pert_model` 换成 `exphormer_w_mpnn`；
- 提交并同步 Gitee/GitHub，运行代码对应提交为 `bee36d5`。

## 正在等待

E205 profile 已进入 `tmux safeconf_e205_profile`。固定任务为：

```text
RPE1 × seed 1 × batch size 64 × 1 epoch
```

队列状态文件：

```text
/home/yyf/data/txpert_official_20260802/e205/profile/E205_PROFILE_QUEUE_STATUS.json
```

记录时两张 Quadro RTX 6000 均被其他计算占用：GPU 0 空闲约 9,488 MiB，GPU 1 空闲约 9,132 MiB。等待器要求至少 20,480 MiB 空闲，且没有占用超过 1,024 MiB 的其他计算进程，因此尚未启动 Exphormer。

## 队列调整

E204 监管器收到正常停止信号时没有活动训练子进程，32 项均未启动，记录已保留。E205 profile 目前优先使用下一张满足条件的 GPU。没有终止或干扰其他用户的 GPU 进程。

## profile 通过门

只有以下条件全部成立，才允许扩为 16 个正式模型：

- 状态为 `COMPLETE`；
- 训练完整走完 1 轮；
- 模型家族记录为 `TxPert-Exphormer`；
- 只改变扰动传播结构；
- 目标扰动表达读取数为 0；
- 目标测试数据未构造；
- `last.ckpt` 存在且非空；
- 峰值显存和单轮耗时已记录。

profile 的训练损失或验证值不用于选择 target、seed、权重或删除任务。

## 下一步

1. profile 通过：建立 4 target × 4 seed 的正式队列；
2. 16 个 checkpoint 全部封存后生成无真值预测；
3. 先封存 Exphormer 幅度、五分量 SafeConf 和固定 SafeConf-M，再读取现有目标结果；
4. 主要检查 20% 复核效用相对幅度的配对增量，完整保留每个目标背景的结果；
5. E205 结束后再启动 E204，避免两条问题同时争抢 GPU。

