# E201 训练队列恢复记录

日期：2026-08-09

状态：剩余 15 个作业的持久队列已经启动；目标扰动真值仍未释放。

## 已完成与未完成

- `K562/seed_1` 已完成 80 epochs，`global_step=368640`，训练期间读取目标
  扰动细胞 0 行；
- 原队列在启动 `RPE1/seed_1` 时失败，其余 14 个作业没有启动；
- 尚未运行 16-checkpoint family seal、正式目标预测、风险特征封存或真值释放。

## 失败原因

原队列把参数 `--python` 指向的虚拟环境解释器调用了 `Path.resolve()`。该解释器是
符号链接，解析后变成 Conda 基础解释器，从而绕过 TxPert 专属虚拟环境。
基础解释器中没有 `hydra`，所以
`RPE1/seed_1` 在模型和数据加载前以 `ModuleNotFoundError` 停止。

首个 `K562/seed_1` 是由已有训练进程完成，队列只负责等待该进程，因此没有触发
这一解释器路径错误。

## 修正

`run_e201_blind_training_queue.py` 现在：

1. 保留虚拟环境 `bin/python` 的绝对路径，不再解析其符号链接；
2. 在写正式运行目录前，用同一解释器和同一工作目录导入 `hydra`、
   `lightning`、`omegaconf`、`pandas`、`torch` 和 `gspp`；
3. 检查 `sys.executable` 和 `sys.prefix`，确认子进程没有逃逸到基础解释器；
4. 在每个剩余作业启动前重复运行上述检查，并把版本写入队列状态。

训练 adapter 没有修改，SHA-256 仍为
`274344472a2e19d91cf8e971e4e46cf08e39868a36d7892310281d1d847d5e7a`，
因此后续 checkpoint 与已经完成的 K562 checkpoint 仍使用同一正式训练实现。

重启前实际环境检查结果：Python 3.12.4、Hydra 1.3.2、Lightning 2.5.1、
OmegaConf 2.3.0、Pandas 2.2.3、PyTorch 2.6.0+cu124；`pip check` 无依赖冲突，
TxPert worktree 位于冻结提交
`08d82eea86746b044cf7531f4ec8c5f60e1cb73f` 且干净，GPU 1 无计算进程。

## 失败现场保留

失败产物没有删除，已移到：

- `DATA/txpert_official_20260802/e201/formal/_failed_attempts/RPE1/seed_1_attempt_001_20260802_hydra_missing/`；
- `DATA/txpert_official_20260802/e201/formal/_queue_history/attempt_001_20260802_failed/`。

原始文件哈希：

| 文件 | SHA-256 |
|---|---|
| `RPE1/seed_1/E201_RUN_STATUS.json` | `ea7fe87549635e6d82ba8c75b444b64d6bd8c87b80a4b53490147ec9036aa3c9` |
| `02_RPE1_seed_1.log` | `e4f0fb1617a38a344ea998183d08574a54785b7dd7100bf92d603d2993918948` |
| `E201_QUEUE_STATUS.json` | `9cce6e1bc600a3b3b6cdc632a8a2612606190198ea20ce814fcf31816dc3ce4f` |

## 继续执行规则

队列继续按冻结顺序运行剩余 15 个作业。任何作业只有达到 80 epochs、目标扰动
访问数为 0、checkpoint 可读且哈希完整时才算完成。16 个作业全部完成前不运行
目标预测；预测、风险表和基线全部封存并双远程推送前不释放目标真值。

## 实际重启

- 双远程代码提交：`6856948d3aa6ea333e3591da20538f51e37be01b`；
- 持久服务：`safeconf-e201-txpert-20260809.service`；
- 服务启动：2026-08-09 20:20:54 CST；
- 队列 PID：`1965937`；首个重启作业 PID：`1966303`；
- 队列先复核并跳过已完成的 `K562/seed_1`，随后于 20:21:06 启动
  `RPE1/seed_1`；
- 正式状态记录的 Python 路径、虚拟环境前缀、SafeConf 双远程提交、TxPert 提交
  和训练 adapter 哈希均符合冻结值；
- `RPE1/seed_1` 已进入 epoch 0，优化步从 0 开始正常增长，初始重建损失为有限值；
- 启动与检查期间读取目标扰动真值 0 行。

服务由用户级 systemd 托管，不依赖当前终端会话。队列内每个模型完成后才会启动
下一个模型；任何工程 gate 失败都会停止队列并保留现场，不会跳过失败作业。
