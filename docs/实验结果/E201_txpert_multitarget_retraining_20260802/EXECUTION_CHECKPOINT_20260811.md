# E201 执行检查点：第一轮四背景完成，K562 第 2 个种子运行中

记录时间：2026-08-11 15:50（Asia/Shanghai）

实验：`E201_txpert_multitarget_retraining`
性质：四个细胞背景整列留出（whole-context holdout）的盲训练阶段。本文只记录
执行状态，不含任何 target 真实扰动表达、误差、相关性或路由结论。

## 1. 本检查点回答什么

每轮将一个细胞背景完全作为 target：训练时可使用其 control 细胞，**不允许读取其
任何扰动后的细胞表达**；其余三个背景提供带扰动的 source 训练数据。四个 target
分别为 K562、RPE1、HepG2 与 Jurkat，每个 target 用 4 个随机种子训练，总共 16
个正式模型。

这补的是“整列/完整背景留出”这一维度。它不替代已有的小矩阵、随机缺格、跨数据集
或 chemical 结果；各类设置仍要独立报告。

## 2. 冻结内容

| 项目 | 固定值 |
|---|---|
| TxPert 代码 | `08d82eea86746b044cf7531f4ec8c5f60e1cb73f`，工作树干净 |
| 正式训练 adapter | `tools/scripts/txpert_blind_training_adapter.py` |
| adapter SHA-256 | `274344472a2e19d91cf8e971e4e46cf08e39868a36d7892310281d1d847d5e7a` |
| 模型 | 公开 TxPert STRING-GAT；不冒充论文中未公开的 PxMap/TxMap 配置 |
| 每个模型 | 4-layer GATv2、hidden 128、2 heads、batch size 64、80 epochs、AdamW |
| 训练队列 | `tools/scripts/run_e201_blind_training_queue.py`，SHA-256 `4ddc9e201097ac7423d9fd5410e042dec9eb1a0efb478bdc060c91cae986e3e1` |
| 目标真值 | `NOT_AUTHORIZED`；未完成 16 个模型前不得预测、计算风险或打开 target 真值 |

每次启动前，队列都用实际调用的虚拟环境导入 `hydra`、`lightning`、`omegaconf`、
`pandas`、`torch`、`gspp`，并检查 `sys.executable` 与 `sys.prefix`。运行环境为
Python 3.12.4、PyTorch 2.6.0+cu124、Lightning 2.5.1、Hydra 1.3.2、Pandas 2.2.3。
2026-08-09 的解释器别名故障与修复见
[`RECOVERY_LOG_20260809.md`](RECOVERY_LOG_20260809.md)，失败现场仍保留在数据盘。

## 3. 已完成的正式训练

下表的 `best source validation` 仅来自三个 source 背景的 validation split，用于
确认训练正常完成；它不是 target 测试表现，不能写成跨背景结果。

| target 背景 | seed | 开始 → 结束 | epochs / steps | 训练时长 | best source validation | target 扰动访问 | 状态文件 SHA-256 |
|---|---:|---|---:|---:|---:|---:|---|
| K562 | 1 | 08-02 07:24:49 → 08-02 20:00:36 | 80 / 368,640 | 45,292.25 s | 0.453448 | 0 | `9f1f93b0ba992440a55db087ca07a6bb945011d067ce944cb67bff9cdd2509c2` |
| RPE1 | 1 | 08-09 20:21:06 → 08-10 07:47:18 | 80 / 341,200 | 41,120.76 s | 0.407204 | 0 | `07125f73cece2afd78e36fedf069d2ff7f769c9504f78be8d04bcf3ebd3cec1d` |
| HepG2 | 1 | 08-10 07:47:32 → 08-10 21:01:52 | 80 / 392,960 | 47,600.34 s | 0.482933 | 0 | `7cbcaa47a231d9d6f2b5204ba863ea4e90fd17fe5b19ad47ca318b4035c86861` |
| Jurkat | 1 | 08-10 21:02:00 → 08-11 09:04:45 | 80 / 352,640 | 43,308.48 s | 0.476302 | 0 | `8867d79cc9f41a7f2cc12ff22ae5b24d7517392c916d357b85a4ef3137df6b1c` |

四项均满足：`status=COMPLETE`、`current_epoch=80`、
`target_test_dataset_constructed=false`、`target_perturbed_cells_accessed=0`。各 run
目录同时保留 `last.ckpt` 与 source validation 最佳 checkpoint，以及它们的大小和
SHA-256，详见各自的 `E201_RUN_STATUS.json`：

- `DATA/txpert_official_20260802/e201/formal/K562/seed_1/`；
- `DATA/txpert_official_20260802/e201/formal/RPE1/seed_1/`；
- `DATA/txpert_official_20260802/e201/formal/hepg2/seed_1/`。
- `DATA/txpert_official_20260802/e201/formal/jurkat/seed_1/`。

## 4. 正在执行与队列顺序

| 队列位置 | target / seed | 状态 | 观察时间 | 说明 |
|---:|---|---|---|---|
| 5 / 16 | K562 / 2 | RUNNING | 08-11 15:50 | 已到 epoch 43，GPU 训练进程 PID `2346937`；使用 target treatments 为 0 的物理盲视图 |
| 6–16 / 16 | RPE1、HepG2、Jurkat / seed 2；四个背景 / seeds 3–4 | QUEUED | 08-11 15:50 | 顺序已冻结，前项完整通过后才启动后项 |

队列由用户级 systemd 单元 `safeconf-e201-txpert-20260809.service` 托管，监督进程
PID 为 `1965937`。它不依赖当前 shell；若任一作业失败，会停止并保留失败现场，
不会静默跳过该作业。此刻队列状态文件 SHA-256 为
`c2b24058203b1e278062f375ada37aabec77ec3ddad3272beebf6066dca46bd7`。

## 5. 资源与并行决定

本检查点时，两张 Quadro RTX 6000（各 24 GiB）状态如下：

| GPU | 已用显存 | 占用任务 | 决定 |
|---:|---:|---|---|
| 0 | 13.91 GiB | 系统 VLLM 服务 | 不抢占、不加入 E201 |
| 1 | 11.56 GiB（运行中） | K562 seed 2 | 只运行一个正式 TxPert 作业 |

此前完整 RPE1 训练的显存峰值为 14.66 GiB allocated、15.18 GiB reserved。两项
batch-64 正式训练不能安全放在同一张卡上；为了追求“并行”而改 batch size、混合不同
训练合同或与系统服务抢卡，会使 16 个模型不再可比。因此本轮保持单卡串行，而不是
加入看似更快但不可审计的并发作业。

数据盘当前可用约 2.8 TiB；原始 H5AD、checkpoint、日志和状态文件均保留在 `DATA/`
对应的数据盘路径，不提交到 Git。

## 6. 真值隔离与下一道门

到本检查点为止：

- 4 个完成模型的 `target_perturbed_cells_accessed` 均为 0；当前 K562 seed 2
  使用的物理盲视图中 `n_target_treatments=0`；
- 未构造 target 测试集，未读 target perturbation expression；
- 尚未生成 E201 正式预测、风险特征、general baseline 或任何 target error；
- 因此目前没有 E201 的生物学正/负结果可报告。

16 个模型都完成后，严格按如下顺序执行，不能倒置：

1. `seal_e201_txpert_checkpoint_family.py` 检查 16 个 `last.ckpt`、训练条件与哈希；
2. `run_e201_txpert_sealed_prediction.py` 在零真值预测视图上生成 16 份预测；
3. `run_e201_pretruth_risk_features.py` 与 `build_e201_official_general_baseline.py`
   只用 source 信息、target control 和冻结预测计算并封存风险量/基线；
4. 将封存产物及其哈希提交并推送到 Gitee、GitHub；
5. `release_e201_target_truth.py` 才可打开 target 真值；
6. `run_e201_formal_core_evaluation.py` 计算误差、四种子 family 指标、相对预测幅度
   的增量、基线比较和按 condition cluster 的 bootstrap 区间。

评价定义、指标与失败规则已在
[`FORMAL_CORE_EVALUATION_FREEZE.md`](FORMAL_CORE_EVALUATION_FREEZE.md) 和
[`TARGET_RELEASE_AND_EVALUATION_FREEZE.md`](TARGET_RELEASE_AND_EVALUATION_FREEZE.md)
冻结；训练中不因 source validation 数字改变模型、种子、任务或阈值。

## 7. Git 留存

本次更新前，代码基线为 `fc334d61393922c9dc216ab289e02d6652e28aa4`，分支为
`exp/task-risk-audit-20260611`。该基线已在 Gitee `origin` 和 GitHub `github` 对齐。
本文件只记录执行检查点；不把 `agents/` 下已有的未提交文件或任何原始数据、模型
checkpoint 混入本次提交。
