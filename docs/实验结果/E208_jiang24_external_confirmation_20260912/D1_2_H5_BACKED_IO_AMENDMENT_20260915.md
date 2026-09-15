# E208｜Jiang24 延迟读取实现补充登记

登记时间：2026-09-15

状态：`CONFIG_EQUIVALENCE_PASS / DATA_BATCH_SMOKE_PENDING`

登记时点：E208 正式模型尚未训练，测试扰动表达尚未读取，真实误差尚未生成。

## 1. 需要解决的工程问题

Jiang24 包含 1,628,476 个细胞和 15,473 个基因，完整 H5AD 为 93,532,364,449
字节。官方 `jiang24.yaml` 使用 AnnData backed 句柄，但构造训练和验证数据集时会把
对应表达切片装入内存。服务器总内存为 125 GiB；元数据预检单进程已观察到约
35 GiB 占用，继续把训练与验证表达整体物化存在系统 OOM 风险。

固定的 PerturBench 提交 `c84038bc1ea409aa54f3832cfa6f34f5059adf0c` 已提供
`H5LitModule` 和带对照匹配的
`perturbench.data.datasets.h5.SingleCellPerturbationWithControls.from_h5`。正式 E208
改用该提交自带的延迟读取实现：每个训练批只从 HDF5 读取当前细胞及其匹配对照，
不把整个训练表达矩阵载入内存。

## 2. 不改变的科学合同

- 数据文件仍为同一 `jiang24_processed.h5ad`；
- 切分仍为同一 `jiang24_split.csv`；
- 扰动列仍为 `condition`，对照值仍为 `control`；
- 细胞背景仍为 `cell_type`，处理状态仍为 `treatment`；
- 基因轴仍为全部 15,473 个基因，不增加特征筛选；
- 对照仍在同一 `cell_type × treatment` 内匹配；
- 仍使用 `LinearModelPipelineControls`；
- LatentAdditive、LinearAdditive 的结构、超参数、种子、训练轮数、early stopping
  和损失函数均不改变；
- `test=false`，正式训练不调用 `trainer.test`。

改变的只有数据访问实现：从“先物化训练/验证表达切片”改为“按批延迟读取”。缓存
固定为 0，防止不同运行因缓存状态产生无法记录的内存差异。

## 3. 配置展开核验

使用正式 LatentAdditive 配置与全部 H5 覆盖参数执行 Hydra `--cfg job`。该命令只
展开配置，不实例化数据模块、不打开表达矩阵、不训练模型。展开结果保存在数据盘：

```text
DATA/perturbench_e208/config_preflight_h5/latent_h5_cfg_job.yaml
SHA-256 49bac480ccf7ed1e0bc48f59b35079e9c7152a6be5d171191ae23bc8e67259f0
```

核验结果：

| 字段 | 展开值 |
|---|---|
| `train` | `true` |
| `test` | `false` |
| `seed` | `1` |
| data module | `perturbench.data.modules.H5LitModule` |
| iterator | `h5.SingleCellPerturbationWithControls.from_h5` |
| cache | `0` |
| batch size | `2000` |
| workers | `8` |
| max/min epochs | `400 / 5` |
| deterministic | `true` |
| model | `LatentAdditive`，4 层、宽 2304、潜变量 64 |

展开值与 D1 上游预测合同一致。

## 4. 下一道执行门

E205 继续独占两张 GPU。E205 释放资源后，先按原合同各执行一个 LatentAdditive 与
LinearAdditive 的单批 smoke：

1. 只允许 1 个训练批和 1 个验证批；
2. 检查前向、反向、对照匹配和 15,473 基因轴；
3. 记录主机内存、GPU 峰值显存和批耗时；
4. 保存 checkpoint 并在新进程中恢复；
5. 检查运行目录不存在测试评价产物；
6. batch size 2000 若 OOM，只按 2000、1000、500、250 的既定顺序下降。

只有 smoke 全部通过，才启动四个 LatentAdditive 种子和一个 LinearAdditive 正式
模型。延迟读取解决内存风险，不代表模型训练门已经通过。

## 5. 已启动的等待监管器

2026-09-15 19:36（Asia/Shanghai）已启动 tmux 会话：

```text
safeconf_e208_after_e205_h5_smoke
```

状态文件位于：

```text
DATA/perturbench_e208/after_e205_smoke_supervisor_20260915/
E208_AFTER_E205_SUPERVISOR_STATUS.json
```

监管器每 60 秒读取 E205 队列状态。只有 E205 达到 16/16、无永久失败且仍未授权读取
真值时，才重新核验 93 GB H5 哈希并运行 E208 smoke。只有 CUDA OOM 可以依次尝试
2000、1000、500、250 四个已登记批量；数据、配置、checkpoint 或隔离错误会立即
停止。Smoke 通过后状态写为 `SMOKE_PASS_FORMAL_QUEUE_NOT_STARTED`，不会擅自打开
测试真值，也不会在正式队列尚未验收时直接启动五个模型。
