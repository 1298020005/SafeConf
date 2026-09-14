# E208｜D1 环境与配置预检

时间：2026-09-14

状态：`CONFIG_PREFLIGHT_PASS / DATA_BATCH_SMOKE_PENDING`

## 独立环境

- 路径：`/home/yyf/.venvs/perturbench-c84038bc`（仅服务器本地，不进入 Git）；
- Python 3.12.4；
- PyTorch 2.6.0 + CUDA 12.4；
- torchvision 0.21.0 + CUDA 12.4；
- Lightning 2.5.1；
- pandas 2.3.3；
- AnnData 0.13.3.post0；
- Scanpy 1.11.1；
- SciPy 1.18.1；
- PerturBench 0.0.1，源码提交 `c84038bc1ea409aa54f3832cfa6f34f5059adf0c`；
- NVIDIA 驱动 535.183.06；GPU 为两块 Quadro RTX 6000 24GB；
- `python -m pip check`：`No broken requirements found`。

环境从当前已稳定运行 TxPert 的 CUDA 12.4 环境复制后独立安装 PerturBench 依赖，不会修改 E205 正在使用的原环境。第一次直接解析依赖时发现包管理器准备安装另一套 PyTorch/CUDA，已在安装前取消；没有影响 E205 进程或正式结果。

## Hydra 配置预检

使用主模型正式命令的同一组覆盖参数运行 `--cfg job`，确认：

- `test: false`；
- `seed: 1`；
- 数据文件为 `jiang24_processed.h5ad`；
- 官方切分为 `jiang24_split.csv`；
- 模型为 `perturbench.modelcore.models.LatentAdditive`；
- `max_epochs: 400`、`min_epochs: 5`、`patience: 50`；
- `batch_size: 2000`、`num_workers: 8`；
- `deterministic: true`；
- 模型宽度、层数、潜变量、dropout 和学习率与冻结合同一致。

该步骤只展开配置，不实例化数据模块，不训练模型，也不读取测试表达。

## 尚未通过的门

完整 H5AD 的 `obs` 元数据在当前 AnnData 版本下单进程可占用约 35GB 内存。预检期间曾误启动多个只读检查进程，发现后已全部终止，内存恢复；E205 两个 GPU 训练进程未被终止或重启。

因此正式 data-batch smoke 延后到 E205 释放 GPU 后进行，并强制单进程、一次只开一个 H5AD 句柄。只有完成一个训练 batch 的前向/反向、显存门和 checkpoint 恢复后，才启动四种子正式队列。
