# E208｜D1 上游预测合同

冻结时间：2026-09-14；冻结时尚未读取测试扰动表达，也未生成目标真实误差。

## 1. 软件与数据版本

- PerturBench 源码提交：`c84038bc1ea409aa54f3832cfa6f34f5059adf0c`；
- `data/jiang24.yaml` SHA-256：`9abe81ebc6c51a0db692454d612d8c43c86cf309b8e2b222595018720b0b78e3`；
- `latent_best_params_jiang24.yaml` SHA-256：`3149703a218c8ef42a2ed42b93e86138738545a7a26d1721456eb01e90169a62`；
- `linear_best_params_jiang24.yaml` SHA-256：`798180ada8e75f521d0def1347cec1b95a171216f6e64dd96c6139d3d8500a39`；
- 数据与官方切分哈希见 `D0_TASK_INVENTORY_REPORT.md`。

正式环境使用独立的 Python 3.11 环境，安装上述固定提交，不修改 PerturBench 源码。环境锁文件和 GPU/软件版本在第一次 smoke 通过后写入，不事后改包版本追逐结果。

## 2. 正式上游模型

### 主模型族：LatentAdditive

采用 PerturBench 为 Jiang24 发布的 `latent_best_params_jiang24.yaml`：4 层编码器、宽度 2304、潜变量 64、dropout 0.3、学习率 5.651369387154785e-05、权重衰减 5.768768409829649e-10，同时向编码器和解码器注入细胞系与处理状态。

- 随机种子：1、2、3、4；
- 每个种子最多 400 epochs，最少 5 epochs；
- early stopping 监控验证集 `val_loss`，patience 50；
- batch size 2000，8 个数据加载进程；
- 单 GPU、混合精度 16、`deterministic=true`；
- 四个种子分别保存最佳验证 checkpoint，不按测试结果挑种子。

主模型四种子用于生成预测质心、预测幅度和同结构分歧。四种子均须完成或均停止；不只保留表现最好的一两个种子。

### 同合同强简单基线：LinearAdditive

采用 PerturBench 官方 `linear_best_params_jiang24.yaml`：学习率 0.0014212823361302294、权重衰减 1.0079009696147276e-08，并注入细胞系与处理状态。固定随机种子 1，训练与 early-stopping 规则同上。该模型用于检验复杂模型是否真正超过官方强简单基线，不参与 LatentAdditive 四种子分歧。

另保留“不改变对照表达”的 no-change 基线。正文必须同时报告主模型相对 LinearAdditive 和 no-change 的预测误差，防止风险排序建立在无效预测器上。

## 3. 正式任务

任务清单固定为 D0 生成的 224 个主测试任务：4 个细胞系（K562、MCF7、HT29、HAP1）× 3 个处理状态（IFNG、INS、TGFB）中的单基因扰动，每项不少于 30 个测试细胞。组合扰动、IFNB/TNFA 状态和非主细胞系不进入主要端点，也不能在揭盲后补进来替换失败结果。

使用官方 train/val/test 切分。训练只允许读取 train 表达，early stopping 只允许读取 val 表达。测试状态中的未扰动对照是部署输入，可以用于产生反事实预测；测试状态中的受扰动表达是目标真值，在预测与风险分数封存前禁止读取。

## 4. 防止自动测试泄漏

PerturBench 的默认训练配置包含 `test: true`，会在训练后自动执行测试。E208 的所有正式训练命令必须显式覆盖：

```text
test=false
```

checkpoint 生成后先退出训练进程，不调用 `trainer.test`。随后从固定的 224 项任务表生成 prediction dataframe，只用测试状态对照细胞产生反事实预测。四种子预测、LinearAdditive 预测和全部风险分数写入只读封存目录并记录 SHA-256；完成 GitHub/Gitee 存档后，才单独授权真实误差评价进程读取测试扰动表达。

## 5. Smoke 与资源门

正式训练前允许做不超过 1 epoch 的工程 smoke，只检查：环境能否导入、H5AD 能否打开、训练批是否能前向/反向、显存是否小于 22 GB、checkpoint 能否恢复、prediction dataframe 能否生成。Smoke 输出不得用于论文指标，也不得用于改变正式模型、种子或任务。

E205 的两张 GPU 队列优先。E208 仅在 E205 正式训练完成或释放一张满足显存门槛的 GPU 后启动；不在同一张 Quadro RTX 6000 上叠加两个正式训练进程。

## 6. 失败处理

- 单纯 OOM 时只允许按预先顺序把 batch size 从 2000 降到 1000、500、250；模型层数、宽度和潜变量不变；
- 四种子中某个任务出现可恢复的系统中断，允许从同一 checkpoint 继续并保留日志；
- 数据字段或预测器无法遵守测试真值隔离时，E208 记为 `PROTOCOL_BLOCKED`，不得改成先评价再补封存；
- 任何公式的结果不决定模型是否保留。全部主任务与完成的正式模型均进入结果表。
