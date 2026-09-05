# E204 四目标训练权重工程验收

日期：2026-09-05  
状态：**PASS，允许进入正式训练队列**

## 验收目的

周老师提出的第二个用途是：训练时更重视不容易预测对的任务，再检查模型是否因此改善。E204 将 source-only difficulty 复制到每个训练细胞的 loss 权重。本轮只检查这套机制在四个整体细胞背景留出任务上是否正确执行，不判断模型效果。

## 结果

| 留出目标 | 训练记录 | 验证记录 | 单轮耗时（秒） | 训练权重回退 | 目标扰动表达读取 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| K562 | 294,951 | 80,340 | 639.46 | 0 | 0 | PASS |
| RPE1 | 273,003 | 76,950 | 594.09 | 0 | 0 | PASS |
| HepG2 | 314,391 | 89,117 | 692.16 | 0 | 0 | PASS |
| Jurkat | 282,132 | 78,682 | 615.85 | 0 | 0 | PASS |

四个作业都使用 seed 1 运行一个完整 epoch。训练 Dataset 应用条件权重，验证 Dataset 固定权重 1；非对照训练样本没有任何未匹配条件。运行期间没有构造 target test Dataset，`target_perturbed_cells_accessed` 均为 0。

## 审计修正

第一次 K562 profile 将验证集 80,340 行的两次遍历误记为 160,680 条训练权重回退。修正后，权重和覆盖审计只作用于训练 Dataset；如果真的出现未覆盖训练条件，启动器会写入 `FAILED_COVERAGE` 并以非零状态退出。修正提交为 `ab7ba1d`。

四个状态文件位于 `$DATA_ROOT/e204/profile_v3/<target>/seed_1_risk/E204_WEIGHTING_STATUS.json`，其 SHA-256 为：

| target | SHA-256 |
| --- | --- |
| K562 | `cd4d575bc106dbc9bc562998540f26842971537bc6a60f20102341954f7d1c26` |
| RPE1 | `6656e79e18c8ac3919bca2601d8a49fcaeb2aecf820845c2ddd30af734cfc304` |
| HepG2 | `d587a8bab1da592964568c2e345945421f32536023fcb3a4d87f7c8a07a50bd5` |
| Jurkat | `cd7883e8c3ac5560e9735da8bb33ada7180ba0e56d2d91428af97c3626aa9dca` |

## 下一步

E201 的 16 个 uniform 模型直接作为基线。正式队列新增 16 个 `risk_weighted` 模型和 16 个 `dispersion_only` 模型，每个目标四个种子。三组预测齐全后再统一读取 target 结果，按相同 target、相同 seed 做成对比较。主判断保持冻结：高难任务 RMSE 是否下降，以及全体平均 RMSE 是否没有恶化超过 5%。

本轮 profile 只证明程序可靠。一轮 validation Pearson delta 不是 E204 效果证据，不用于提前选权重、目标或种子。
