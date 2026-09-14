# E208｜Jiang24 D0 元数据任务清单

生成时间：2026-09-14 14:18（Asia/Shanghai）

状态：`PASS`

## 已核准的数据合同

- 表达矩阵规模：1,628,476 个细胞 × 15,473 个基因；
- 扰动字段：`condition`；未扰动对照值：`control`；
- 细胞背景字段：`cell_type`；附加处理状态字段：`treatment`；
- 官方切分与表达矩阵细胞条码一一对应，无缺失、无重复；
- 主分析固定为 4 个细胞系（K562、MCF7、HT29、HAP1）× 3 个处理状态（IFNG、INS、TGFB）；
- 主分析共有 12 个细胞状态、224 个单基因测试任务、53 个不同基因和 214,901 个测试细胞；
- 224 个主任务均不少于 30 个测试细胞，且其扰动基因在训练层均有历史支持。

## 完整性记录

| 文件 | SHA-256 |
|---|---|
| `jiang24_processed.h5ad.gz` | `dd890a8019b8a615010963b32e6e28ce42fd0b94a749602bc93efb2fa8af56cc` |
| `jiang24_processed.h5ad` | `5d876c0fa5770dc632ad8ed8b211ad7aef00ccc93481f6ac029d439a0d7cd4d9` |
| `jiang24_split.csv` | `5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d` |
| `E208_PRIMARY_TEST_TASKS.csv` | `53e3a5f363bf70a81f4074b0c001712698db8acd0852354745989db091d9d4de` |

## 访问边界

本步骤只读取 `obs` 元数据、细胞条码和官方切分，不读取 `X` 或 `layers[counts]` 的表达数值。测试扰动表达没有用于公式选择、模型选择或风险打分。原始清单保存在数据盘 `jiang24/task_inventory/`，不把大型数据文件提交到 Git。

下一步是冻结上游模型、简单预测基线、训练轮数和资源上限；预测与全部风险分数完成哈希封存后，才允许读取测试表达生成真实误差。
