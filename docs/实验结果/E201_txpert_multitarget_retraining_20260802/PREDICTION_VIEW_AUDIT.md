# E201 零真值预测视图审计

审计日期：2026-08-02

## 目的

正式模型封存后，预测前还需要读取 target 的扰动标签、细胞系和实验 batch，
但 SafeConf 风险特征封存前不应取得 target 扰动表达。为此建立单独的
`E201_prediction_blind` H5AD：保留四个公开细胞系的 obs、var 和 control 表达，
将所有扰动细胞的 X 物理置零，并删除 `uns` 中的差异表达结果。

## 构建记录

第一次构建已经正确写出 H5AD，最后创建 runtime hardlink 时失败。原因是
`TxPert/cache` 整体就是数据盘 cache 的符号链接，两个参数解析后是同一路径。
失败产物保留在数据盘的
`E201_prediction_blind_attempt1_failed_runtime_alias`，没有覆盖后续正式视图。

修正后第二次构建成功。正式视图：

| 项目 | 结果 |
|---|---:|
| 行数 | 581,172 |
| 基因 | 3,352 |
| control 行 | 39,165 |
| 扰动行 | 542,007 |
| control 非零值 | 60,248,865 |
| 扰动表达非零值 | 0 |
| 排除的 `K562_adamson` 行 | 51,316 |
| `uns` keys | 0 |

H5AD 大小为 140,792,831 bytes，SHA-256 为
`85f93d1b29ded34d9dcece9ecdba1ef722a3f14aeedbfbe740eed9f045fbe486`。
manifest SHA-256 为
`27448df0378aab32e1a9fd22bf20c18c90089816cee6c28b9710cd2d6f812e7d`。

## 独立复核

独立审计程序没有调用构建函数，也没有读取 source 扰动表达。它重新完成了：

- source 中四个公开细胞系的 obs 与 blind view 逐行、逐列比较；
- var 完整比较；
- 39,165 个 control 的表达稀疏矩阵逐值比较；
- 542,007 个扰动行分块检查非零值；
- source、blind H5AD 和 manifest 的 SHA-256 复核。

结果为 PASS：control mismatch 为 0，最大绝对差为 0，blind 扰动表达非零值为
0，审计过程中打开的 source 扰动表达行数为 0。数据盘审计记录位于
`DATA/txpert_official_20260802/cache/E201_prediction_blind/E201_BLIND_PREDICTION_VIEW_AUDIT.json`。

obs 顺序 SHA-256：
`3c541cded88c99e6ec91d83ead6724e2bb0b88b1472411fc03b357cf42e92e93`；
condition 顺序 SHA-256：
`d738a9b91477ab10152848fbdf708ba012174e74dadfd770b3eeede0fb5cf766`。

## checkpoint 推理预检

使用 RPE1 的早期工程 checkpoint 和正式公开 STRING-GAT 配置，在 CPU 上从
blind prediction view 构造 test dataloader 并执行一个 2-cell batch。实际 test
dataset 为 67,034 行、3,352 基因；checkpoint、STRING 图和扰动 ID 映射均可
读取，输出形状为 `2 × 3,352`，全部有限。把 dummy `batch.x` 从全 0 改为全 1
后，预测逐元素完全相同，最大绝对差为 0。该预检只证明当前公开推理路径不使用
dummy X，正式运行仍会逐 batch 检查非零值并记录访问计数。

## 边界

这个视图只用于 checkpoint 已全部封存后的目标预测。模型训练仍使用四份更严格的
target-specific blind training H5AD。预测视图中虽然有 target 任务标签，但没有
target 扰动表达，因此可以先生成多种子分歧和其他预测前风险量。正式真值由后续
独立 release 程序在风险特征双远程封存后提取。
