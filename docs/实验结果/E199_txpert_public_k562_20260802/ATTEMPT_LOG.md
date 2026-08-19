# E199 尝试记录

## 2026-08-02：运行时目标表达隔离测试

冻结提交：`d165ca9a737d460623105fc087142b4aece2cdf4`

### attempt 1：GPU，未通过逐位一致门

- 设备：第二块 NVIDIA Quadro RTX 6000；
- 模型：官方 `K562_unseen_pert_gat.ckpt`；
- 批次：8 个测试细胞，5,000 个基因；
- 操作：保持 control、扰动索引和 embedding 不变，把 `batch.x` 清零后重新预测；
- 结果：`exact_equal=False`，最大绝对差 `2.384185791015625e-07`；
- 裁决：按冻结的逐位一致规则记为 `FAIL_EXACT`，未用该结果放行 formal。

该差值处在单精度 GPU 归约的数值量级，但本次没有追加事后容差，也没有把原因直接
判定为 CUDA 非确定性。保持代码、模型、批次和判据不变，改在 CPU 上复核。

### attempt 2：CPU，正式通过

- 模型、批次和清零操作与 attempt 1 相同；
- `prediction_exact_equal=True`；
- `prediction_max_abs_delta=0.0`；
- 清零前后 prediction SHA-256 均为
  `cf674d83722a37b478d8b00da14b8c81cb5a8b7abc90fd3b97f44e9838da068a`；
- 前后 control SHA-256 均为
  `22705db75f27d2bd5f9dece5cb5251137858f9ed423f102acd669851de54321e`；
- 外部原始 JSON SHA-256：
  `e692196a9a9062f4542ec9a835f01adf2ff5d448af4dc21e1da99c46700e2c64`。

CPU 结果满足冻结的逐位一致门。它与静态调用图共同证明，官方预测路径虽会把
`batch.x` 保存为 ground truth，但目标表达没有进入 `forward/sample_inference`。
后续批量推理可以使用 GPU；GPU 输出仍需通过跨模型任务顺序、有限值和重复抽查门。

## 2026-08-02：预特征生成 attempt 1

- 冻结提交：`c0653ca5b969336280c86e409df096972b161ed8`；
- 程序在读取已封存预测和公开图后，尚未进入逐任务特征计算时失败；
- 错误：Pandas 2.2 在 `GroupBy.apply(..., include_groups=False)` 中移除了
  `target` 列，后续邻居计数触发 `KeyError: 'target'`；
- 输出：未写入 `pretruth_release`，未读取目标表达，未计算任何误差或相关；
- 修正：显式遍历每个 `target` 分组，在保留分组列的前提下逐组执行
  `nlargest(20)`；特征定义和裁决规则不变。

## 2026-08-02：正式评价 attempt 1

- 冻结提交：`ae44f86a6376bd54eab3b17359ab36ad7e7fdcbe`；
- 输入哈希、远端提交、预测矩阵对齐和预特征重算均已通过，程序尚未进入
  scPertEval 指标计算；
- 错误：当前 AnnData 0.11 的 `concat` 只接受 `inner/outer`，不接受
  `join="exact"`；
- 输出：未创建 `formal_evaluation`，没有产生或查看任何正式指标；
- 资源：峰值内存 10,166,628 KiB，墙钟时间 76.62 秒；
- 修正：连接前已逐项确认 5,000 个基因名称和顺序完全相同，因此把连接参数改为
  该版本支持的 `join="inner"`。基因集合、样本、端点和裁决规则均不改变。

## 2026-08-02：正式评价 attempt 2（主动停止）

- 提交：`8941fe84f27adf1ff527011bb01eb6d1f97f649f`；
- 程序已完成 scPertEval 主端点的内存计算，进入 TxPert 次要端点，尚未创建
  `formal_evaluation`，没有写出或检查任何正式得分；
- 发现：`batch-matched control` 的预测差值恒为零。官方 `pearson_delta` 会把
  常量输入的 NaN 显式记为 0，但 `RetrievalMetric` 会继续排序 NaN，输出的有限名次
  没有数学意义；
- 裁决：主动中断，不接受该伪名次。Pearson 保留官方 0 分约定，retrieval 改为
  明确的 NA，并在逐任务表中写入定义状态；
- 性能修正：官方 retrieval 在每个参考扰动上重复计算同一预测均值。预先计算一次
  差值均值并以单行矩阵传入，数值定义不变；
- 资源：峰值内存 24,666,932 KiB，墙钟时间 246.97 秒。主端点和 SafeConf 的
  冻结判据未改变。

## 2026-08-02：正式评价 attempt 3

- 提交：`0eef2c99234b0f2ae582143db5a3e88a422c29d9`；
- 哈希、矩阵对齐、证书、scPertEval 主端点和 TxPert 次要端点均已在内存中完成；
- 错误：风险分析合并封存特征与证书时，两表都有 `diversity_lower_bound`，Pandas
  自动生成 `_x/_y` 后，程序按无后缀原名取列触发 `KeyError`；
- 输出：程序在统计汇总阶段停止，仍未创建 `formal_evaluation`，也未显示或检查
  任何正式结果；
- 修正：显式使用后缀，先要求证书重算值与封存值最大差不超过 `1e-12`，再只保留
  封存列用于风险分析；
- 资源：峰值内存 24,669,648 KiB，墙钟时间 250.76 秒。评价样本、端点、特征和
  gate 均不改变。
