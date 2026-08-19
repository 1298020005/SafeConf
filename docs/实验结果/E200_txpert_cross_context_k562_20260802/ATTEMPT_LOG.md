# E200 尝试记录

## 2026-08-02：prepare attempt 1

- 程序完成官方 ZIP、H5AD、split 和 checkpoint 的哈希读取，并以 backed 模式读取
  AnnData 元数据；
- 在统计 K562 control 数量时停止：`control` 是 Pandas categorical，不能直接执行
  `sum()`；
- 尚未写出 prepare 表或状态文件，未读取 K562 扰动表达矩阵，也未执行模型；
- 修正为 `astype(bool).sum()`；资产、拆分、特征和评价判据均不改变；
- 资源：墙钟 57.08 秒，峰值内存 885,372 KiB。

## 2026-08-02：prepare attempt 2

- 修正 Pandas categorical 计数后重新执行同一冻结程序；
- 官方 ZIP、H5AD、split、subgroup 和 checkpoint 的字节数与 SHA-256 全部匹配；
- TxPert 与 scPertEval 源码仓库均在冻结 commit 且工作树干净；
- 严格 context-only 为 580 个任务，其中主分析 566 个、低细胞数敏感性 14 个；
- validation-only 202 个、train/validation 均未见 33 个、背景与扰动同时未见 272 个，全部从主分析剔除并单列；
- prepare gates：38/38 PASS；墙钟 57.24 秒，峰值内存 885,444 KiB。

## 2026-08-02：runtime leakage smoke attempt 1

- 设备固定为 CPU，batch size 为 8，输出基因数为 3,352；
- 先用原始扰动真值张量执行一次模型前向，随后将该真值张量全部清零，在不改变对照、扰动编号、剂量和细胞背景的情况下再次前向；
- 两次预测 SHA-256 均为 `f29591eaa840aef0cf4de0c0acc8e41418f82b59d9d1a2d3bc46f85ead7ea916`，逐元素完全相等，最大绝对差为 0；
- 前后对照输入 SHA-256 均为 `6ef6c917aac1f71766aabf36ce4701aad8d392c3a8316bec40ba5b29bdffe6b3`；
- 原始运行记录 SHA-256 为 `ed7b8a2880c9a01cbd5daa2ed6eb2968ffe8a2dd85b44e58ccaa43cd48e88466`；
- 结论：runtime target leakage smoke PASS；墙钟 69.45 秒，峰值内存 23,395,788 KiB。

## 2026-08-02：official GAT inference attempt 1

- 使用 GPU 1、batch size 64 和官方 `K562_unseen_cell_gat.ckpt`；
- 生成 150,472 个 K562 扰动细胞 × 3,352 个基因的 prediction、truth 和 batch-matched control；
- 覆盖官方 1,087 个测试任务和 48 个批次；
- 运行成功；墙钟 211.42 秒，峰值内存 34,524,216 KiB。

## 2026-08-02：official general baseline attempt 1

- 按官方 cross-cell baseline 代码从 RPE1、HepG2 和 Jurkat 构造训练背景扰动启发式；
- 生成与 GAT 同尺寸的 150,472 × 3,352 矩阵；
- 运行成功；墙钟 391.87 秒，峰值内存 42,915,612 KiB。

## 2026-08-02：prediction seal and alignment audit

- 两套预测的原始文件均已记录字节数和 SHA-256；
- GAT 与 general baseline 的行名、基因名、任务标签、细胞背景和批次顺序一致；
- truth 矩阵逐元素完全相同，最大绝对差为 0；
- 最初要求两套 control 逐位相同时停止；追查发现差异为浮点写出精度，504,382,144 个值的 RMS 差为 `4.3524381881028845e-08`，最大差为 `9.5367431640625e-07`，没有任何值超过 `1e-6`；
- 改为冻结的数值等价判据 `atol=1e-6, rtol=0`后通过，不改写原始产物；
- 设计已在打开真值前提交。封存前技术 QC 读取了首尾各 8 行并只检查有限性与整体极值；未计算任务误差、排名、相关性或任何裁决指标。

## 2026-08-02：pretruth feature release attempt 1

- 运行前本地、GitHub 和 Gitee 均位于冻结 commit `28f1fa3ece76f9c8ec9076e994a9650f92c713f4`；
- 重新校验 8 个允许输入的字节数和 SHA-256；
- 只读取 RPE1、HepG2 和 Jurkat 的 28,474 个对照细胞与 112,286 个严格训练扰动细胞；K562 扰动表达读取 0 行；
- 生成 580 个冻结任务的 `transfer_risk` 和 `predicted_magnitude`；75 个单背景任务用冻结中位数 `0.046303631` 填补跨背景离散度并保留标记；
- 任务数、目标预测数、训练细胞数、支持背景数、Z-score 和组合恒等式共 21/21 项 PASS；
- 尚未计算任务真实误差、风险相关、复核效用或任何结果裁决；
- 运行成功；墙钟 83.63 秒，峰值内存 2,633,496 KiB。

## 2026-08-02：formal evaluation attempt 1

- 所有冻结输入哈希、80,153 个严格目标细胞、580 个任务、预真值特征重算和 8,700 条 scPertEval 任务端点记录均已通过；
- 任务 RMSE、5 个冻结端点、5,000 次 bootstrap、路由效用和门槛表已计算并写入部分输出；
- 生成白底图时停止：`decile.sem` 被 Pandas 解析为 `sem()` 方法，Matplotlib 不能将方法对象作为 `yerr`；
- 未写出图、正式报告或 `E200_FINAL_STATUS.json`；已生成的 13 个表逐文件哈希后移至 `DATA/txpert_official_20260802/e200/failed_attempts/formal_attempt_001/`；
- 修正仅为将 `yerr=decile.sem` 改为 `yerr=decile["sem"]`；数据、任务集、端点、特征、bootstrap、门槛和统计代码均不改变；
- 资源：墙钟 227.26 秒，峰值内存 10,359,728 KiB。

## 2026-08-02：formal evaluation attempt 2

- 从已双推的纯展示修复 commit `f384b5101b83c12b2e675940b447cd410482ddbb` 重新执行完整流程；
- 完整性 PASS；预真值特征最大重算残差为 `2.220446049250313e-16`；
- `transfer_risk` 对 GAT centroid RMSE 的 Spearman 为 0.4240（95% CI 0.3506–0.4953），20% 复核效用为 0.3648（95% CI 0.2356–0.4813），经验路由门槛通过；
- `predicted_magnitude` 更强：Spearman 0.8797，复核效用 0.9133；`transfer_risk - predicted_magnitude` 的 ΔSpearman 和 Δutility 均为显著负值，新增价值门槛不通过；
- GAT 相对 general baseline 的 centroid RMSE 均值差为 -0.004224（95% CI -0.004633–-0.003814）；相对 batch-matched control 则为 +0.001886（95% CI 0.000582–0.003173）；
- 与 general baseline 相比，GAT 在 MSE 和 DE-AUPRC 上更好，Pearson 和 energy distance 无稳定差异，rank 更差；不写成全端点优越；
- GAT 与 general baseline 的任务误差排名 Spearman 为 0.9700（95% CI 0.9609–0.9769），表明当前信号主要反映两预测器共享的任务难度；
- 除运行时表外，attempt 1 与 attempt 2 的其余 12 个统计表字节级完全一致；
- 白底 PNG/PDF 图、报告和状态全部写出；墙钟 228.88 秒，峰值内存 10,271,180 KiB。
