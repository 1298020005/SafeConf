# E62 外部数据获取状态

这份记录区分“本地可计算的数据”和“已找到但当前网络通道未打通的数据”，避免后续汇报把候选数据说成已经做完。

## 已可直接用于实验

| 数据线 | 状态 | 当前用途 |
|---|---|---|
| scPerturb + scPerturBench | 83/83 本地文件已核验可读 | 基因单扰动、组合扰动、化学扰动、剂量、调控、多模态和跨物种设置 |
| OpenProblems NeurIPS 2023 | 35/35 个官方对象已完成，共 53.421 GB；全部通过文件大小和 HDF5 根节点可读性检查 | PBMC 化学扰动、donor/cell-type holdout、基线 multiome prior |
| Tahoe-100M | 3,388/3,388 原始 parquet 分片已完成，约 337.645 GB | 更大规模的上下文、细胞类型和药物条件分析 |

## 已识别但尚未落盘：X-Atlas/Orion

- 官方数据页：[Figshare article 29190726](https://figshare.com/articles/dataset/29190726)
- 内容：HCT116 与 HEK293T 两条 genome-wide Perturb-seq 线；官方处理数据总计约 559.7 GB（十进制）。
- 价值：这是现有 Tahoe / scPerturb 之外的大规模人类基因扰动数据，可用于跨细胞系、未见基因和剂量相关后续验证。
- 当前状态：**下载通道阻塞，尚无文件完成下载。** Figshare 的下载端点在本网络中先返回仅 10 秒有效的 S3 临时 URL；到 S3 的 TLS 建连常超过该时限，出现 TLS EOF 或 403。小型 `guide_library.csv` 可偶发完成，两个 h5ad 不能据此视为可用。
- 留存：`tools/scripts/download_xatlas_orion.py` 保存了官方文件 ID、预期字节数、aria2 断点续传清单和状态命令；数据目录在 `/home/yyf/data/singlecell_perturbation_atlas/mega_external/X_Atlas_Orion_2025/`。

## 为什么不把数据下载数量当成结果

周老师要的是在更多 setting 下检验风险排序。Tahoe 原始层已落盘，但“已下载”不等于“已进入结果”；还要完成字段识别、可复现 split、预测器输出和 error 定义。当前优先级仍是先完成独立数据集上的严格预测器审计，再把大数据接入同一合同。
