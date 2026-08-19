# E8 scPerturBench 公开资源只读审计

审计日期：2026-06-14

## 结论

**对完整 SafeConf 跨架构外部评分：当前 NO-GO。**

**对后续 task-only 跨方法误差研究：PARTIAL-GO。**

这两个结论不能混写。

## 查到的资源

官方主仓库在 commit
`6e24e7a9827e55d4567d2139427be9af0d1e7a6c` 下公开了按
dataset、method、perturbation、metric、seed 组织的汇总 CSV。例如：

- genetic `all_dataset_genetic.csv`：55,119,282 bytes；
- chemical `all_dataset_chemical.csv`：14,773,597 bytes；
- cellular context IID/OOD：多个逐任务指标 CSV。

这些表能提供每个方法在每个 perturbation 上的 MSE、PCC-distance 等**误差标签**，
但不是预测表达向量。

官方 reproducibility 网站在 commit
`e9800bd01039afacdfc7197dbe406731d2f998f0` 下主要包含：

- 静态 HTML 结果页；
- 聚合指标图；
- 每个 perturbation 的预测可视化 PNG。

抽查 Frangieh 与 McFarland 页面，只发现 PNG 链接，没有 CSV/NPZ/H5AD/PKL
预测向量下载链接。PNG 不能作为数值预测输出使用。

## 为什么暂时不能直接跑完整 SafeConf

完整外部评分至少需要：

1. 每个 predictor 的逐任务预测 effect vector；
2. 对应 true effect vector；
3. context、perturbation、split 的稳定标识；
4. 能从训练侧构造 support、historical、OOD 和 disagreement 特征。

公开聚合 CSV 只有第 3 项的一部分和误差指标，不提供第 1 项。因此不能计算
prediction-output 特征，也不能重新计算模型分歧，更不能验证其误差定义是否和
SafeConf RMSE 完全一致。

## 可行的后续小实验

可以另行设计一个**task-only external error association**：

- 选择与现有数据可准确对齐的数据集，例如 Frangieh 或 sciplex3；
- 从 SafeConf 训练侧只构造 pre-model task features；
- 将 scPerturBench 各方法的逐 perturbation MSE 作为外部误差标签；
- 按 method 分层评估 task-risk 与误差的相关。

这能检验“任务难度是否跨预测架构迁移”，但它不是完整 SafeConf score，也不能替代
预测向量级验证。启动前还需要先核实 perturbation、context 和 split 的一一对应。

## 当前决策

- 不下载 40 GB Podman 镜像；
- 不把网站 PNG 当预测数据；
- 不声称已有 27 方法的完整 SafeConf 验证；
- 保留 aggregate-error 对齐为下一轮低成本 feasibility audit。

## 官方来源

- https://github.com/bm2-lab/scPerturBench
- https://github.com/bm2-lab/scPerturBench-reproducibility
- https://bm2-lab.github.io/scPerturBench-reproducibility/
- https://zenodo.org/records/15904698
