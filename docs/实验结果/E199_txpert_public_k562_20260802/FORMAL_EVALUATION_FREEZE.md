# E199 formal 评价冻结

冻结时间：2026-08-02

## 评价对象

- 测试任务固定为 TxPert 官方 K562 `unseen perturbation` split 中的 272 个
  单基因扰动；
- 预测器固定为 GAT、Exphormer、Exphormer-MG、三成员等权均值、官方
  general baseline 和 batch-matched control；
- 所有预测和风险特征的哈希已在查看误差前提交。Formal 必须重算特征并
  要求最大绝对残差不超过 `1e-12`。

## scPertEval 主端点

使用 `scPertEval@8709eb07a0e7d4ecf1c60c977f2018690a749975`的参考实现：

| 维度 | 端点 | 方向 |
|---|---|---|
| 绝对表达 | `mse` | 低好 |
| 响应方向 | `pearson_pert` | 高好 |
| 扰动识别 | `rank` | 低好 |
| 细胞群体 | `energy_distance_pca_k=50` | 低好 |
| 差异表达 | `de_auprc` | 高好 |

群体评价使用模型实际输出的逐细胞预测。家族预测在同一细胞位置对三成员
取等权均值，不复制 centroid 伪造细胞。PCA 使用测试扰动细胞和原始数据中
10,691 个真实 control 细胞拟合。

主分析是每个扰动不少于 30 个真实细胞的 263 个任务。10–29 个细胞的 9 个任务
使用完整 272 任务的参考空间单列敏感性结果。

## TxPert 复现端点

直接调用哈希固定的 `gspp/metrics.py`，报告：

- 使用每个实验批次配对 control 的 `pearson_delta`；
- 官方固定随机子集实现的 `fast_retrieval`。

两者是对 TxPert 论文评价语义的核对，不替代五个 scPertEval 主端点。

### 运行前修正 01：零响应基线

正式 attempt 2 在尚未写出任何结果时发现：对 `batch-matched control` 而言，
`prediction-control` 恒为零。TxPert 的 `pearson_delta` 明确把常量输入产生的 NaN
记为 0；`RetrievalMetric` 没有相同保护，会把一组 NaN 相关系数排序成没有数学意义的
有限名次。因此本次对该基线的 Pearson 依官方约定记 0，retrieval 明确记为 NA；其余
预测器仍调用哈希固定的官方实现。这个修正不改变五个主端点、证书或三层 SafeConf
gate。为减少官方 retrieval 内部重复计算同一均值，先计算一次预测差值均值，再作为
单行矩阵传入；相关向量及排序定义不变。

## 证书、路由和新增价值

1. 证书完整性：`family_RMS² = centroid_RMSE² + diversity_lower_bound²`
   最大残差不超过 `1e-8`，且 family RMS 和 worst member 的下界违例数均为 0；
2. 经验路由：在 263 个主任务上，diversity–family RMS error 的 Spearman
   95% CI 下限和 20% 复核预算 utility 的 95% CI 下限都必须大于 0；
3. 新增价值：diversity 减 predicted magnitude 的配对 `Δρ` 或配对 utility
   至少有一项 95% CI 下限大于 0。

使用 5,000 次扰动基因 bootstrap，种子由 `20260802 + 统计量名称` 确定。每个基因
在该 split 中只有一个任务，因此基因簇 bootstrap 与任务 bootstrap 一致。

## 失败模式和解释边界

事前加入一个次要诊断：比较家族预测与真实目标基因效应的方向，用于检查 TxPert
论文公开讨论的“未见 CRISPRi 目标自身下调不足”。该诊断不参与三层 gate。

E199 只回答 K562 内的整列留出和固定公开模型家族。它不能回答整个 context 留出、
跨数据集迁移、双扰动或新模型家族的通用性。即使全部 gate 通过，也不把公开
STRING+GO 复现写成包含未公开 PxMap/TxMap 的论文最强模型。
