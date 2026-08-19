# E199 冻结协议：TxPert 公共知识图的 K562 未见扰动复现

冻结日期：2026-08-02

实验类别：`RETROSPECTIVE_PUBLIC_MODEL_REPRODUCTION_AND_RISK_AUDIT`

## 1. 回答的问题

E199 只回答两个问题：

1. 官方公开的 TxPert 检查点能否在 K562 的未见基因扰动划分上完整产生预测，并在
   同一批任务上超过简单基线；
2. 三个公开架构的预测分歧能否在不读取目标扰动表达的情况下，识别这个注册模型
   家族中误差较大的任务。

E199 不回答跨细胞背景和双扰动。前者留给 E200，后者使用 Wessels/Norman 的独立
实验。E199 也不冒充新的盲测：官方缓存包含真实测试表达，正式运行属于公开数据上的
回顾性复现。

## 2. 固定来源

| 资产 | 固定版本或校验值 |
|---|---|
| TxPert 官方代码 | `valence-labs/TxPert@08d82eea86746b044cf7531f4ec8c5f60e1cb73f` |
| TxPert Zenodo | record `15420279` |
| `checkpoints.zip` | 291,708,781 bytes；MD5 `4058d19e14882a3bd2545b1512f1acde` |
| `K562_single_cell_line.zip` | 678,058,077 bytes；MD5 `6be4c8239fd7e6b70d2ffe7b80c3c7bc` |
| scPertEval | `8709eb07a0e7d4ecf1c60c977f2018690a749975` |
| scPerturBench | `6e24e7a9827e55d4567d2139427be9af0d1e7a6c`，只作外部评价设计核对，不参与数值计算 |

原始缓存、检查点和逐细胞预测只放在 `/home/yyf/data`，不进入 Git。Git 只保存来源、
哈希、冻结合同、聚合表和审计结果。

## 3. 固定预测对象

使用官方 `K562_single_cell_line` 缓存和其中的 `train_test_split.pkl`，不重划测试集。
注册模型家族固定为三个等权成员：

| 成员 | 官方配置 | 公共关系图 |
|---|---|---|
| TxPert-GAT | `config-gat.yaml` | STRING |
| TxPert-Exphormer | `config-exphormer.yaml` | STRING |
| TxPert-Exphormer-MG | `config-exphormer-mg.yaml` | STRING + GO |

三者都使用官方单一检查点。它们构成“公共架构家族”，不是多随机种子家族。论文中
最强模型还使用未公开的 PxMap/TxMap，本实验没有这些图，不能把公开复现结果写成
论文最优模型的完整复现。

固定基线：

- `batch-matched control`：零扰动效应；
- TxPert 官方 `general mean baseline`；
- 三成员等权 centroid；
- 单个公开模型，全部逐一报告，不只保留表现最好者。

## 4. 预测期可用的风险输入

以下量在目标扰动表达参与评价前计算并单独落盘：

- `family_diversity`：三个模型预测在欧氏空间的平均平方离差；
- `predicted_magnitude`：三模型均值预测相对 batch-matched control 的 RMSE；
- `model_baseline_gap`：家族均值与 general baseline 的 RMSE；
- `string_train_neighbor_count`：目标基因在冻结 STRING 图中与训练扰动直接相连的数量；
- `go_train_neighbor_count`：目标基因在冻结 GO 图中与训练扰动直接相连的数量；
- `graph_isolated`：在两个公共图中都没有训练邻居；
- `historical_support`：未见扰动固定为 0，用作合同检查，不作为可学习变量。

K562 只有一个 context，因此 `context_similarity` 记为
`NOT_APPLICABLE_SINGLE_CONTEXT`，不能填入常数后假装获得跨背景证据。

## 5. 固定评价端点

E198 在独立 arch1 数据上先校准协议。E199 的五个主要端点按 E198 事前选择结果固定：

| 维度 | 主要端点 | 方向 |
|---|---|---|
| 绝对表达 | `mse` | 越低越好 |
| 响应方向 | `pearson_pert` | 越高越好 |
| 扰动识别 | `rank` | 越低越好 |
| 细胞群体 | `energy_distance_pca_k=50` | 越低越好 |
| 差异表达 | `de_auprc` | 越高越好 |

为核对 TxPert 论文自身的主要指标，另报告 batch-matched control 中心化的
`pearson_ctrl`（对应 Pearson-Δ）及官方 `fast_retrieval`。这些是复现端点，不替换
E198 已选的主要端点。

主要分析只纳入真实细胞数不少于 30 的扰动；10–29 个细胞的任务单列敏感性结果；
少于 10 个细胞不评价。不会为了增加显著性临时降低门槛。

## 6. 误差对象与证书

每一行结果固定为：

```text
target_gene × predictor × endpoint
```

欧氏 MSE 空间同时检查三成员家族恒等式：

```text
family_RMS_error² = centroid_error² + family_diversity
```

因此 `sqrt(family_diversity)` 是这个注册家族 RMS 误差的确定性下界。该结论只属于
欧氏端点；Pearson、rank、Energy 和 DE-AUPRC 只做经验风险关联，不能套用同一个
下界解释。

## 7. 固定统计与裁决

- Spearman 相关以扰动基因为独立单位；
- 置信区间使用 5,000 次基因簇 bootstrap，种子 `20260802`；
- 复核预算固定为 20%，高错误集合也固定为误差最高 20%；
- `family_diversity`、`predicted_magnitude`、`model_baseline_gap` 越大表示风险越高；
- 图邻居数取负值后作为风险量，邻居越少表示风险越高。

三层裁决分开给出：

1. **证书完整性**：欧氏恒等式最大残差不超过 `1e-8`，family RMS 和
   worst-member 下界均为 0 违例；
2. **经验路由可用**：diversity–family error 的相关区间下限大于 0，且 20% 复核
   utility 的区间下限大于 0；
3. **相对 magnitude 的新增价值**：配对 `Δρ` 或配对 utility 区间下限大于 0。

第二层失败时仍保留确定性证书，但经验路由返回 `ABSTAIN`。第三层失败时不能声称
分歧优于 predicted magnitude。

## 8. 防泄漏与失败规则

官方 `predict_step` 会把 `batch.x` 一并保存为 ground truth，但模型前向只接收
control、perturbation index 和 perturbation embedding。正式审计必须同时通过：

1. 静态调用图确认目标 `batch.x` 没有进入 `forward/sample_inference`；
2. 固定小批次中将 `batch.x` 置零，预测逐元素不变；
3. 三模型预测完成并记录哈希后，才运行误差、相关和复核收益计算。

若运行代码、缓存、检查点、官方 split 或配置哈希不符，formal 直接拒绝。若不同模型
的基因集合、任务集合或样本顺序不能严格对齐，不补行、不均值填充，也不做 family
比较。群体和 DE 端点只接受模型实际输出的逐细胞预测，不复制 centroid 伪造细胞。

## 9. 许可边界

TxPert 代码和权重受 Recursion Non-Commercial End User License Agreement 约束。本次
使用限定为河南大学的非商业学术研究。仓库不重新分发代码、权重、缓存或逐细胞模型
输出；公开材料保留模型来源、是否修改和许可说明。若未来项目转为商业用途，必须
重新取得许可。

## 10. 冻结后不得修改的项目

数据版本、官方 split、模型成员、成员权重、端点、细胞数门槛、风险方向、bootstrap
次数、复核预算和裁决阈值全部冻结。结果不理想时只允许如实解释或登记新的 E 编号，
不得覆盖 E199。
