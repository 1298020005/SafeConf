# SafeConf 论文主张锁定（2026-07-12）

这份文件约束后续实验、图表和论文用语。已有结果如果与这里冲突，以原始记录和本文件的边界为准，不通过修改措辞掩盖负结果。

## 1. 论文对象

SafeConf 是接在单细胞扰动预测器之后的 post-hoc 风险分诊层。预测器先输出目标任务的扰动效应向量，SafeConf 再利用部署时可取得的证据给任务排序，帮助决定优先复核、拒绝或安排湿实验的对象。

方法分为两个层次：

1. **证书层**：两个异构预测器的 RMSE 分歧只由预测向量计算。分歧的一半是二者平均误差和最大误差的下界；高分歧能证明 pair-level 风险，不能定位具体错误模型。
2. **路由层**：在来源域或验证折中，用任务新颖性、历史支持、模型分歧和 predicted magnitude 校准筛选预算。固定四项等权公式保留为透明基线，不再作为普遍有效的主方法。

这两个层次属于同一条风险分诊链。证书提供无需目标真值的确定性信息，路由层处理低分歧时仍可能共同犯错、不同任务域量纲不同等问题。

## 2. 当前允许的主张

> 在 Adamson、Norman 和 Frangieh 三个遗传扰动数据集的两套不重叠未见基因面板上，同模型 seed 分歧不能稳定识别高误差任务；GEARS 与正式微调 scGPT 的模型家族分歧为 pair mean/max RMSE 提供无目标真值下界，并能在 144 个冻结任务上排序 pair risk。该信号相对 predicted-magnitude 聚合具有正的增量，但不能判断具体哪个模型出错，也不能把低分歧解释为安全。

对应证据：

- 理论与首批 72 任务：`E74_pair_risk_certificate_20260711/`
- 两套不重叠面板、144 任务复现：`E77_repeated_panel_pair_risk_20260711/`
- 负对照：E60/E66/E71 seed disagreement，E64 global-baseline disagreement
- 跨域边界：E69/E73

## 3. 投稿前需要补成的主张

> 在低覆盖子矩阵、整行、整列、双未见和跨数据集场景中，所有风险分数均只使用训练数据、可用 control、固定预测器输出或外部扰动表征；在统一记录合同下，与 predicted magnitude、模型原生 uncertainty 和直接不确定性方法比较，SafeConf 能改善至少一种可复核的筛选指标，并清楚报告失效边界。

这段目前不能写进摘要。它是接下来实验要验证的假设。

## 4. 禁止使用的表述

- SafeConf 对所有预测模型、所有数据集都有效。
- 模型分歧是单模型置信度、误差上界或校准概率。
- 低分歧任务是安全任务。
- 固定四项等权分数普遍优于 predicted magnitude。
- true effect magnitude 能用于前置打分。
- 三个数据集或大量下载已经保证二区/一区录用。
- 将轻量 reference predictor、GEARS/scGPT、CPA/chemCPA 的结果混成同一种 predictor 而不标名称。

## 5. 统一结果合同

主结果的每一行至少包含：

```text
split_setting, dataset_name, perturbation_family,
predictor_name, predictor_training_scope, task_key,
predicted_effect_key, true_effect_key, gene_order_hash,
score_name, score_input_provenance, error_name, error_value,
predicted_magnitude, true_magnitude_oracle, seed, frozen_manifest_id
```

`true_magnitude_oracle` 只能进入诊断列。跨模态可以使用适配各自任务的 predictor，但记录字段、真值隔离规则、误差定义和统计流程必须统一。

## 6. 主图草案

1. SafeConf 的部署位置、输入来源和真值隔离。
2. pair-risk 下界与平方误差分解。
3. 三数据集×两不重叠面板的分层结果与 magnitude 增量。
4. 子矩阵四象限：seen/seen、new-context、new-perturbation、double-unseen。
5. 行列留出和跨数据集的 risk–coverage / top-k 结果。
6. 负结果与适用边界：seed、McFarland、跨域失败、低分歧共同犯错。
