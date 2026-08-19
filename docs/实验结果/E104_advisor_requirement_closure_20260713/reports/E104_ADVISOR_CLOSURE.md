# E104｜周老师要求逐项闭环

## 1. 老师追问的“输入到底是什么”

对一个没见过的 `(context, perturbation)`，当前多背景矩阵实验只给预测器和风险层以下信息：

1. 训练子矩阵里已经观测到的 perturbation effect；
2. 目标 context 的未扰动 control 表达；
3. perturbation 的 scGPT 预训练 embedding；
4. 两个预测器对目标任务产生的预测向量。

模型分歧由两个预测向量之间的 RMSE 计算。预测幅度由预测向量自身的均方根计算。训练支持数只统计训练子矩阵。背景新颖度只比较 control 表达。目标任务受扰动后的真实表达不参与预测、分数、权重或阈值，只在全部输出冻结后计算误差。

这一输入合同已在 E98、E100、E103 的 `RUN_STATUS.json` 和 strict PredictionRecord 中落盘。需要明确的部署假设是：目标背景的未扰动 control 样本可获得；没有 control 时，当前 context-aware 分数不能直接使用。

## 2. 老师提出的实验 setting

| 老师提出的内容 | 当前实现 | 数据与规模 | 结论 |
|---|---|---|---|
| 随机缺失 pair | 已完成 | Frangieh、Lara、Santinha、Cui；19 个 context-holdout folds | 四套数据均可计算，排序多数为正 |
| 只给训练矩阵的一部分 | 已完成 | 25%、50%、75%、100% 四档嵌套训练子矩阵 | 训练量改变后重新拟合预测器，没有复用全量预测冒充 |
| 整行 holdout | 已完成 | gene 13 folds；cytokine 6 folds | gene 三数据集整行平均排序为正；Cui 同样为正 |
| 整列 holdout | 已完成 | 每折约 20% perturbations 整列未见 | gene 与 cytokine 均完成；magnitude 在部分数据更强 |
| context 与 perturbation 双未见 | 已完成 | 每折 held-out row × held-out columns | 可计算；validation q80 在该 setting 上仍可能失效 |
| 不同扰动类型 | 已完成三条线 | gene knockout：E97–E101；cytokine：E102–E103；chemical：E84/E87/E89 | 三类都能运行，增量强度不同 |
| 跨数据集预测/评估 | 已运行 | gene 风险迁移 E69；chemical E87/E89；gene 元分析 E101 | 可运行，但不能写成普遍超过 magnitude |
| 方法是否能在这些 setting 上算 | 已完成可执行适配 | SourceEffect-scGPTKNN + scGPTEmbedding-ContextRidge | strict 记录通过；尚不等同于 context-aware GEARS + 端到端 scGPT |

## 3. 遗传扰动主结果

100% 训练子矩阵、四类 test setting pooled 后：

| 数据集 | contexts × perturbations | frozen pair risk ρ | magnitude ρ | disagreement ρ | Δρ vs magnitude |
|---|---:|---:|---:|---:|---:|
| Frangieh | 3×189 | 0.688 | 0.643 | 0.596 | 0.045 |
| Lara ex vivo | 5×31 | 0.229 | 0.043 | 0.085 | 0.186 |
| Santinha | 5×23 | 0.357 | 0.385 | 0.342 | -0.028 |
| 等权宏平均 | 3 datasets | 0.425 | 0.357 | 0.341 | 0.067 |

E101 的 dataset-population + fold + perturbation cluster bootstrap 显示：

- frozen pair risk 相对 disagreement：Δρ=0.083，95% CI `[0.017, 0.151]`；
- frozen pair risk 相对 magnitude：Δρ=0.067，95% CI `[-0.057, 0.213]`。

当前证据稳定支持“比单纯模型分歧多提供风险信息”。相对预测幅度仍是正趋势，尚未形成跨数据集稳定增量。删除 Lara 后，frozen−magnitude 只剩 0.008。

## 4. 不同扰动类型

Cui 细胞因子线只保留 41/86 个可通过机械规范化直接命中 scGPT 词表的标签，没有猜商品名、复合亚基或别名。六背景 pooled：calibrated pair risk ρ=0.413，magnitude ρ=0.391；相对 magnitude 的 cluster CI 为 `[-0.104, 0.193]`。

Chemical 线已有 sciPlex3 的低覆盖、整行、整列、双未见和跨数据集实验。SafeConf 能标记高误差任务，但 magnitude 在 Tahoe 和多项 chemical setting 中更强；E87 还观察到跨域时两个预测器共同失准。这些结果保留为方法边界。

## 5. 目前能向老师汇报的句子

“上次您问没见过的组合到底输入什么，我这次把合同重新做了。测试任务只使用训练子矩阵、目标背景的未扰动 control、扰动 embedding 和模型预测；真实扰动表达最后才解封算误差。现在随机 pair、小训练矩阵、整行、整列和双未见都在三套遗传数据上跑完了，还单独加了一套细胞因子刺激。遗传数据上冻结风险分数相对单纯模型分歧的保守区间已经稳定为正；相对预测幅度还是两套正、一套略负，所以这部分我保留为边界。下一步主要补 context-aware 正式模型和匹配难度的校准，不再回头调这三套测试集。”

## 6. 投稿状态

| 项目 | 状态 | 依据 |
|---|---|---|
| 周老师提出的 setting 覆盖 | 完成 | E97–E103；chemical E84/E87/E89 |
| 输入与真值隔离 | 完成 | strict records issue_count=0；RUN_STATUS truth flags |
| 独立 gene 数据复制 | 完成 | Frangieh、Lara、Santinha |
| 超过模型分歧 | 稳定 | dataset-population cluster CI 不跨 0 |
| 超过 predicted magnitude | 未稳定 | population CI 跨 0；Santinha 为负 |
| validation 阈值风险控制 | 未完成 | 双未见接受集误差未下降 |
| context-aware 正式 GEARS/端到端 scGPT | 未完成 | 当前为 embedding/transfer predictors |
| 跨数据集普遍有效 | 未完成 | gene/chemical 均有方向性失败 |

现阶段已经达到可向老师完整汇报和形成方法论文实验骨架的程度。期刊分区录用无法由实验数量保证；在冻结主表前仍需解决正式模型适配、匹配 setting 的校准和高风险任务机制解释。

## 7. 文件入口

- Frangieh 合同与结果：`../../E97_frangieh_gene_cartesian_contract_20260713/`、`../../E98_frangieh_gene_cartesian_predictions_20260713/`
- 外部 gene 合同与结果：`../../E99_multicontext_external_contract_20260713/`、`../../E100_gene_external_cartesian_predictions_20260713/`
- 三 gene 数据元分析：`../../E101_gene_cartesian_meta_audit_20260713/`
- Cui 直接映射合同与结果：`../../E102_cui_direct_mapping_contract_20260713/`、`../../E103_cui_cartesian_predictions_20260713/`
- 总关卡：`../../GATE_STATUS_20260712.md`
