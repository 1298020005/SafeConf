# E154｜Replogle 随机种子与训练量稳健性结果

## 完整性

6 个分析运行全部进入汇总：E151 全量种子复用 1 个，新训练 5 个。每个运行均为 2 folds、340 个诊断测试任务、680 条 strict PredictionRecord、0 issue；主分析固定为 256 个 held-out-context 唯一任务，E134 保存真值与源数据重建真值检查全部通过。

## 主要结果

全量训练时，calibrated SafeConf 的绝对 RMSE 排序在 3 个种子中均为正，范围 0.173–0.375；冻结方向分数的复合方向误差排序也均为正，范围 0.191–0.389。三种子均值中，绝对 SafeConf (0.292) 与 magnitude (0.294) 基本相同；冻结方向分数 (0.318) 略高于方向 magnitude (0.301)，但只有 3 个种子，不能写成稳定优越。模型分歧的范围包含负值，随机种子敏感性最明显。

E149 的 `25/50/75/100` 是预先冻结的哈希阈值标签，不是强制等量抽样。对应实际训练对为 18/58/105/140，即 12.9%/41.4%/75.0%/100%。因此下表保留冻结标签以便复跑，同时并列写出实际样本量；后文不把最小子集称作严格的 25% 训练。固定种子的最小子集使 scGPT RMSE 升至 1.122、ensemble RMSE 升至 0.570；第二档子集后模型 RMSE 恢复到约 0.10。冻结方向复合排序由 0.002、0.157、0.210 增至 0.389。calibrated SafeConf 在标签 25 的第二折、标签 50 的第一折以及标签 75 的两折成为常数，因此这三个子集都没有完整的两折绝对 Spearman，统一标为 NA，绝不以单折补位。这说明校准分支在缩减训练集时存在明显退化；冻结方向分数和 magnitude 仍可计算。逐折唯一值与标准差见 `tables/E154_SCORE_DEGENERACY_AUDIT.csv`。

## 全量训练的三种子范围

| 指标 | 三种子均值 | 最小 | 最大 | 范围 |
|---|---:|---:|---:|---:|
| absolute_safeconf_spearman | 0.2924 | 0.1726 | 0.3748 | 0.2022 |
| absolute_magnitude_spearman | 0.2938 | 0.2143 | 0.3462 | 0.1318 |
| absolute_disagreement_spearman | 0.2187 | -0.0364 | 0.3625 | 0.3989 |
| directional_pearson_spearman | 0.3185 | 0.1937 | 0.3888 | 0.1951 |
| directional_cosine_spearman | 0.3165 | 0.1871 | 0.3889 | 0.2018 |
| directional_composite_spearman | 0.3178 | 0.1905 | 0.3894 | 0.1989 |
| magnitude_composite_spearman | 0.3009 | 0.2081 | 0.3488 | 0.1407 |
| rmse_scgpt_mean | 0.1074 | 0.1000 | 0.1184 | 0.0185 |
| rmse_gears_mean | 0.1088 | 0.1065 | 0.1131 | 0.0066 |
| rmse_ensemble_mean | 0.1006 | 0.0983 | 0.1031 | 0.0049 |
| rmse_training_perturbed_mean | 0.5808 | 0.5808 | 0.5808 | 0.0000 |
| ensemble_minus_simple_baseline | -0.4802 | -0.4825 | -0.4776 | 0.0049 |

## 固定种子 2026071542 的训练量趋势

| 指标 | label 25（12.9%, n=18） | label 50（41.4%, n=58） | label 75（75.0%, n=105） | label 100（100%, n=140） | 全量−最小子集 | 子集规模 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| absolute_safeconf_spearman | NA | NA | NA | 0.3748 | NA | NA |
| absolute_magnitude_spearman | -0.0135 | 0.0717 | 0.0444 | 0.3462 | +0.3596 | +0.800 |
| absolute_disagreement_spearman | 0.0127 | 0.1087 | 0.0907 | 0.3625 | +0.3498 | +0.800 |
| directional_pearson_spearman | 0.0025 | 0.1520 | 0.2124 | 0.3888 | +0.3863 | +1.000 |
| directional_cosine_spearman | 0.0082 | 0.1610 | 0.2081 | 0.3889 | +0.3807 | +1.000 |
| directional_composite_spearman | 0.0018 | 0.1566 | 0.2102 | 0.3894 | +0.3875 | +1.000 |
| magnitude_composite_spearman | -0.0092 | 0.0732 | 0.0442 | 0.3488 | +0.3580 | +0.800 |
| rmse_scgpt_mean | 1.1218 | 0.1192 | 0.0966 | 0.1038 | -1.0181 | -0.800 |
| rmse_gears_mean | 0.1068 | 0.1089 | 0.1074 | 0.1131 | +0.0064 | +0.800 |
| rmse_ensemble_mean | 0.5705 | 0.1055 | 0.0976 | 0.1003 | -0.4702 | -0.800 |
| rmse_training_perturbed_mean | 0.5825 | 0.5818 | 0.5816 | 0.5808 | -0.0017 | -1.000 |
| ensemble_minus_simple_baseline | -0.0120 | -0.4763 | -0.4840 | -0.4805 | -0.4685 | -0.800 |

## 解释边界

E154 在 E152 解封之后设计，用于回答结果是否依赖单个训练随机种子，以及 E149 预先冻结的训练成员减少时结果怎样变化。它不构成独立确认，不重新判定 E152 gate，也没有在 Replogle上重拟合 E135 方向模型。四个训练比例点只作描述；小训练集同时改变预测模型、训练支持度和Systema 训练质心，不能把曲线解释成单一因素的因果效应。

75% 训练的第二折有 49 个冻结训练对，E112 默认 batch=16 会留下一个单样本末批，GEARS BatchNorm 无法训练。失败运行未进入结果；重跑时只把该折训练 batch 调为 15，保留全部 49 对。完整的事后技术偏差披露见 `TECHNICAL_DEVIATION.md`。

结果范围仍限于同一 Replogle 研究内、目标 control 可见的 K562/RPE1 跨细胞系任务。它不等于跨研究泛化、完全 zero-shot 或湿实验验证。逐运行任务表、fold 指标、模型/质心基线及源真值审计均保存在 `runs/`；汇总数值在 `tables/`。
