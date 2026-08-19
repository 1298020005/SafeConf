# E166 分析合同｜跨研究折内秩组合

冻结日期：2026-07-16。本合同针对 E153 已公开的八研究任务表做方法开发；不读取单细胞表达矩阵，不重新训练 GEARS/scGPT，也不把本实验称为新的独立确认。

## 1. 要回答的问题

E153 显示 SafeConf 相对模型分歧有稳定平均增益，但相对 predicted magnitude 的八研究合并区间跨 0。另一个已确认的问题是不同 outer fold 的原始校准分数不在同一量尺，直接跨 fold 排序会产生尺度混排。

E166 检验：先在每个 fold 内把三个可部署风险量转为百分位，再仅用其余七个研究学习一个非负、和为 1 的组合，能否在完全留出的第八个研究中比 magnitude 更好地排序预测误差。

## 2. 冻结输入

- 输入：`docs/实验结果/E153_eight_study_formal_meta_20260714/tables/E153_ABSOLUTE_TASK_INPUT.csv`
- SHA-256：`b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a`
- 固定规模：3,465 行、8 个研究、34 个 outer folds。
- 端点：`error_two_predictor_mean_rmse`，数值越高表示预测越差。
- 三个输入分数：`baseline_predicted_magnitude`、`risk_model_disagreement`、`safeconf_calibrated_pair_risk`。三者均在原实验中先于测试真值生成。

正式运行必须验证输入哈希、行数、研究数、fold 数、必需字段、有限数值和 Git 中的合同/脚本一致性。正式结果只写入 append-only `release/`。

## 3. 折内转换

每个研究的每个 outer fold 独立处理。三个风险分数分别转换为 `[0,1]` 百分位秩；并列值使用平均秩。训练端点也只在训练研究内按 fold 转为误差百分位，用于拟合组合权重。测试研究的误差在权重固定前不得进入拟合、选参或分数转换。

折内百分位解决 fold 间量尺不一致，代价是该方法面向一批待质检任务的相对排序，不提供单个任务的绝对失败概率。

## 4. 留一研究组合

依次留出一个完整研究：

1. 其余七研究为训练集，留出研究为测试集。
2. 训练目标为折内误差百分位。
3. 三个特征为折内 magnitude、disagreement、SafeConf 百分位。
4. 学习 `w_mag, w_dis, w_safe >= 0` 且三者之和为 1，使加权平方误差最小。
5. 训练目标中每个研究总权重相同；同一研究内每个 fold 总权重相同；同一 fold 内每行权重相同。
6. 不调超参数，不查看留出研究真值来选择权重，不按研究结果切换模型。

测试分数为三个测试折内百分位的冻结加权和。比较器为 magnitude、disagreement 和原 SafeConf 各自的折内百分位。

## 5. 评价与不确定性

- 每个研究的主值：各 outer fold 内 `score` 与原始 RMSE 的 Spearman 相关，再对 folds 等权平均。
- 主比较：`LODO rank stack − magnitude`。
- 次要比较：相对原 SafeConf、相对 disagreement。
- 每研究按 perturbation 为簇做 3,000 次配对 bootstrap；同一扰动的所有 context/fold 行同步抽取，已冻结的组合权重不在测试 bootstrap 中重拟合。
- 总体值为八研究等权平均。总体区间用两层 bootstrap：先对研究有放回抽样，再从被抽中的研究对应 cluster-bootstrap draws 中抽取一次，固定 10,000 次。
- 同时报告八研究中差值为正的研究数、所有 LODO 权重和每折 top-20% 高误差富集倍数。

预设严格通过条件同时满足：

1. 八研究等权平均 `Δrho(stack−magnitude) > 0`；
2. 两层 bootstrap 95% CI 下界大于 0；
3. 至少 6/8 个留出研究的点估计差值大于 0。

未通过时保留完整负结果，不改端点、不删研究、不另选权重规则。

## 6. 解释边界

E166 是在已公开八研究上开展的 post-hoc 方法开发。留一研究保证每次拟合不使用该测试研究真值，但研究集合、候选特征和问题本身已受先前结果启发。因此通过只能支持“跨研究交叉验证中的可迁移增量”，仍需一个设计阶段未参与的全新数据集做最终确认；失败则说明现有三个分数不足以稳定超过 magnitude。
