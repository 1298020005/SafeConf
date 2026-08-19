# E153 分析合同｜八研究正式扩展元分析

冻结日期：2026-07-14；本合同在 E153 读取并汇总端点前建立。

## 1. 分析性质

E153 将 E140 的七研究 absolute-RMSE 结果与 E151/E152 的 Replogle 正式主任务合并。E152 结果已经解封，因此 E153 是 **post-E152 expanded meta-analysis**，不是新的预注册独立 gate，也不能把八研究汇总改写成新的前瞻性确认。

Replogle 的任务选择、双模型预测和 Directional-SafeConf gate 分别由 E149、E151、E152 冻结和执行；E153 不修改风险分数、任务、模型、端点或阈值。

## 2. 冻结输入与完整性要求

Absolute-RMSE：

- E146 冻结的 E140 七研究任务快照，3,209 行，SHA-256 `95e5c776f97efbefaaf4d312dbb906ac100701e9682480020e8dd32153188ec9`。
- E151 `PRIMARY_TASK_RISK_TABLE.csv`，256 个 held-out-cell-line 主任务，SHA-256 `f1c1d9290cf5519ac702e214b2783f37cf6a9bc723f81d4405bb3891aa625118`。
- E151 顶层与数据集状态必须均为 complete、strict issue count 必须为 0；`STRICT_ISSUES.csv` 除表头外不得有记录。
- Replogle 主表必须恰有两个 folds、128 个 perturbations、每个 perturbation 在 K562/RPE1 各有一个主任务，共 256 个唯一 `(context, perturbation)`。

Directional-SafeConf：

- Nadig 使用 E146 冻结的 E139 方向输入，SHA-256 `5c2ec6b5dfa7de56fbb873403a5910faf71c0972f188feb346fa851005517268`。
- Replogle 使用 E152 `E152_TASK_AUDIT.csv`，SHA-256 `c5dd3bb6b61dab9938f88e389e068a3ab856d44f59db76a587b24f4452a0c353`。
- E152 冻结分数文件 SHA-256 必须为 `611ac06b5630b33dd3e2f16e62f5a1c5bd903d1cb895ce69ee97e56578011b42`，并与 `SCORE_FREEZE_STATUS.json`、`RUN_STATUS.json` 和任务表逐行一致。
- E135 方向模型 SHA-256 必须为 `77caf3b7b46071ced9577a8bd5289ce4c7bf5899c329ab37e835c41bda07d4b3`；不得在 Replogle 上重拟合。
- E152 必须保留256个唯一主任务、source-truth audit通过、strict issue count为0。

freeze 阶段把三个分析输入复制为 E153 内部快照并记录合同、脚本、manifest与全部源文件哈希。analyze 阶段只能读取这些快照；完成后必须能够从相同快照安全重跑。

## 3. Absolute-RMSE 主分析

- 端点：`error_two_predictor_mean_rmse`，越高表示误差越大。
- SafeConf：`safeconf_calibrated_pair_risk`。
- 比较器：`risk_model_disagreement`、`baseline_predicted_magnitude`。
- 主 estimand：每个 outer fold 内计算风险与误差的 Spearman，再对同一研究所有fold等权平均。
- 每个研究以 `perturbation` 为独立重采样簇；同一扰动的全部context和fold记录同步出现。
- 每个研究固定3,000次cluster bootstrap；三种分数必须共享同一抽样，从而配对计算 SafeConf 与比较器的差值。
- 研究效应使用 Fisher z。单分数研究内标准误来自其 bootstrap Fisher-z 分布；SafeConf减比较器的标准误来自配对 `Δ Fisher-z` 分布。
- 八研究随机效应：REML估计tau²，modified Knapp–Hartung均值区间，t分布prediction interval；同时报告Q与I²。
- LODO：每次删除一个完整研究，重算相同模型。

正方向含义：风险越高、误差越高，Spearman越正越好；`SafeConf − comparator` 大于0表示SafeConf排序更强。

## 4. Pooled-median sensitivity

每个 `(dataset, context, perturbation)` 跨fold的端点和分数分别取中位数，只保留一行；随后计算研究内pooled Spearman。该分析会改变fold-macro estimand，只能标为 `pooled_context_task_median_sensitivity`。

为避免把共享扰动的不同context当独立样本，pooled-median sensitivity同样按perturbation同步全部context进行3,000次cluster bootstrap，并独立给出八研究随机效应和LODO。不得用它覆盖主fold-macro结果。

## 5. Directional-SafeConf：两个独立研究的描述性汇总

- 研究：Nadig、Replogle；两者均未参与 E135 方向风险头的开发或重拟合。
- 端点：`error_centered_pearson_mean`、`error_centered_cosine_mean`、`direction_error_rank_target`；均为越高越差。
- 分数：`directional_risk_frozen`为主；同时报告magnitude、disagreement和原absolute SafeConf。
- 每个研究先计算fold-macro Spearman，再以perturbation为簇同步所有context/fold进行3,000次bootstrap。
- 两研究描述性合并使用等研究权重的平均Spearman；可报告固定这两个研究条件下的配对bootstrap区间、两研究最小值与最大值。
- `k=2` 不进行REML、Knapp–Hartung、I²、prediction interval或“跨研究稳定保证”宣称。

## 6. 输出与解释边界

必须落盘：冻结状态、输入manifest、绝对任务快照、方向任务快照、fold结果、cluster bootstrap draws/summary、研究效应、八研究meta、LODO、pooled-median敏感性、方向双研究描述、白底图、报告与完成状态。

E153 可以说明加入 Replogle 后现有八研究的平均效应、异质性和敏感性；不能保证未来研究、期刊录用、湿实验因果机制或完全zero-shot泛化。
