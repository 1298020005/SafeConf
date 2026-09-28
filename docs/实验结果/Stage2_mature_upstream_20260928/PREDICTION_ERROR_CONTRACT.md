# SafeConf Prediction/Error Contract v1

状态：冻结为执行合同（2026-09-29）

本合同把不同上游模型的输出转换为可审计的任务级 `PredictionRecord`。它不改变上游模型，不把不同输出尺度的原始误差直接混合。

## 1. 两种合法 effect 路径

### Direct-effect model

上游直接输出扰动效应：

```text
predicted_effect = upstream_direct_effect
```

不得为了统一接口人为构造 control。

### Treated-state model

上游输出 treated state 时：

```text
predicted_effect = predicted_treated_state - matched_predicted_or_observed_control
true_effect = observed_treated_state - matched_observed_control
```

每条记录必须保存：

```text
effect_definition
control_source
conversion_rule
output_contract_id
```

不同 conversion contract 不直接混合原始 RMSE。

## 2. PredictionRecord 最低字段

```text
task_id
dataset_id
context_id
perturbation_type
perturbation_target
perturbation_condition
control_id
upstream_model_id
prediction_vector
gene_ids
effect_definition
control_source
conversion_rule
normalization_id
output_contract_id
provenance
```

## 3. Gene space 与 normalization

- gene ID 映射、顺序、公共交集和缺失策略必须在 adapter 中固定。
- 缺失 gene 不得未经说明填零；必须使用预登记的 mask/intersection 规则。
- normalization 只能在训练分区拟合，验证和测试只应用训练变换。
- 不能共享 output contract 的数据，在各 dataset/context 内评价后再做 macro aggregation，不 pooled 原始 RMSE。

## 4. Error target

主标签为：

```text
task-level RMSE on aligned effect vector
```

同时报告 MAE、Pearson/Spearman 和 directional/cosine error。absolute error 与 directional error 分开建模。

## 5. 历史与泄漏

PredictionRecord 的历史来源必须通过 `History Eligibility Contract`。禁止使用当前任务 truth、同 experimental unit replicate、target-specific 派生统计量和未在 outer-train 内计算的全局统计量。

## 6. Adapter 验收

每个 adapter 必须验证：

1. prediction/truth 向量形状一致；
2. gene order 与 gene hash 一致；
3. control_source 和 conversion_rule 可追溯；
4. normalization_id 已登记；
5. 预测向量、truth 和 error 可重算；
6. 当前任务 truth 不进入输入特征；
7. direct-effect 与 treated-state 不能静默混合。
