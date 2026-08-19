# E20 adapter contract validator

生成时间：2026-07-07 21:46

## 1. 这次做了什么

E20 检查当前已有预测输出是否满足 SafeConf 统一适配器合同。这个合同主要约束三件事：CSV 任务表、预测/真实 effect 向量、基因顺序与归一化说明。

## 2. 总结

| bundle_id | bundle_group | n_records | n_tasks | n_predictors | strict_status | non_strict_status | primary_strict_issue | array_shape_unique_gene_dims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E17_SCIPLEX3_FULL743_GENE5000 | sciplex3_full743 | 22290 | 2229 | 2 | fail | warn | inconsistent_true_effect_key_for_task | 5000 |
| GEARS_FORMAL_ADAMSON_SEED_1 | gears_formal_legacy | 7 | 7 | 1 | fail | pass | missing_contract_columns | 5043 |
| GEARS_FORMAL_ADAMSON_SEED_2 | gears_formal_legacy | 7 | 7 | 1 | fail | pass | missing_contract_columns | 5043 |
| GEARS_FORMAL_ADAMSON_SEED_3 | gears_formal_legacy | 7 | 7 | 1 | fail | pass | missing_contract_columns | 5043 |
| GEARS_FORMAL_DIXIT_SEED_1 | gears_formal_legacy | 1 | 1 | 1 | fail | pass | missing_contract_columns | 6000 |
| GEARS_FORMAL_DIXIT_SEED_2 | gears_formal_legacy | 1 | 1 | 1 | fail | pass | missing_contract_columns | 6000 |
| GEARS_FORMAL_DIXIT_SEED_3 | gears_formal_legacy | 1 | 1 | 1 | fail | pass | missing_contract_columns | 6000 |
| GEARS_FORMAL_NORMAN_SEED_1 | gears_formal_legacy | 10 | 10 | 1 | fail | pass | missing_contract_columns | 5025 |
| GEARS_FORMAL_NORMAN_SEED_2 | gears_formal_legacy | 10 | 10 | 1 | fail | pass | missing_contract_columns | 5025 |
| GEARS_FORMAL_NORMAN_SEED_3 | gears_formal_legacy | 10 | 10 | 1 | fail | pass | missing_contract_columns | 5025 |

## 3. 问题类型统计

| bundle_group | mode | issue_type | n |
| --- | --- | --- | --- |
| gears_formal_legacy | strict | missing_contract_columns | 9 |
| sciplex3_full743 | non_strict | legacy_true_effect_key_record_scoped_check_skipped | 1 |
| sciplex3_full743 | strict | inconsistent_true_effect_key_for_task | 11145 |

## 4. 适配器要求

| requirement | plain_meaning | why_it_matters | current_e17_status | current_gears_legacy_status | future_action |
| --- | --- | --- | --- | --- | --- |
| PREDICTION_RECORDS.csv | 每一行是一条模型预测，至少要能说明数据集、任务、预测器、真实误差。 | 没有这张表，SafeConf 无法把置信度分数和真实失败对应起来。 | present | present | 保留统一列名，不允许每个模型随意改字段。 |
| predicted_effect_key / true_effect_key | CSV 里的 key 必须能在 npz 里找到对应向量。 | 这是从表格跳到真实基因表达向量的桥。 | present; key coverage audited | present; key coverage audited | 同一任务不同预测器应共享同一个 true_effect_key。 |
| gene_panel_id + gene_order_hash | 说明用了哪些基因，以及这些基因的顺序。 | 两个向量长度相同也可能基因顺序不同，不校验会产生假一致。 | present; strict fail because true_effect_key is record-scoped across predictors | legacy missing strict provenance columns in old outputs | GEARS 导出器已补写；scGPT/CPA adapter 必须同样写出。 |
| normalization_id + effect_definition | 说明 effect 是均值差、logFC 还是其它定义，归一化怎么做。 | 不同 effect 定义不能直接混在一起比较 RMSE。 | present | legacy missing in old outputs | 统一使用 mean_diff 或明确转换。 |
| strict contract pass | 能直接进入跨模型、同任务、同基因顺序的比较。 | 这是把 GEARS、scGPT、CPA 放在一张表里比较的门槛。 | present; strict fail because true_effect_key is record-scoped across predictors | legacy missing strict provenance columns in old outputs | 先修 adapter，再重跑小规模三模型 smoke，最后扩展正式验证。 |

## 5. 结论口径

- E17 sciplex3 full-743 gene5000 结果包可继续作为当前 strongest formal package 使用；它的数组 key coverage 正常。
- E17 在严格跨模型合同下仍有一个老问题：同一任务下不同预测器使用 record-scoped true_effect_key。后续三模型统一验证前要改成 task-scoped true_effect_key。
- 旧 GEARS formal 输出能做 non-strict 审计；严格模式失败的主因是旧 CSV 缺少 gene_panel_id、gene_order_hash、normalization_id 等来源字段。
- GEARS 导出脚本已经补写这些严格字段。以后新跑 GEARS 时，输出会更接近 strict contract。
