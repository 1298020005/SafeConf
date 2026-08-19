# E20 adapter contract validator

先看结论：

- 这不是新模型结果，而是一次“能不能安全接真实预测器”的源码级体检。
- E17 sciplex3 full-743 gene5000 包继续可用；数组 key coverage 正常。
- 旧 GEARS 输出能 non-strict 审计，但 strict 合同失败，主要因为旧记录缺少 gene order / normalization 等字段。
- GEARS 导出器已补写严格字段，后续重跑会更干净。

入口：

- HTML 报告：`reports/E20_ADAPTER_CONTRACT_VALIDATOR.html`
- Markdown 报告：`reports/E20_ADAPTER_CONTRACT_VALIDATOR_REPORT.md`
- 合同流程图：`figures/adapter_contract_validator.svg`
- 汇总表：`tables/ADAPTER_CONTRACT_BUNDLE_SUMMARY.csv`
- 问题明细：`tables/ADAPTER_CONTRACT_ISSUES.csv`

核心表：

| bundle_id | bundle_group | n_records | strict_status | non_strict_status | primary_strict_issue |
| --- | --- | --- | --- | --- | --- |
| E17_SCIPLEX3_FULL743_GENE5000 | sciplex3_full743 | 22290 | fail | warn | inconsistent_true_effect_key_for_task |
| GEARS_FORMAL_ADAMSON_SEED_1 | gears_formal_legacy | 7 | fail | pass | missing_contract_columns |
| GEARS_FORMAL_ADAMSON_SEED_2 | gears_formal_legacy | 7 | fail | pass | missing_contract_columns |
| GEARS_FORMAL_ADAMSON_SEED_3 | gears_formal_legacy | 7 | fail | pass | missing_contract_columns |
| GEARS_FORMAL_DIXIT_SEED_1 | gears_formal_legacy | 1 | fail | pass | missing_contract_columns |
| GEARS_FORMAL_DIXIT_SEED_2 | gears_formal_legacy | 1 | fail | pass | missing_contract_columns |
| GEARS_FORMAL_DIXIT_SEED_3 | gears_formal_legacy | 1 | fail | pass | missing_contract_columns |
| GEARS_FORMAL_NORMAN_SEED_1 | gears_formal_legacy | 10 | fail | pass | missing_contract_columns |
| GEARS_FORMAL_NORMAN_SEED_2 | gears_formal_legacy | 10 | fail | pass | missing_contract_columns |
| GEARS_FORMAL_NORMAN_SEED_3 | gears_formal_legacy | 10 | fail | pass | missing_contract_columns |

适配器合同：

| requirement | plain_meaning | future_action |
| --- | --- | --- |
| PREDICTION_RECORDS.csv | 每一行是一条模型预测，至少要能说明数据集、任务、预测器、真实误差。 | 保留统一列名，不允许每个模型随意改字段。 |
| predicted_effect_key / true_effect_key | CSV 里的 key 必须能在 npz 里找到对应向量。 | 同一任务不同预测器应共享同一个 true_effect_key。 |
| gene_panel_id + gene_order_hash | 说明用了哪些基因，以及这些基因的顺序。 | GEARS 导出器已补写；scGPT/CPA adapter 必须同样写出。 |
| normalization_id + effect_definition | 说明 effect 是均值差、logFC 还是其它定义，归一化怎么做。 | 统一使用 mean_diff 或明确转换。 |
| strict contract pass | 能直接进入跨模型、同任务、同基因顺序的比较。 | 先修 adapter，再重跑小规模三模型 smoke，最后扩展正式验证。 |
