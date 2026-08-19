# E41 多维数据第一批 smoke

- 生成时间：2026-07-10T16:50:14
- Git：`900abafd5dad`
- 工作区 dirty：`True`

## 1. 这次实际做了什么

- OpenProblems / NeurIPS 2023 Kaggle DGE：用官方 prediction 和 test logFC 计算真实误差，再看 support、SMILES 相似度、预测幅度这些前置风险线索。
- Tahoe raw：读取已完成 shard 的字段，统计 drug、cell line、MoA、SMILES、PubChem、plate 覆盖。
- 同时生成下一批实验队列，避免后续只停留在“想做”。

## 2. OpenProblems 结果快照

- train/test/prediction shape：[614, 18211] / [255, 18211] / [255, 18211]
- test tasks：255；test cell types：B cells, Myeloid cells

Spearman 最高的几项：

                       dataset_name                       risk_score_name target_error  n_tasks  spearman  top20_k  all_mean_error  top20_mean_error  top20_enrichment
OpenProblems_NeurIPS2023_Kaggle_DGE risk_oracle_true_magnitude_diagnostic     rmse_all      255  0.892658       51        0.682245          1.116114          1.635943
OpenProblems_NeurIPS2023_Kaggle_DGE risk_oracle_true_magnitude_diagnostic      mae_all      255  0.883491       51        0.526200          0.878118          1.668790
OpenProblems_NeurIPS2023_Kaggle_DGE              risk_predicted_magnitude     rmse_all      255  0.837452       51        0.682245          1.043933          1.530145
OpenProblems_NeurIPS2023_Kaggle_DGE                risk_safeconf_op_smoke     rmse_all      255  0.837088       51        0.682245          1.043933          1.530145
OpenProblems_NeurIPS2023_Kaggle_DGE              risk_predicted_magnitude      mae_all      255  0.827897       51        0.526200          0.825387          1.568579
OpenProblems_NeurIPS2023_Kaggle_DGE                risk_safeconf_op_smoke      mae_all      255  0.827541       51        0.526200          0.825387          1.568579
OpenProblems_NeurIPS2023_Kaggle_DGE                 risk_low_drug_support      mae_all      255  0.153163       51        0.526200          0.619186          1.176712
OpenProblems_NeurIPS2023_Kaggle_DGE                 risk_low_drug_support     rmse_all      255  0.152175       51        0.682245          0.790780          1.159086

## 3. Tahoe raw 字段审计快照

- 当前可见完整 shard：343；本次审计 shard：24；审计行数：677400
- gene metadata rows：62710；obs metadata rows：100648790

各字段聚合 top 值见 `tables/TAHOE_RAW_FIELD_COUNTS.csv`。

## 4. 第一批实验队列

 priority experiment_id                     experiment_name current_status
        1          E41A       OpenProblems Kaggle DGE smoke       finished
        2          E41B      Tahoe raw shard metadata audit       finished
        3           E42          sciplex3 cell-line holdout         queued
        4           E43                   TCDD dose holdout         queued
        5           E44    KaggleCrossPatient donor holdout         queued
        6           E45        crossSpecies species holdout         queued
        7           E46              Norman single-to-combo         queued
        8           E47 Gasperini regulatory target holdout         queued
        9           E48    Papalexi RNA-protein consistency         queued