# E167a｜RIAG v2 批次级、tie-aware 协议修正

预设回归测试：`PASS`。E167 v1 的正式 `FAIL` 保持不变；E167a 仍是历史数据上的方法开发。

## 修正后的含义

预测向量是否随任务变化、风险分数是否非退化、具体 cutoff 能否给出唯一集合，现已分开判定。所有历史状态统一写为 EVALUABLE 或 ABSTAIN，`deployment_authorized` 固定为 false。缺少重复稳定性或上游预测器无效时，不再输出部署授权措辞。

正式分析包含 `45` 个历史部署批次；其中 `43` 个通过 G2a，`38` 个通过 G3a，`26` 个在全部登记 cutoff 上集合唯一。

## Replogle 与 Santinha

| batch_id | n_tasks | score_quantized_unique | G3a_prediction_task_dependence_passed | low_risk_20pct_boundary_status | high_risk_20pct_boundary_status | evaluation_status |
|---|---|---|---|---|---|---|
| Replogle_cellline_holdout_1_K562 | 128 | 3 | True | TIEBREAK_REQUIRED | TIEBREAK_REQUIRED | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED |
| Replogle_cellline_holdout_2_RPE1 | 128 | 122 | True | EXACT_SET | EXACT_SET | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED |
| Santinha_context_holdout_1_Interneurons_Sst_Pvalb | 51 | 6 | True | TIEBREAK_REQUIRED | TIEBREAK_REQUIRED | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED |
| Santinha_context_holdout_2_Interneurons_Vip_Adarb2 | 51 | 4 | True | TIEBREAK_REQUIRED | TIEBREAK_REQUIRED | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED |
| Santinha_context_holdout_3_Neurons_L_2_3 | 51 | 51 | True | EXACT_SET | EXACT_SET | EVALUABLE_SELECTIVE_RANKING_G4_NOT_EVALUATED |
| Santinha_context_holdout_4_Neurons_L_5 | 51 | 6 | True | TIEBREAK_REQUIRED | TIEBREAK_REQUIRED | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED |
| Santinha_context_holdout_5_Neurons_L_6 | 51 | 6 | True | TIEBREAK_REQUIRED | TIEBREAK_REQUIRED | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED |

## ties 对历史 AURC 的影响

| batch_id | pretruth_evaluation_status | postgate_interpretation_status | candidate_aurc_tie_average | candidate_aurc_best_legal_tie_order | candidate_aurc_worst_legal_tie_order | candidate_aurc_partial_identification_width |
|---|---|---|---|---|---|---|
| Replogle_cellline_holdout_1_K562 | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED | UPSTREAM_VIABILITY_NOT_EVALUATED | 0.0878 | 0.0696 | 0.1074 | 0.0378 |
| Liang_context_holdout_4_DM2_4 | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED | UPSTREAM_VIABILITY_NOT_EVALUATED | 0.0829 | 0.0804 | 0.0855 | 0.0051 |
| frangieh_context_holdout_3_IFNγ | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED | UPSTREAM_VIABILITY_NOT_EVALUATED | 0.0532 | 0.0508 | 0.0557 | 0.0049 |
| Santinha_context_holdout_5_Neurons_L_6 | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED | UPSTREAM_VIABILITY_NOT_EVALUATED | 0.0577 | 0.0555 | 0.0601 | 0.0045 |
| Santinha_context_holdout_2_Interneurons_Vip_Adarb2 | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED | UPSTREAM_VIABILITY_NOT_EVALUATED | 0.0601 | 0.0583 | 0.0620 | 0.0037 |
| Santinha_context_holdout_1_Interneurons_Sst_Pvalb | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED | UPSTREAM_VIABILITY_NOT_EVALUATED | 0.0509 | 0.0492 | 0.0526 | 0.0034 |
| Santinha_context_holdout_4_Neurons_L_5 | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED | UPSTREAM_VIABILITY_NOT_EVALUATED | 0.0465 | 0.0449 | 0.0480 | 0.0031 |
| Nadig_cellline_holdout_1_HepG2 | EVALUABLE_ASSOCIATION_TIE_AWARE_G4_NOT_EVALUATED | UPSTREAM_VIABILITY_NOT_EVALUATED | 0.1439 | 0.1438 | 0.1440 | 0.0002 |

普通 mergesort AURC 会受 CSV 原始行顺序影响。表中的 candidate tie-average、best 和 worst 对任意行排列不变；magnitude 也按相同规则单独计算。这些区间是门后历史真值审计，未参与 G2–G5 状态判定。

E87 的上游预测器在历史 truth 上没有一个任务优于 no-change，因此被明确标记为 `UPSTREAM_PREDICTOR_INVALID`；RIAG 的数学可评价性不能覆盖该失败。

## 下一步

RIAG v2 必须先原样写入新的外部实验合同，再生成预测和风险分数，最后一次性解封 test truth。TianKampmann2019 在旧 SafeTrans 归档中已有历史分析，只能作为桥接重分析；不能再写成 untouched independent confirmation。
