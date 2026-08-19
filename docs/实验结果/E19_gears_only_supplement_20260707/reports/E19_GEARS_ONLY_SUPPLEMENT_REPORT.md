# E19 GEARS-only supplement consolidation

生成时间：2026-07-07 21:35

## 1. 结论

E19 汇总本地已有 GEARS-only 结果，目标是形成一个可引用的补充证据包。它不等价于 GEARS、CPA、scGPT 的统一多模型验证。

核心结果：

| source_id | source_label | source_status | exists | n_records | n_scores | n_datasets | datasets | best_score_name | best_aligned_spearman | best_n | has_uncertainty_confidence | boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | True | 54 | 54 | 3 | adamson,dixit,norman | gears_prediction_magnitude_risk | 0.623937 | 54 | False | GEARS-only; no native uncertainty; not aligned with sciplex3/CPA/scGPT. |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | True | 62 | 124 | 1 | frangieh | gears_prediction_magnitude_risk | 0.941125 | 62 | True | GEARS-only Frangieh run; source name contains smoke lineage; use as supplementary signal, not main formal claim. |

## 2. Dataset-level summary

| source_id | source_label | dataset_name | n_records | n_unique_perturbations | n_seeds | mean_rmse | median_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | adamson | 21 | 18 | 3 | 0.0420858 | 0.0347614 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | dixit | 3 | 2 | 3 | 0.424841 | 0.371982 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | norman | 30 | 27 | 3 | 0.0558785 | 0.0521798 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | frangieh | 62 | 58 | 3 | 0.0383705 | 0.0341559 |

## 3. Evaluation summary

| source_id | source_label | source_status | level | dataset_family | dataset_name | score_name | score_type | n | spearman_score_vs_rmse | direction_aligned_spearman | mean_rmse | risk_cov_improve_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | dataset | gears_supplement | adamson | gears_prediction_magnitude_risk | risk | 21 | 0.422078 | 0.422078 | 0.0420858 | 0.81296 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | dataset | gears_supplement | dixit | gears_prediction_magnitude_risk | risk | 3 | 0.5 | 0.5 | 0.424841 | 0 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | dataset | gears_supplement | norman | gears_prediction_magnitude_risk | risk | 30 | 0.623582 | 0.623582 | 0.0558785 | 7.6099 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | family | gears_supplement | ALL | gears_prediction_magnitude_risk | risk | 54 | 0.623937 | 0.623937 | 0.0710125 |  |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | overall | ALL | ALL | gears_prediction_magnitude_risk | risk | 54 | 0.623937 | 0.623937 | 0.0710125 |  |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | dataset | gears_supplement | frangieh | gears_prediction_magnitude_risk | risk | 62 | 0.941125 | 0.941125 | 0.0383705 | 12.8177 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | dataset | gears_supplement | frangieh | gears_uncertainty_confidence | confidence | 62 | 0.20143 | -0.20143 | 0.0383705 | 1.02977 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | family | gears_supplement | ALL | gears_prediction_magnitude_risk | risk | 62 | 0.941125 | 0.941125 | 0.0383705 |  |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | family | gears_supplement | ALL | gears_uncertainty_confidence | confidence | 62 | 0.20143 | -0.20143 | 0.0383705 |  |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | overall | ALL | ALL | gears_prediction_magnitude_risk | risk | 62 | 0.941125 | 0.941125 | 0.0383705 |  |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | overall | ALL | ALL | gears_uncertainty_confidence | confidence | 62 | 0.20143 | -0.20143 | 0.0383705 |  |

## 4. Risk coverage

| source_id | source_label | source_status | dataset_family | dataset_name | score_name | score_type | coverage | mean_rmse | full_mean_rmse | risk_cov_improve_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | adamson | gears_prediction_magnitude_risk | risk | 1 | 0.0420858 | 0.0420858 | 0 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | adamson | gears_prediction_magnitude_risk | risk | 0.9 | 0.0419322 | 0.0420858 | 0.364985 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | adamson | gears_prediction_magnitude_risk | risk | 0.8 | 0.0417437 | 0.0420858 | 0.81296 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | adamson | gears_prediction_magnitude_risk | risk | 0.7 | 0.0418309 | 0.0420858 | 0.605628 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | adamson | gears_prediction_magnitude_risk | risk | 0.6 | 0.0418366 | 0.0420858 | 0.592072 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | adamson | gears_prediction_magnitude_risk | risk | 0.5 | 0.0405705 | 0.0420858 | 3.60052 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | dixit | gears_prediction_magnitude_risk | risk | 1 | 0.424841 | 0.424841 | 0 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | dixit | gears_prediction_magnitude_risk | risk | 0.9 | 0.424841 | 0.424841 | 0 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | dixit | gears_prediction_magnitude_risk | risk | 0.8 | 0.424841 | 0.424841 | 0 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | dixit | gears_prediction_magnitude_risk | risk | 0.7 | 0.424841 | 0.424841 | 0 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | dixit | gears_prediction_magnitude_risk | risk | 0.6 | 0.363818 | 0.424841 | 14.3636 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | dixit | gears_prediction_magnitude_risk | risk | 0.5 | 0.363818 | 0.424841 | 14.3636 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | norman | gears_prediction_magnitude_risk | risk | 1 | 0.0558785 | 0.0558785 | 0 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | norman | gears_prediction_magnitude_risk | risk | 0.9 | 0.0530659 | 0.0558785 | 5.03331 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | norman | gears_prediction_magnitude_risk | risk | 0.8 | 0.0516262 | 0.0558785 | 7.6099 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | norman | gears_prediction_magnitude_risk | risk | 0.7 | 0.0492989 | 0.0558785 | 11.7748 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | norman | gears_prediction_magnitude_risk | risk | 0.6 | 0.0477724 | 0.0558785 | 14.5066 |
| GEARS_FORMAL_54 | Norman/Adamson/Dixit formal | formal_existing | gears_supplement | norman | gears_prediction_magnitude_risk | risk | 0.5 | 0.0471499 | 0.0558785 | 15.6206 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_prediction_magnitude_risk | risk | 1 | 0.0383705 | 0.0383705 | 0 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_prediction_magnitude_risk | risk | 0.9 | 0.0348926 | 0.0383705 | 9.06403 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_prediction_magnitude_risk | risk | 0.8 | 0.0334523 | 0.0383705 | 12.8177 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_prediction_magnitude_risk | risk | 0.7 | 0.03293 | 0.0383705 | 14.1788 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_prediction_magnitude_risk | risk | 0.6 | 0.032369 | 0.0383705 | 15.6408 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_prediction_magnitude_risk | risk | 0.5 | 0.0317278 | 0.0383705 | 17.312 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_uncertainty_confidence | confidence | 1 | 0.0383705 | 0.0383705 | 0 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_uncertainty_confidence | confidence | 0.9 | 0.0382611 | 0.0383705 | 0.285113 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_uncertainty_confidence | confidence | 0.8 | 0.0379753 | 0.0383705 | 1.02977 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_uncertainty_confidence | confidence | 0.7 | 0.0384606 | 0.0383705 | -0.234878 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_uncertainty_confidence | confidence | 0.6 | 0.0395162 | 0.0383705 | -2.98596 |
| GEARS_FRANGIEH_62 | Frangieh third-predictor run03 | supplement_probe_existing | gears_supplement | frangieh | gears_uncertainty_confidence | confidence | 0.5 | 0.0396407 | 0.0383705 | -3.31031 |

## 5. 边界

- `GEARS_FORMAL_54` 是 Norman/Adamson/Dixit 的正式旧结果，整体 magnitude risk aligned Spearman = 0.624。
- `GEARS_FRANGIEH_62` 是 Frangieh third-predictor run03，包含 uncertainty confidence，但来源目录带 smoke lineage；可以作为补充探针，不能单独变成主线 formal claim。
- 两者都不是 sciplex3 full-743，也没有 CPA/scGPT 的同任务向量。

## 6. 下一步

1. 若写补充材料，可用 E19 说明 GEARS-only 预测输出可被 SafeConf 风格审计。
2. 若写主线结论，必须另建 unified adapter：同一任务、同一 gene order、同一 split 下同时导出 GEARS、CPA、scGPT。
