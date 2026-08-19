# E18 真实模型预测向量资产审计

生成时间：2026-07-07 21:30

## 1. 结论

E18 回答一个很具体的问题：本地是否已有 GEARS、CPA、scGPT 这类真实模型的逐任务预测向量，足以支撑“模型级 SafeConf 验证”。

结论：GEARS 有部分可用资产，scGPT 和 CPA 当前没有可直接进入 SafeConf 协议的 PredictionRecord + predicted/true vector。GEARS 的资产可读，但只覆盖 Norman、Adamson、Dixit 的 single-gene 任务，合计 54 条记录；它不能直接替代 sciplex3 full-743 的多模型验证。

## 2. 模型级就绪度

| model | local_code_or_data | prediction_records | runs_or_seeds | datasets | gene_space | predicted_vectors | true_vectors | native_uncertainty | proxy_uncertainty | score_rows | readiness | blocking_reason | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GEARS | True | 54 | 9 | 3 | 5025,5043,6000 | True | True | False | True | 5 | PARTIAL_READY_GEARS_ONLY | Records/vectors exist, but only for GEARS on Norman/Adamson/Dixit single-gene tasks; not aligned to sciplex3 full-743, CPA or scGPT; native uncertainty absent. | Use as GEARS-only supplement or rerun/align GEARS with the same task_id/gene order as current benchmarks. |
| scGPT | True | 0 | 0 | 0 |  | False | False | False | False | 0 | NOT_READY_NO_PREDICTION_RECORDS | Local environment/archive exists, but no SafeConf PredictionRecord + predicted_effect vectors were found. | Implement or locate a scGPT adapter that exports PREDICTION_RECORDS.csv plus predicted/true NPZ on a frozen benchmark. |
| CPA / chemCPA | False | 0 | 0 | 0 |  | False | False | False | False | 0 | NOT_READY_NO_LOCAL_VECTOR_OUTPUT | Only literature/PDF traces were found; no local CPA vector outputs under the SafeConf contract. | Treat CPA as future external adapter work; do not claim completed CPA validation. |

## 3. GEARS 运行与数组审计

| dataset | seed | status | records_n_rows | records_csv_reported_exists | records_csv_repaired_exists | predicted_npz_repaired_exists | predicted_npz_n_keys | predicted_npz_first_shape | predicted_keys_match_records | true_npz_repaired_exists | true_npz_n_keys | true_keys_match_records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| norman | 1 | ok | 10 | False | True | True | 10 | 5025 | True | True | 10 | True |
| norman | 2 | ok | 10 | False | True | True | 10 | 5025 | True | True | 10 | True |
| norman | 3 | ok | 10 | False | True | True | 10 | 5025 | True | True | 10 | True |
| adamson | 1 | ok | 7 | False | True | True | 7 | 5043 | True | True | 7 | True |
| adamson | 2 | ok | 7 | False | True | True | 7 | 5043 | True | True | 7 | True |
| adamson | 3 | ok | 7 | False | True | True | 7 | 5043 | True | True | 7 | True |
| dixit | 1 | ok | 1 | False | True | True | 1 | 6000 | True | True | 1 | True |
| dixit | 2 | ok | 1 | False | True | True | 1 | 6000 | True | True | 1 | True |
| dixit | 3 | ok | 1 | False | True | True | 1 | 6000 | True | True | 1 | True |

## 4. GEARS 数据集规模

| dataset_name | n_records | n_unique_perturbations | n_seeds | mean_rmse | median_rmse | mean_cosine_error |
| --- | --- | --- | --- | --- | --- | --- |
| adamson | 21 | 18 | 3 | 0.0420858 | 0.0347614 | 0.631153 |
| dixit | 3 | 2 | 3 | 0.424841 | 0.371982 | 0.306773 |
| norman | 30 | 27 | 3 | 0.0558785 | 0.0521798 | 0.671322 |

## 5. 已有 GEARS 风险分数

| level | dataset_family | dataset_name | score_name | score_type | n | spearman_score_vs_rmse | direction_aligned_spearman | mean_rmse | risk_cov_improve_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset | gears_supplement | adamson | gears_prediction_magnitude_risk | risk | 21 | 0.422078 | 0.422078 | 0.0420858 | 0.81296 |
| dataset | gears_supplement | dixit | gears_prediction_magnitude_risk | risk | 3 | 0.5 | 0.5 | 0.424841 | 0 |
| dataset | gears_supplement | norman | gears_prediction_magnitude_risk | risk | 30 | 0.623582 | 0.623582 | 0.0558785 | 7.6099 |
| family | gears_supplement | ALL | gears_prediction_magnitude_risk | risk | 54 | 0.623937 | 0.623937 | 0.0710125 |  |
| overall | ALL | ALL | gears_prediction_magnitude_risk | risk | 54 | 0.623937 | 0.623937 | 0.0710125 |  |

## 6. GEARS 不确定性

| asset | exists | n_rows |
| --- | --- | --- |
| GEARS_RECORDS_FOR_UNCERTAINTY.csv | True | 54 |
| GEARS_UNCERTAINTY_SCORES.csv | True | 12 |
| native_uncertainty | False | 0 |
| seed_ensemble_proxy | True | 12 |

## 7. 可以说与不能说

可以说：

- GEARS 的 PredictionRecord 与 predicted/true NPZ 在 `safeconf_runtime/outputs/gears_prediction_records_formal` 下可读。
- 旧 status 文件中的 `/home/yyf/codex_cout/...` 绝对路径已失效，但能映射到 `/home/yyf/safeconf_runtime/outputs/...`。
- GEARS-only 的 magnitude risk 在 54 条记录上 aligned Spearman = 0.624。

不能说：

- 不能说 GEARS、CPA、scGPT 已在同一 benchmark、同一 split、同一 gene space 下完成模型级验证。
- 不能说 sciplex3 full-743 已经接入 GEARS/CPA/scGPT。
- 不能把 GEARS 的 perturbation-level 或 seed-level 汇总指标当作逐任务置信度。

## 8. 下一步

1. 把 E18 作为模型级扩展的入口审计。
2. 若短期需要实证结果，先做 GEARS-only supplement：只在 Norman/Adamson/Dixit 54 条记录上报告。
3. 若目标是主线升级，必须重新定义一个 shared benchmark，把 GEARS、CPA、scGPT 都导出为统一的 `task_id + gene_order + predicted_effect_key + true_effect_key`。
