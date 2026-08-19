# CuiHacohen2023 go/no-go report

这是 CuiHacohen2023（崔，免疫细胞基因扰动大数据集）的第一关报告。

核心问题：SafeConf（单细胞扰动预测结果可信度打分）在大数据集上是否真的有信号。

## Decision

- decision（结论）: **GO**
- reason（原因）: 冻结 protocol v0.2 在 Cui 上超过 0.30，可以全速推进下一批大数据集。

## Main score

- protocol v0.2 aligned rho（方向对齐相关）: 0.4454
- partial rho control magnitude（控制效应大小后的相关）: 0.3285
- magnitude-only rho（只看效应大小的基线）: 0.7358
- risk-coverage@80% improvement（保留 80% 低风险预测后的误差改善）: 21.5886%

## Single-feature diagnosis

- best single feature（最好单特征）: context_similarity_score
- best single aligned rho: 0.4396
- best single partial rho: 0.2840

## Per-fold main-score rho

```
                     score_name  fold_id   n  aligned_rho  partial_rho_control_magnitude  risk_coverage80_improve_pct
protocol_v0_2_family_confidence        0 502     0.430311                       0.343793                    19.335415
protocol_v0_2_family_confidence        1 502     0.408995                       0.299095                    16.501694
protocol_v0_2_family_confidence        2 500     0.459319                       0.368610                    22.261526
protocol_v0_2_family_confidence        3 500     0.502144                       0.319035                    26.749109
protocol_v0_2_family_confidence        4 502     0.418431                       0.312428                    20.238593
```

## Top score summary

```
                             score_name score_type    n  aligned_rho  raw_score_vs_rmse_rho  normalized_rmse_rho  magnitude_only_rho  partial_rho_control_magnitude  risk_coverage80_improve_pct  mean_rmse
                     learned_risk_score       risk 2506     0.675681               0.675681            -0.006931            0.735846                       0.435665                    24.196931   0.211044
             simple_combined_confidence confidence 2506     0.486242              -0.486242             0.144234            0.735846                       0.384813                    21.895125   0.211044
        protocol_v0_2_family_confidence confidence 2506     0.445376              -0.445376             0.109108            0.735846                       0.328479                    21.588637   0.211044
               context_similarity_score confidence 2506     0.439631              -0.439631             0.059975            0.735846                       0.284047                    21.796668   0.211044
protocol_v0_2_with_stability_confidence confidence 2506     0.360663              -0.360663             0.253450            0.735846                       0.358732                    20.940688   0.211044
                model_disagreement_risk       risk 2506     0.354409               0.354409             0.018880            0.735846                       0.224825                    13.211304   0.211044
               historical_residual_risk       risk 2506     0.272738               0.272738             0.022517            0.735846                       0.199830                     6.111398   0.211044
                      ood_distance_risk       risk 2506     0.053483               0.053483             0.342691            0.735846                       0.221095                     4.790910   0.211044
                    support_count_score confidence 2506     0.036195              -0.036195             0.063970            0.735846                       0.069104                     0.884921   0.211044
                           random_score confidence 2506     0.004888              -0.004888             0.006783            0.735846                      -0.001992                     1.921986   0.211044
           perturbation_stability_score confidence 2506    -0.203064               0.203064             0.312115            0.735846                       0.080405                    -4.897559   0.211044
              prediction_magnitude_risk       risk 2506    -0.209962              -0.209962             0.216432            0.735846                       0.008771                    -6.240056   0.211044
```

## Predictor summary

```
                     score_name     predictor_name    n  aligned_rho  raw_score_vs_rmse_rho  normalized_rmse_rho  magnitude_only_rho  partial_rho_control_magnitude  risk_coverage80_improve_pct  mean_rmse
        model_disagreement_risk ContextSimBaseline 1253     0.271159               0.271159            -0.075032            0.730172                       0.098223                    11.355549   0.209447
        model_disagreement_risk   V0StrongBaseline 1253     0.437161               0.437161             0.115180            0.742333                       0.353132                    14.689364   0.212642
protocol_v0_2_family_confidence ContextSimBaseline 1253     0.420221              -0.420221             0.086775            0.730172                       0.289491                    21.583236   0.209447
protocol_v0_2_family_confidence   V0StrongBaseline 1253     0.470236              -0.470236             0.131922            0.742333                       0.367858                    21.604918   0.212642
```

## How to read this

- aligned rho（方向对齐相关）越高，说明高风险/低可信预测更容易真的错。
- magnitude-only rho（效应大小基线）很高时，要小心分数只是看见了扰动效应大。
- partial rho（控制效应大小后的相关）还为正，才说明分数可能有独立价值。
- per-fold（每折）都要看，不能只看一个 pooled（混合总数）。
