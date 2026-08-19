# C2 task-risk source explanation

This is a descriptive stratified table, not a causal explanation. It avoids SHAP and avoids claiming that learned-model feature patterns are discoveries.

- Feature root: `/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609`
- Merged score-feature rows: 27504
- Features summarized: ['context_similarity_max', 'context_similarity_mean', 'perturbation_support_count', 'perturbation_effect_stability', 'perturbation_effect_variance', 'historical_residual_risk', 'model_disagreement_rmse', 'model_disagreement_cosine', 'ood_nearest_distance', 'ood_mean_k_distance', 'prediction_l2_norm', 'prediction_abs_mean', 'prediction_magnitude_deviation', 'prediction_norm_ratio']

## Main caution

For learned risk models, high-risk feature patterns can partly reflect the model's own inputs. Treat these tables as interpretation aids, not as independent biological findings.

Stronger caution for `safeconf_lodo_risk`: `prediction_l2_norm` is one of the model-side magnitude-related inputs. If high LODO risk is separated mainly by `prediction_l2_norm`, that can partly reflect the learned model's own magnitude dependence rather than an independent biological discovery. The next cleaner explanation should repeat this analysis after controlling or stratifying by magnitude.

## Largest high-vs-low differences for safeconf_lodo_risk

```text
                  dataset_name                 feature_name  low_risk_median  high_risk_median  high_minus_low
  LaraAstiasoHuntly2023_invivo           prediction_l2_norm        76.503422        399.677658      323.174236
  LaraAstiasoHuntly2023_invivo perturbation_effect_variance         5.086579        116.439186      111.352607
  LaraAstiasoHuntly2023_exvivo           prediction_l2_norm        24.267807         67.506714       43.238907
        McFarlandTsherniak2020           prediction_l2_norm        45.144993         80.835598       35.690605
             SantinhaPlatt2023           prediction_l2_norm         7.626786         18.737221       11.110435
                CuiHacohen2023           prediction_l2_norm         5.995552         15.632776        9.637225
  LaraAstiasoHuntly2023_invivo     historical_residual_risk         1.555755          8.029567        6.473812
  LaraAstiasoHuntly2023_invivo      model_disagreement_rmse         0.720200          6.574828        5.854628
  LaraAstiasoHuntly2023_exvivo perturbation_effect_variance         0.094850          5.411076        5.316226
        McFarlandTsherniak2020   perturbation_support_count        61.000000         66.000000        5.000000
  LaraAstiasoHuntly2023_invivo        prediction_norm_ratio         0.714884          3.679859        2.964976
SrivatsanTrapnell2020_sciplex3           prediction_l2_norm         1.207578          3.686434        2.478856
```
