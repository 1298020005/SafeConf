# GEARS confidence evaluation

- input_root: `/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/gears_frangieh_formal_probe_20260605`
- records: 62
- scores: 124

## Evaluation

```
  level   dataset_family dataset_name                      score_name score_type  n  spearman_score_vs_rmse  direction_aligned_spearman  mean_rmse  risk_cov_improve_pct
dataset gears_supplement     frangieh gears_prediction_magnitude_risk       risk 62                0.894085                    0.894085   0.034922             14.809430
dataset gears_supplement     frangieh    gears_uncertainty_confidence confidence 62               -0.095977                    0.095977   0.034922             -0.824733
 family gears_supplement          ALL gears_prediction_magnitude_risk       risk 62                0.894085                    0.894085   0.034922                   NaN
 family gears_supplement          ALL    gears_uncertainty_confidence confidence 62               -0.095977                    0.095977   0.034922                   NaN
overall              ALL          ALL gears_prediction_magnitude_risk       risk 62                0.894085                    0.894085   0.034922                   NaN
overall              ALL          ALL    gears_uncertainty_confidence confidence 62               -0.095977                    0.095977   0.034922                   NaN
```
