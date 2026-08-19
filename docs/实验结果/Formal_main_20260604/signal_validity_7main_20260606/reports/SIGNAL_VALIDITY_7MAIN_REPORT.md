# Signal validity audit for 7 formal main datasets

本审计只使用 `Formal_main_20260604` 的正式打分结果，不引用旧 v6 signal validity 作为主表证据。

目的：检查 SafeConf 的 confidence/risk 信号是否只是 effect magnitude（效应大小）伪相关。

## Main score preview

```
                  dataset_name dataset_family    n  raw_spearman  magnitude_l2_baseline_rho  partial_rho_control_magnitude  within_perturbation_weighted_rho  within_context_weighted_rho
                CuiHacohen2023      gene_main 2506      0.445376                   0.735846                       0.328479                          0.380367                     0.247001
                      Frangieh      gene_main 1266      0.582722                   0.797334                       0.473820                          0.633531                     0.597478
  LaraAstiasoHuntly2023_exvivo      gene_main  662      0.561460                   0.512723                       0.430118                          0.513652                     0.253057
  LaraAstiasoHuntly2023_invivo      gene_main  780      0.412561                   0.639455                       0.357764                          0.225893                     0.403836
        McFarlandTsherniak2020    chem_robust 2326     -0.085934                   0.795446                      -0.060924                          0.056881                    -0.166740
             SantinhaPlatt2023    chem_robust  566      0.205935                   0.824004                       0.224185                          0.018801                     0.374294
SrivatsanTrapnell2020_sciplex3    chem_robust 1128      0.427781                   0.740302                       0.628799                          0.074433                     0.537566
```

## Interpretation notes

- `raw_spearman`: aligned score vs raw RMSE.
- `magnitude_l2_baseline_rho`: true effect magnitude vs raw RMSE; high values indicate possible magnitude confounding.
- `partial_rho_control_magnitude`: score vs RMSE after controlling true effect magnitude ranks.
- `within_perturbation_weighted_rho`: whether the score still ranks errors within the same perturbation.
- `within_context_weighted_rho`: whether the score still ranks errors within the same context.

## Scope guard

- Tahoe remains sampled/smoke-tagged external validation, not part of this 7-main formal signal-validity table.
- GEARS alignment is not forced into this table because the context space is mismatched.
