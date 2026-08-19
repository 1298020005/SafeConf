# SafeConf feature ablation audit

This audit recomputes frozen protocol variants from existing records. It does not train perturbation predictors.

## Full protocol summary

```
                  dataset_name dataset_family    n  aligned_rho  partial_rho_control_magnitude  risk_coverage80_improve_pct
                CuiHacohen2023      gene_main 2506     0.445376                       0.328479                    21.581018
                      Frangieh      gene_main 1266     0.582722                       0.473820                     5.019082
  LaraAstiasoHuntly2023_exvivo      gene_main  662     0.561460                       0.430118                    56.119896
  LaraAstiasoHuntly2023_invivo      gene_main  780     0.412561                       0.357764                    12.826016
        McFarlandTsherniak2020    chem_robust 2326    -0.085934                      -0.060924                     3.966416
             SantinhaPlatt2023    chem_robust  566     0.205935                       0.224185                     2.151250
SrivatsanTrapnell2020_sciplex3    chem_robust 1128     0.427781                       0.628799                    15.096599
```

## Leave-one-feature-out delta vs full

Negative delta means removing the feature hurts the metric.

```
                  dataset_name                     score_name  partial_rho_control_magnitude  delta_partial_vs_full  aligned_rho  delta_aligned_vs_full
                CuiHacohen2023      loo_no_context_confidence                       0.188309              -0.140170     0.272459              -0.172917
                CuiHacohen2023 loo_no_disagreement_confidence                       0.238997              -0.089482     0.339032              -0.106344
                CuiHacohen2023      loo_no_support_confidence                       0.361684               0.033205     0.496299               0.050923
                      Frangieh      loo_no_context_confidence                       0.654192               0.180373     0.563116              -0.019606
                      Frangieh loo_no_disagreement_confidence                       0.309565              -0.164255     0.418258              -0.164464
                      Frangieh      loo_no_support_confidence                       0.064905              -0.408914     0.536026              -0.046696
  LaraAstiasoHuntly2023_exvivo      loo_no_context_confidence                       0.224161              -0.205957     0.171846              -0.389614
  LaraAstiasoHuntly2023_exvivo loo_no_disagreement_confidence                       0.216628              -0.213490     0.413502              -0.147958
  LaraAstiasoHuntly2023_exvivo      loo_no_support_confidence                       0.542717               0.112599     0.646762               0.085302
  LaraAstiasoHuntly2023_invivo      loo_no_context_confidence                       0.357280              -0.000484     0.499388               0.086827
  LaraAstiasoHuntly2023_invivo loo_no_disagreement_confidence                       0.220784              -0.136980     0.129615              -0.282946
  LaraAstiasoHuntly2023_invivo      loo_no_support_confidence                       0.311167              -0.046598     0.326182              -0.086379
        McFarlandTsherniak2020      loo_no_context_confidence                      -0.060924               0.000000    -0.085934               0.000000
        McFarlandTsherniak2020 loo_no_disagreement_confidence                      -0.090710              -0.029786    -0.158025              -0.072091
        McFarlandTsherniak2020      loo_no_support_confidence                       0.084337               0.145261     0.153663               0.239597
             SantinhaPlatt2023      loo_no_context_confidence                       0.224185               0.000000     0.205935               0.000000
             SantinhaPlatt2023 loo_no_disagreement_confidence                       0.103339              -0.120847     0.117550              -0.088385
             SantinhaPlatt2023      loo_no_support_confidence                       0.225363               0.001178     0.175713              -0.030222
SrivatsanTrapnell2020_sciplex3      loo_no_context_confidence                       0.628799               0.000000     0.427781               0.000000
SrivatsanTrapnell2020_sciplex3 loo_no_disagreement_confidence                       0.285689              -0.343110     0.176515              -0.251267
SrivatsanTrapnell2020_sciplex3      loo_no_support_confidence                       0.689184               0.060385     0.472204               0.044423
```
