# SafeConf formal main audit

本报告把已有 PredictionRecord 重新用冻结 `protocol_v0_2_family_confidence` 打分，
并同时报告 effect magnitude（效应大小）控制后的 partial rho 与 bootstrap CI。

## Main table preview

```
                  dataset_name dataset_family    n  aligned_rho  aligned_rho_ci_low  aligned_rho_ci_high  partial_rho_control_magnitude  partial_rho_ci_low  partial_rho_ci_high  magnitude_only_rho  risk_coverage80_improve_pct
                CuiHacohen2023      gene_main 2506     0.445376            0.412539             0.476380                       0.328479            0.293323             0.361506            0.735846                    21.581018
                      Frangieh      gene_main 1266     0.582722            0.536528             0.621097                       0.473820            0.430385             0.510374            0.797334                     5.033196
  LaraAstiasoHuntly2023_exvivo      gene_main  646     0.563151            0.498936             0.620518                       0.442952            0.376149             0.505676            0.486006                    56.186238
  LaraAstiasoHuntly2023_invivo      gene_main  750     0.394162            0.329535             0.454790                       0.356783            0.289803             0.423739            0.634214                    12.513362
        McFarlandTsherniak2020    chem_robust 2326    -0.085934           -0.127408            -0.042752                      -0.060924           -0.099611            -0.022708            0.795446                     3.950613
             SantinhaPlatt2023      gene_main  546     0.152410            0.070712             0.230650                       0.212387            0.128622             0.296609            0.839854                    -1.037155
SrivatsanTrapnell2020_sciplex3    chem_robust 1128     0.427781            0.375344             0.476642                       0.628799            0.594810             0.660198            0.740302                    15.102695
```

## Input status

```
                                                                                                                                                   run_dir                    dataset_names  n_records  n_scores status
    /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/safeconf_cui_go_nogo_probe                 [CuiHacohen2023]      12530    175420     ok
                      /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/Frangieh                       [Frangieh]       6330     88620     ok
  /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/LaraAstiasoHuntly2023_exvivo   [LaraAstiasoHuntly2023_exvivo]       3230     45220     ok
  /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/LaraAstiasoHuntly2023_invivo   [LaraAstiasoHuntly2023_invivo]       3760     52640     ok
        /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/McFarlandTsherniak2020         [McFarlandTsherniak2020]      11630    162820     ok
             /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/SantinhaPlatt2023              [SantinhaPlatt2023]       2730     38220     ok
/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/SrivatsanTrapnell2020_sciplex3 [SrivatsanTrapnell2020_sciplex3]       5640     78960     ok
```
