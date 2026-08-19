# SafeConf formal main audit

本报告把已有 PredictionRecord 重新用冻结 `protocol_v0_2_family_confidence` 打分，
并同时报告 effect magnitude（效应大小）控制后的 partial rho 与 bootstrap CI。

## Main table preview

```
                  dataset_name dataset_family    n  aligned_rho  aligned_rho_ci_low  aligned_rho_ci_high  partial_rho_control_magnitude  partial_rho_ci_low  partial_rho_ci_high  magnitude_only_rho  risk_coverage80_improve_pct
                CuiHacohen2023      gene_main 2506     0.445376            0.412539             0.476380                       0.328479            0.293323             0.361506            0.735846                    21.588637
                      Frangieh      gene_main 1266     0.582722            0.536528             0.621097                       0.473820            0.430385             0.510374            0.797334                     5.033196
  LaraAstiasoHuntly2023_exvivo      gene_main  662     0.561460            0.500639             0.615852                       0.430118            0.364618             0.489730            0.512723                    56.119896
  LaraAstiasoHuntly2023_invivo      gene_main  780     0.412561            0.353287             0.467816                       0.357764            0.294477             0.417306            0.639455                    12.826016
        McFarlandTsherniak2020    chem_robust 2326    -0.085934           -0.130622            -0.045968                      -0.060924           -0.098878            -0.023593            0.795446                     3.950613
             SantinhaPlatt2023    chem_robust  566     0.205935            0.118398             0.281585                       0.224185            0.141697             0.300089            0.824004                     2.078644
SrivatsanTrapnell2020_sciplex3    chem_robust 1128     0.427781            0.382460             0.481105                       0.628799            0.595842             0.661401            0.740302                    15.102695
```

## Input status

```
                                                                                                                      run_dir                    dataset_names  n_records  n_scores status
                          /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_cui_go_nogo_probe                 [CuiHacohen2023]      12530    175420     ok
/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_20260604/McFarlandTsherniak2020         [McFarlandTsherniak2020]      11630    162820     ok
                       /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_phase1_main/Frangieh                       [Frangieh]       6330     88620     ok
 /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_phase1_main/SrivatsanTrapnell2020_sciplex3 [SrivatsanTrapnell2020_sciplex3]       5640     78960     ok
              /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_phase1_main/SantinhaPlatt2023              [SantinhaPlatt2023]       2830     39620     ok
   /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_phase1_main/LaraAstiasoHuntly2023_invivo   [LaraAstiasoHuntly2023_invivo]       3910     54740     ok
   /home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_phase1_main/LaraAstiasoHuntly2023_exvivo   [LaraAstiasoHuntly2023_exvivo]       3310     46340     ok
```
