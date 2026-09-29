# Dual-Memory Error Adaptation

DEV/SEEN feedback experiment; McFaline test remained sealed.

The shared core is frozen. Every model-specific adapter uses only the same upstream's legal OOF errors. PertEMA is represented by a shallow HGB proxy under the same feedback budget; official-package compatibility remains a separate audit.

## Feedback-curve AUC

```text
        upstream                 method  u20_feedback_auc  spearman_feedback_auc  min_delta_u20_vs_shared
TxPert_Exphormer      DirectHGB_PPublic          0.733995               0.726710                -0.190895
TxPert_Exphormer    DirectRidge_PPublic          0.755925               0.754707                -0.065899
TxPert_Exphormer PertEMA_style_HGB_P_proxy          0.721290               0.697243                -0.183517
TxPert_Exphormer      ResidualHGBShrink          0.786666               0.749749                -0.013161
TxPert_Exphormer    ResidualRidgeShrink          0.790975               0.759539                -0.008732
TxPert_Exphormer             SharedCore          0.790606               0.752125                 0.000000
TxPert_Exphormer  ShuffledErrorResidual          0.775203               0.733777                -0.027910
TxPert_Exphormer  WrongUpstreamResidual          0.787885               0.747506                -0.010106
      TxPert_GAT      DirectHGB_PPublic          0.746840               0.731974                -0.182129
      TxPert_GAT    DirectRidge_PPublic          0.774068               0.762135                -0.064080
      TxPert_GAT PertEMA_style_HGB_P_proxy          0.734333               0.706531                -0.149025
      TxPert_GAT      ResidualHGBShrink          0.787412               0.754101                -0.011072
      TxPert_GAT    ResidualRidgeShrink          0.791606               0.766414                -0.009709
      TxPert_GAT             SharedCore          0.789658               0.758659                 0.000000
      TxPert_GAT  ShuffledErrorResidual          0.769124               0.738105                -0.039145
      TxPert_GAT  WrongUpstreamResidual          0.794876               0.754663                -0.004415
```

## Paired bootstrap

```text
        upstream  budget                  comparison              method_a               method_b  delta_utility20  ci95_lower  ci95_upper  bootstrap_replicates
TxPert_Exphormer    0.10       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore        -0.005771   -0.024651    0.012960                  5000
TxPert_Exphormer    0.10      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.174718    0.118730    0.235861                  5000
TxPert_Exphormer    0.10 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.187616    0.134565    0.245503                  5000
TxPert_Exphormer    0.10     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore        -0.005026   -0.023261    0.013845                  5000
TxPert_Exphormer    0.25       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore         0.004609   -0.017054    0.025237                  5000
TxPert_Exphormer    0.25      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.094447    0.057599    0.134639                  5000
TxPert_Exphormer    0.25 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.089716    0.046196    0.132259                  5000
TxPert_Exphormer    0.25     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore         0.003621   -0.018504    0.026785                  5000
TxPert_Exphormer    0.50       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore         0.003797   -0.022044    0.028575                  5000
TxPert_Exphormer    0.50      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.040661    0.006560    0.075778                  5000
TxPert_Exphormer    0.50 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.046768    0.011872    0.083059                  5000
TxPert_Exphormer    0.50     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore        -0.000420   -0.028986    0.026674                  5000
TxPert_Exphormer    0.75       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore        -0.008529   -0.038561    0.020920                  5000
TxPert_Exphormer    0.75      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.047630    0.012893    0.085420                  5000
TxPert_Exphormer    0.75 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.013260   -0.012301    0.039101                  5000
TxPert_Exphormer    0.75     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore        -0.005131   -0.035220    0.024289                  5000
TxPert_Exphormer    1.00       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore        -0.001326   -0.026672    0.025233                  5000
TxPert_Exphormer    1.00      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.047393    0.011546    0.086097                  5000
TxPert_Exphormer    1.00 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.016900   -0.011058    0.045299                  5000
TxPert_Exphormer    1.00     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore         0.001951   -0.027859    0.031462                  5000
      TxPert_GAT    0.10       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore        -0.010154   -0.028362    0.008171                  5000
      TxPert_GAT    0.10      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.147667    0.093409    0.208393                  5000
      TxPert_GAT    0.10 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.174803    0.120820    0.233588                  5000
      TxPert_GAT    0.10     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore        -0.005981   -0.024117    0.011510                  5000
      TxPert_GAT    0.25       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore        -0.002419   -0.023558    0.018998                  5000
      TxPert_GAT    0.25      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.084458    0.043629    0.130066                  5000
      TxPert_GAT    0.25 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.077682    0.031477    0.125088                  5000
      TxPert_GAT    0.25     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore        -0.002962   -0.027358    0.018771                  5000
      TxPert_GAT    0.50       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore         0.000220   -0.029247    0.029255                  5000
      TxPert_GAT    0.50      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.044724    0.003689    0.085719                  5000
      TxPert_GAT    0.50 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.016979   -0.016634    0.051029                  5000
      TxPert_GAT    0.50     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore         0.003746   -0.021623    0.029090                  5000
      TxPert_GAT    0.75       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore        -0.006985   -0.035855    0.023007                  5000
      TxPert_GAT    0.75      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.027828   -0.009365    0.067246                  5000
      TxPert_GAT    0.75 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.010819   -0.019183    0.041644                  5000
      TxPert_GAT    0.75     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore         0.004096   -0.026173    0.033440                  5000
      TxPert_GAT    1.00       ResidualHGB_vs_Shared     ResidualHGBShrink             SharedCore         0.003399   -0.023375    0.031003                  5000
      TxPert_GAT    1.00      ResidualHGB_vs_PertEMA     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.035173    0.001231    0.071297                  5000
      TxPert_GAT    1.00 ResidualHGB_vs_DirectPublic     ResidualHGBShrink      DirectHGB_PPublic         0.012792   -0.020162    0.046496                  5000
      TxPert_GAT    1.00     WrongUpstream_vs_Shared WrongUpstreamResidual             SharedCore         0.006351   -0.019059    0.032774                  5000
```
