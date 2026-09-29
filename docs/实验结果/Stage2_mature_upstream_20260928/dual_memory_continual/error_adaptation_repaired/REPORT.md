# Dual-Memory Error Adaptation

DEV/SEEN feedback experiment; McFaline test remained sealed.

The shared core is frozen. Every model-specific adapter uses only the same upstream's legal OOF errors. A shallow HGB provides a PertEMA-style proxy under the same feedback budget; it is not presented as the official implementation, whose compatibility is audited separately.

## Feedback-curve AUC

```text
        upstream                    method  u20_feedback_auc  spearman_feedback_auc  min_delta_u20_vs_shared
TxPert_Exphormer         DirectHGB_PPublic          0.733995               0.726710                -0.203451
TxPert_Exphormer       DirectRidge_PPublic          0.755925               0.754707                -0.078454
TxPert_Exphormer PertEMA_style_HGB_P_proxy          0.721290               0.697243                -0.196072
TxPert_Exphormer         ResidualHGBShrink          0.790206               0.748341                -0.019698
TxPert_Exphormer       ResidualRidgeShrink          0.785670               0.760917                -0.021424
TxPert_Exphormer                SharedCore          0.803162               0.754013                 0.000000
TxPert_Exphormer     ShuffledErrorResidual          0.774534               0.735713                -0.045738
TxPert_Exphormer     WrongUpstreamResidual          0.796250               0.749050                -0.013314
      TxPert_GAT         DirectHGB_PPublic          0.746840               0.731974                -0.190053
      TxPert_GAT       DirectRidge_PPublic          0.774068               0.762135                -0.072004
      TxPert_GAT PertEMA_style_HGB_P_proxy          0.734333               0.706531                -0.156950
      TxPert_GAT         ResidualHGBShrink          0.792046               0.753415                -0.019522
      TxPert_GAT       ResidualRidgeShrink          0.789250               0.765049                -0.014277
      TxPert_GAT                SharedCore          0.797582               0.759108                 0.000000
      TxPert_GAT     ShuffledErrorResidual          0.777517               0.737613                -0.043615
      TxPert_GAT     WrongUpstreamResidual          0.794125               0.753512                -0.018727
```

## Paired bootstrap

```text
        upstream  budget                  comparison              method_a                  method_b  delta_utility20  ci95_lower  ci95_upper  bootstrap_replicates
TxPert_Exphormer    0.10       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.010561   -0.029918    0.008497                  5000
TxPert_Exphormer    0.10 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.177028    0.120287    0.239288                  5000
TxPert_Exphormer    0.10 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.189926    0.137140    0.247519                  5000
TxPert_Exphormer    0.10     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore        -0.007459   -0.024845    0.008991                  5000
TxPert_Exphormer    0.25       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.000653   -0.023363    0.023835                  5000
TxPert_Exphormer    0.25 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.096643    0.060104    0.136426                  5000
TxPert_Exphormer    0.25 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.091912    0.048175    0.134327                  5000
TxPert_Exphormer    0.25     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore         0.003947   -0.019079    0.031297                  5000
TxPert_Exphormer    0.50       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.007300   -0.033691    0.019901                  5000
TxPert_Exphormer    0.50 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.036718    0.005953    0.070260                  5000
TxPert_Exphormer    0.50 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.042825    0.008594    0.080292                  5000
TxPert_Exphormer    0.50     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore         0.001003   -0.026115    0.030849                  5000
TxPert_Exphormer    0.75       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.009519   -0.039516    0.021753                  5000
TxPert_Exphormer    0.75 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.054049    0.017509    0.092574                  5000
TxPert_Exphormer    0.75 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.019679   -0.009627    0.047440                  5000
TxPert_Exphormer    0.75     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore        -0.002844   -0.031685    0.029109                  5000
TxPert_Exphormer    1.00       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.005902   -0.032677    0.024419                  5000
TxPert_Exphormer    1.00 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.049928    0.015493    0.087133                  5000
TxPert_Exphormer    1.00 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.019435   -0.009079    0.048873                  5000
TxPert_Exphormer    1.00     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore         0.003890   -0.023438    0.033039                  5000
      TxPert_GAT    0.10       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.014170   -0.033584    0.003636                  5000
      TxPert_GAT    0.10 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.150749    0.094291    0.212962                  5000
      TxPert_GAT    0.10 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.177885    0.121790    0.238898                  5000
      TxPert_GAT    0.10     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore        -0.014378   -0.035091    0.006612                  5000
      TxPert_GAT    0.25       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.003375   -0.029123    0.020421                  5000
      TxPert_GAT    0.25 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.090580    0.049046    0.135992                  5000
      TxPert_GAT    0.25 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.083804    0.035597    0.130398                  5000
      TxPert_GAT    0.25     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore        -0.009734   -0.036240    0.016596                  5000
      TxPert_GAT    0.50       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.007078   -0.036847    0.022277                  5000
      TxPert_GAT    0.50 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.044612    0.007216    0.082332                  5000
      TxPert_GAT    0.50 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.016868   -0.016477    0.051134                  5000
      TxPert_GAT    0.50     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore        -0.008050   -0.030953    0.014541                  5000
      TxPert_GAT    0.75       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.014716   -0.044703    0.018545                  5000
      TxPert_GAT    0.75 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.027014   -0.013602    0.072829                  5000
      TxPert_GAT    0.75 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.010005   -0.023091    0.045431                  5000
      TxPert_GAT    0.75     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore        -0.002788   -0.030331    0.027409                  5000
      TxPert_GAT    1.00       ResidualHGB_vs_Shared     ResidualHGBShrink                SharedCore        -0.000272   -0.024549    0.026083                  5000
      TxPert_GAT    1.00 ResidualHGB_vs_PertEMAProxy     ResidualHGBShrink PertEMA_style_HGB_P_proxy         0.038553    0.005270    0.074701                  5000
      TxPert_GAT    1.00 ResidualHGB_vs_DirectPublic     ResidualHGBShrink         DirectHGB_PPublic         0.016171   -0.015636    0.049541                  5000
      TxPert_GAT    1.00     WrongUpstream_vs_Shared WrongUpstreamResidual                SharedCore        -0.003564   -0.027475    0.022510                  5000
```
