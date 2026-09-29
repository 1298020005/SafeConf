# Zero-target-error cross-predictor transfer

The source upstream supplies all risk labels. The target upstream supplies zero error labels to fitting, preprocessing, scaling or error-CDF construction. Source rows from the queried biological fold are excluded.

## Summary

| source_upstream | target_upstream | method | n_strata | utility20 | spearman | aurc |
|---|---|---|---|---|---|---|
| TxPert_Exphormer | TxPert_GAT | HGB_ManualPublic | 4 | 0.789601 | 0.751049 | 0.048760 |
| TxPert_Exphormer | TxPert_GAT | HGB_P | 4 | 0.767467 | 0.736975 | 0.049049 |
| TxPert_Exphormer | TxPert_GAT | HGB_RepairedPublic | 4 | 0.794104 | 0.756861 | 0.048725 |
| TxPert_Exphormer | TxPert_GAT | Magnitude | 4 | 0.768216 | 0.741977 | 0.048831 |
| TxPert_Exphormer | TxPert_GAT | Ridge_ManualPublic | 4 | 0.770362 | 0.757685 | 0.048506 |
| TxPert_Exphormer | TxPert_GAT | Ridge_P | 4 | 0.770774 | 0.746952 | 0.048797 |
| TxPert_Exphormer | TxPert_GAT | Ridge_RepairedPublic | 4 | 0.779894 | 0.759729 | 0.048511 |
| TxPert_GAT | TxPert_Exphormer | HGB_ManualPublic | 4 | 0.778664 | 0.741388 | 0.048700 |
| TxPert_GAT | TxPert_Exphormer | HGB_P | 4 | 0.770663 | 0.728580 | 0.048817 |
| TxPert_GAT | TxPert_Exphormer | HGB_RepairedPublic | 4 | 0.795157 | 0.751825 | 0.048474 |
| TxPert_GAT | TxPert_Exphormer | Magnitude | 4 | 0.758726 | 0.727634 | 0.048927 |
| TxPert_GAT | TxPert_Exphormer | Ridge_ManualPublic | 4 | 0.771476 | 0.747638 | 0.048492 |
| TxPert_GAT | TxPert_Exphormer | Ridge_P | 4 | 0.766612 | 0.734894 | 0.048867 |
| TxPert_GAT | TxPert_Exphormer | Ridge_RepairedPublic | 4 | 0.781331 | 0.751600 | 0.048490 |

## Paired perturbation-cluster bootstrap

| source_upstream | target_upstream | comparison | method_a | method_b | delta_utility20 | ci95_lower | ci95_upper | bootstrap_replicates |
|---|---|---|---|---|---|---|---|---|
| TxPert_Exphormer | TxPert_GAT | HGBRepaired_vs_Magnitude | HGB_RepairedPublic | Magnitude | 0.027936 | -0.008901 | 0.072414 | 5000 |
| TxPert_Exphormer | TxPert_GAT | HGBRepaired_vs_HGBP | HGB_RepairedPublic | HGB_P | 0.021692 | -0.005976 | 0.051920 | 5000 |
| TxPert_Exphormer | TxPert_GAT | HGBRepaired_vs_HGBManual | HGB_RepairedPublic | HGB_ManualPublic | 0.009253 | -0.010063 | 0.028848 | 5000 |
| TxPert_Exphormer | TxPert_GAT | RidgeRepaired_vs_Magnitude | Ridge_RepairedPublic | Magnitude | 0.007682 | -0.028658 | 0.051540 | 5000 |
| TxPert_Exphormer | TxPert_GAT | RidgeRepaired_vs_RidgeP | Ridge_RepairedPublic | Ridge_P | 0.000859 | -0.023958 | 0.027703 | 5000 |
| TxPert_Exphormer | TxPert_GAT | RidgeRepaired_vs_RidgeManual | Ridge_RepairedPublic | Ridge_ManualPublic | -0.000078 | -0.021608 | 0.020891 | 5000 |
| TxPert_GAT | TxPert_Exphormer | HGBRepaired_vs_Magnitude | HGB_RepairedPublic | Magnitude | 0.037366 | 0.002665 | 0.082318 | 5000 |
| TxPert_GAT | TxPert_Exphormer | HGBRepaired_vs_HGBP | HGB_RepairedPublic | HGB_P | 0.024618 | 0.002213 | 0.050642 | 5000 |
| TxPert_GAT | TxPert_Exphormer | HGBRepaired_vs_HGBManual | HGB_RepairedPublic | HGB_ManualPublic | 0.013328 | -0.002679 | 0.031326 | 5000 |
| TxPert_GAT | TxPert_Exphormer | RidgeRepaired_vs_Magnitude | Ridge_RepairedPublic | Magnitude | 0.020363 | -0.014849 | 0.063347 | 5000 |
| TxPert_GAT | TxPert_Exphormer | RidgeRepaired_vs_RidgeP | Ridge_RepairedPublic | Ridge_P | 0.011024 | -0.014319 | 0.038156 | 5000 |
| TxPert_GAT | TxPert_Exphormer | RidgeRepaired_vs_RidgeManual | Ridge_RepairedPublic | Ridge_ManualPublic | 0.004994 | -0.012694 | 0.023201 | 5000 |
