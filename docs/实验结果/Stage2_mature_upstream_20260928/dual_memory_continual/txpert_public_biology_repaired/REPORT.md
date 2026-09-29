# Dual-Memory Public Biology and Shared Risk

This is a DEV/SEEN experiment. McFaline test truth remained sealed.

## Biological reconstruction

| method | n_tasks | effect_rmse | effect_cosine | effective_sources |
|---|---|---|---|---|
| CellWeighted | 1808 | 0.063678 | 0.525004 | 2.218892 |
| LearnedHGB | 1808 | 0.062924 | 0.509135 | 1.666901 |
| LearnedHGBRegularized | 1808 | 0.062113 | 0.529424 | 2.118432 |
| LearnedRidge | 1808 | 0.063318 | 0.505370 | 1.668822 |
| NearestControl | 1808 | 0.067951 | 0.468131 | 1.000000 |
| Uniform | 1808 | 0.064002 | 0.527729 | 2.620022 |

## Shared risk

| upstream | method | n_strata | utility20 | spearman | aurc |
|---|---|---|---|---|---|
| TxPert_Exphormer | Magnitude | 4 | 0.758726 | 0.727634 | 0.048927 |
| TxPert_Exphormer | SharedHGB_Learned | 4 | 0.790606 | 0.752125 | 0.048462 |
| TxPert_Exphormer | SharedHGB_Manual | 4 | 0.789953 | 0.745927 | 0.048616 |
| TxPert_Exphormer | SharedHGB_P | 4 | 0.770472 | 0.732703 | 0.048800 |
| TxPert_Exphormer | SharedHGB_Regularized | 4 | 0.803162 | 0.754013 | 0.048442 |
| TxPert_Exphormer | SharedRidge_Learned | 4 | 0.777017 | 0.750274 | 0.048487 |
| TxPert_Exphormer | SharedRidge_Manual | 4 | 0.773135 | 0.751676 | 0.048431 |
| TxPert_Exphormer | SharedRidge_P | 4 | 0.768024 | 0.735198 | 0.048844 |
| TxPert_Exphormer | SharedRidge_Regularized | 4 | 0.781591 | 0.753386 | 0.048425 |
| TxPert_GAT | Magnitude | 4 | 0.768216 | 0.741977 | 0.048831 |
| TxPert_GAT | SharedHGB_Learned | 4 | 0.789658 | 0.758659 | 0.048605 |
| TxPert_GAT | SharedHGB_Manual | 4 | 0.788090 | 0.752236 | 0.048766 |
| TxPert_GAT | SharedHGB_P | 4 | 0.774242 | 0.739784 | 0.048906 |
| TxPert_GAT | SharedHGB_Regularized | 4 | 0.797582 | 0.759108 | 0.048602 |
| TxPert_GAT | SharedRidge_Learned | 4 | 0.776776 | 0.758803 | 0.048530 |
| TxPert_GAT | SharedRidge_Manual | 4 | 0.772171 | 0.757861 | 0.048516 |
| TxPert_GAT | SharedRidge_P | 4 | 0.767661 | 0.746977 | 0.048798 |
| TxPert_GAT | SharedRidge_Regularized | 4 | 0.776754 | 0.760074 | 0.048510 |

## Cross-architecture transfer

| train_upstream | test_upstream | n_strata | utility20 | spearman | aurc |
|---|---|---|---|---|---|
| TxPert_Exphormer | TxPert_GAT | 4 | 0.782097 | 0.759082 | 0.048511 |
| TxPert_GAT | TxPert_Exphormer | 4 | 0.780884 | 0.749022 | 0.048531 |

## Paired gene-cluster bootstrap

| upstream | comparison | method_a | method_b | delta_utility20 | ci95_lower | ci95_upper | bootstrap_replicates |
|---|---|---|---|---|---|---|---|
| TxPert_Exphormer | RidgeLearned_vs_RidgeP | SharedRidge_Learned | SharedRidge_P | 0.008264 | -0.018320 | 0.035836 | 5000 |
| TxPert_Exphormer | RidgeLearned_vs_RidgeManual | SharedRidge_Learned | SharedRidge_Manual | 0.001674 | -0.020627 | 0.024827 | 5000 |
| TxPert_Exphormer | HGBLearned_vs_HGBP | SharedHGB_Learned | SharedHGB_P | 0.019922 | -0.004238 | 0.048571 | 5000 |
| TxPert_Exphormer | HGBLearned_vs_HGBManual | SharedHGB_Learned | SharedHGB_Manual | 0.003440 | -0.019398 | 0.027562 | 5000 |
| TxPert_Exphormer | HGBLearned_vs_Magnitude | SharedHGB_Learned | Magnitude | 0.031141 | -0.002324 | 0.073089 | 5000 |
| TxPert_Exphormer | HGBLearned_vs_RidgeLearned | SharedHGB_Learned | SharedRidge_Learned | 0.013471 | -0.009872 | 0.038065 | 5000 |
| TxPert_Exphormer | HGBRegularized_vs_HGBP | SharedHGB_Regularized | SharedHGB_P | 0.027023 | 0.001133 | 0.057539 | 5000 |
| TxPert_Exphormer | HGBRegularized_vs_HGBManual | SharedHGB_Regularized | SharedHGB_Manual | 0.010540 | -0.008699 | 0.030213 | 5000 |
| TxPert_Exphormer | HGBRegularized_vs_HGBLearned | SharedHGB_Regularized | SharedHGB_Learned | 0.007100 | -0.013746 | 0.026087 | 5000 |
| TxPert_Exphormer | HGBRegularized_vs_Magnitude | SharedHGB_Regularized | Magnitude | 0.038241 | 0.001255 | 0.082041 | 5000 |
| TxPert_GAT | RidgeLearned_vs_RidgeP | SharedRidge_Learned | SharedRidge_P | 0.004625 | -0.019938 | 0.031275 | 5000 |
| TxPert_GAT | RidgeLearned_vs_RidgeManual | SharedRidge_Learned | SharedRidge_Manual | 0.002188 | -0.019647 | 0.024070 | 5000 |
| TxPert_GAT | HGBLearned_vs_HGBP | SharedHGB_Learned | SharedHGB_P | 0.019623 | -0.005721 | 0.048665 | 5000 |
| TxPert_GAT | HGBLearned_vs_HGBManual | SharedHGB_Learned | SharedHGB_Manual | 0.002419 | -0.022383 | 0.026552 | 5000 |
| TxPert_GAT | HGBLearned_vs_Magnitude | SharedHGB_Learned | Magnitude | 0.025395 | -0.008173 | 0.065777 | 5000 |
| TxPert_GAT | HGBLearned_vs_RidgeLearned | SharedHGB_Learned | SharedRidge_Learned | 0.014582 | -0.007978 | 0.036790 | 5000 |
| TxPert_GAT | HGBRegularized_vs_HGBP | SharedHGB_Regularized | SharedHGB_P | 0.026945 | -0.000164 | 0.057005 | 5000 |
| TxPert_GAT | HGBRegularized_vs_HGBManual | SharedHGB_Regularized | SharedHGB_Manual | 0.009741 | -0.010455 | 0.032541 | 5000 |
| TxPert_GAT | HGBRegularized_vs_HGBLearned | SharedHGB_Regularized | SharedHGB_Learned | 0.007322 | -0.011108 | 0.026218 | 5000 |
| TxPert_GAT | HGBRegularized_vs_Magnitude | SharedHGB_Regularized | Magnitude | 0.032717 | -0.001109 | 0.072511 | 5000 |

## Integrity

- Public biological retrieval never used an upstream error as a label.
- Every prediction of the same biological task used the same gene-cluster outer fold.
- Error ranks were fitted only from each outer-train partition.
