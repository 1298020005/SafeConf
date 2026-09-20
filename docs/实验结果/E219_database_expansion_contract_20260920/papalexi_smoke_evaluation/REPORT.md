# E219 Papalexi 三技术重复冒烟结果

- 测试行：111；不同背景—扰动对：72。
- 两预测器平均 RMSE：0.050454；no-change：0.050576。
- 整体上游能力门：PASS。
- replicate 是技术重复，本实验不写成跨生物细胞背景验证。

## 各任务类型的上游能力

| setting                         |   n_tasks |   mean_model_error |   mean_no_change_error |   mean_absolute_gain |   relative_error_reduction | upstream_gate   |
|:--------------------------------|----------:|-------------------:|-----------------------:|---------------------:|---------------------------:|:----------------|
| context_and_perturbation_unseen |        15 |          0.0553956 |              0.0494313 |          -0.00596425 |                 -0.120657  | NOT_SUPPORTED   |
| context_unseen                  |        57 |          0.0450102 |              0.0502687 |           0.00525855 |                  0.104609  | PASS            |
| perturbation_unseen             |        30 |          0.0571157 |              0.0492918 |          -0.00782391 |                 -0.158726  | NOT_SUPPORTED   |
| random_pair                     |         9 |          0.0544945 |              0.0587149 |           0.00422038 |                  0.0718792 | PASS            |

## 三折宏平均风险结果

| setting                         | score                     |   n_folds |   mean_spearman |   mean_top20_error_enrichment |
|:--------------------------------|:--------------------------|----------:|----------------:|------------------------------:|
| context_and_perturbation_unseen | e218_transfer_router      |         3 |        0.157361 |                     0.129235  |
| context_and_perturbation_unseen | model_disagreement        |         3 |       -0.166667 |                     0.0432979 |
| context_and_perturbation_unseen | predicted_magnitude       |         3 |        0.6      |                     0.129235  |
| context_and_perturbation_unseen | safeconf_structural_proxy |         3 |       -0.166667 |                     0.0432979 |
| context_unseen                  | e218_transfer_router      |         3 |        0.889474 |                     0.443837  |
| context_unseen                  | model_disagreement        |         3 |        0.845029 |                     0.461163  |
| context_unseen                  | predicted_magnitude       |         3 |        0.914035 |                     0.443837  |
| context_unseen                  | safeconf_structural_proxy |         3 |        0.701261 |                     0.335775  |
| perturbation_unseen             | e218_transfer_router      |         3 |        0.684848 |                     0.227937  |
| perturbation_unseen             | model_disagreement        |         3 |        0.313131 |                     0.126995  |
| perturbation_unseen             | predicted_magnitude       |         3 |        0.705051 |                     0.227937  |
| perturbation_unseen             | safeconf_structural_proxy |         3 |        0.313131 |                     0.126995  |
| random_pair                     | e218_transfer_router      |         3 |        0.5      |                     0.172138  |
| random_pair                     | model_disagreement        |         3 |        0.333333 |                     0.172138  |
| random_pair                     | predicted_magnitude       |         3 |        0.5      |                     0.172138  |
| random_pair                     | safeconf_structural_proxy |         3 |        0.666667 |                     0.172138  |
