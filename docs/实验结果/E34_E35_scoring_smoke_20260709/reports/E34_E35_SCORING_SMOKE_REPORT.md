# E34/E35 scoring smoke

生成时间：2026-07-09 20:23

## 已触发

- 数据任务构建成功：7/8
- split scoring 成功：98/121

## 怎么理解

这不是 formal 结果，但已经完成了周老师要求的下一步实验链第一跳：在小矩阵、整行、整列 split 上，用两个参考预测器产生真实 error，再看 SafeConf smoke risk、disagreement、magnitude、support、context similarity 是否能排序错误。

## 当前最该看

- `tables/E34_E35_SCORING_SUMMARY.csv`
- `tables/E34_E35_SCORE_TABLE.csv`
- `tables/E34_E35_SPLIT_SCORE_STATUS.csv`

## Spearman 最高的前几项

dataset_name         setting  coverage_target     risk_score_name          target_error  n_tasks  spearman  top20_k  top20_mean_error  top20_enrichment
      Parekh E35_row_holdout              NaN risk_safeconf_smoke error_contextsim_rmse       30  0.791769        6          0.098783          1.579867
      Parekh E35_row_holdout              NaN risk_safeconf_smoke       error_mean_rmse       30  0.787319        6          0.098866          1.573176
      Parekh E35_row_holdout              NaN risk_safeconf_smoke         error_v0_rmse       30  0.781980        6          0.098949          1.566553
    sciplex3 E35_row_holdout              NaN risk_safeconf_smoke         error_v0_rmse      105  0.769315       21          0.217975          1.393287
    sciplex3 E35_row_holdout              NaN risk_safeconf_smoke       error_mean_rmse      105  0.760917       21          0.213296          1.383585
    sciplex3 E35_row_holdout              NaN risk_safeconf_smoke error_contextsim_rmse      105  0.750861       21          0.208616          1.373591
    sciplex3   E34_submatrix             0.75 risk_safeconf_smoke error_contextsim_rmse       35  0.723879        7          0.198919          1.668114
    sciplex3   E34_submatrix             0.75 risk_safeconf_smoke       error_mean_rmse       35  0.718384        7          0.196765          1.672837
