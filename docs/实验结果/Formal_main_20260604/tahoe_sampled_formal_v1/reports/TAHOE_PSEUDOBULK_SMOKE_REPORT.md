# Tahoe pseudobulk adapter smoke report

## 结论

当前状态：`PASS_ADAPTER_SMOKE`。

这不是论文正式结果，只是检查 Tahoe pseudobulk 能否转成 SafeConf PredictionRecord。

## 输入

- Tahoe root: `/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M`
- pseudobulk shards available: 1026
- shards scanned: 100
- skipped/corrupt shards: 1
- selected tasks: 3000
- selected genes: 1000

## Smoke 输出

- PredictionRecords: 20742
- Predictor names: ['V0DrugMeanAcrossDose', 'V0ExactDoseMean']
- test pair leakage rows: 0
- missing context support in test: 0
- missing perturbation support in test: 0
- concentration leakage rows: 2506
- test plate seen in train ratio: 1.0
- true_error_rmse CV: 0.22845528294960765
- mean RMSE: 0.8841272892153933
- median RMSE: 0.8441064442056243
- formal eval: True
- aligned rho: 0.332837160050067
- partial rho controlling effect magnitude: 0.29308876320689525
- RC@80 improvement pct: 5.019655832814262
- held-out drug split note: V0-family Tahoe predictors require same-drug support; held-out-drug split is reported as feasibility audit, not as the main score.

## 怎么理解

如果 `PredictionRecords > 0` 且 leakage 为 0，说明 Tahoe 可以进入下一步 adapter/formal external validation。

如果这里失败，不代表 SafeConf 失败，只说明 Tahoe 的 pseudobulk 结构暂时不能直接进入当前协议。
