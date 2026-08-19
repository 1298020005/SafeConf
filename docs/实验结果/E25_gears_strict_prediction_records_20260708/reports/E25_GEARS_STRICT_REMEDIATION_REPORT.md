# E25 GEARS strict PredictionRecord remediation

生成时间：2026-07-08T02:48:34

## 结论

E25 将 `/home/yyf/safeconf_runtime/outputs/gears_prediction_records_formal/` 中的真实 GEARS 输出升级为严格 SafeConf PredictionRecord 合同。升级后，合并包通过 `validate_prediction_record_artifacts(strict=True)`。

- 严格校验状态：PASS
- 数据集：adamson, dixit, norman
- 预测记录：54
- 来源：真实 GEARS formal runs，不重新训练，不重写误差。

## 数据集汇总

| dataset_name | n_runs | n_prediction_records | n_genes | mean_rmse | median_rmse | min_rmse | max_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adamson | 3 | 21 | 5043 | 0.0420858 | 0.0347614 | 0.0229923 | 0.112453 |
| dixit | 3 | 3 | 6000 | 0.424841 | 0.371982 | 0.355655 | 0.546885 |
| norman | 3 | 30 | 5025 | 0.0558785 | 0.0455374 | 0.0300964 | 0.0891275 |

## 每个 run 汇总

| dataset_name | seed | source_run_dir | n_prediction_records | n_predicted_arrays | n_true_arrays | n_genes | gene_panel_id | gene_order_hash | mean_rmse | median_rmse | min_rmse | max_rmse | mean_gears_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adamson | 1 | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal/adamson/seed_1 | 7 | 7 | 7 | 5043 | gears::adamson::single::n_genes_5043 | sha256:8b461482a501bc0fa4966cf0d6ba0a20365560359528edf44ab7876dc831278d | 0.0387541 | 0.0301224 | 0.0229923 | 0.0725773 | nan |
| adamson | 2 | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal/adamson/seed_2 | 7 | 7 | 7 | 5043 | gears::adamson::single::n_genes_5043 | sha256:8b461482a501bc0fa4966cf0d6ba0a20365560359528edf44ab7876dc831278d | 0.0406253 | 0.0448727 | 0.0230502 | 0.0548689 | nan |
| adamson | 3 | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal/adamson/seed_3 | 7 | 7 | 7 | 5043 | gears::adamson::single::n_genes_5043 | sha256:8b461482a501bc0fa4966cf0d6ba0a20365560359528edf44ab7876dc831278d | 0.046878 | 0.0347614 | 0.0248091 | 0.112453 | nan |
| dixit | 1 | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal/dixit/seed_1 | 1 | 1 | 1 | 6000 | gears::dixit::single::n_genes_6000 | sha256:09cfde75cdbdbdc2f6f496e5fa2cdfb4b9dfd758f582c2212777c2dd525b66e3 | 0.546885 | 0.546885 | 0.546885 | 0.546885 | nan |
| dixit | 2 | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal/dixit/seed_2 | 1 | 1 | 1 | 6000 | gears::dixit::single::n_genes_6000 | sha256:09cfde75cdbdbdc2f6f496e5fa2cdfb4b9dfd758f582c2212777c2dd525b66e3 | 0.355655 | 0.355655 | 0.355655 | 0.355655 | nan |
| dixit | 3 | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal/dixit/seed_3 | 1 | 1 | 1 | 6000 | gears::dixit::single::n_genes_6000 | sha256:09cfde75cdbdbdc2f6f496e5fa2cdfb4b9dfd758f582c2212777c2dd525b66e3 | 0.371982 | 0.371982 | 0.371982 | 0.371982 | nan |
| norman | 1 | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal/norman/seed_1 | 10 | 10 | 10 | 5025 | gears::norman::single::n_genes_5025 | sha256:9059bd6e8ff28482b90f1dd4dd423df7e1990b35f9a8ad5b4b6aee9ef0eb7ed0 | 0.0706898 | 0.0678329 | 0.0490057 | 0.0891275 | nan |
| norman | 2 | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal/norman/seed_2 | 10 | 10 | 10 | 5025 | gears::norman::single::n_genes_5025 | sha256:9059bd6e8ff28482b90f1dd4dd423df7e1990b35f9a8ad5b4b6aee9ef0eb7ed0 | 0.0512447 | 0.0455374 | 0.0300964 | 0.0774888 | nan |
| norman | 3 | $SAFECONF_RUNTIME/outputs/gears_prediction_records_formal/norman/seed_3 | 10 | 10 | 10 | 5025 | gears::norman::single::n_genes_5025 | sha256:9059bd6e8ff28482b90f1dd4dd423df7e1990b35f9a8ad5b4b6aee9ef0eb7ed0 | 0.0457009 | 0.0421205 | 0.0354888 | 0.0668361 | nan |

## 严格校验

| scope | strict | issue_count | issues |
| --- | --- | --- | --- |
| combined_e25_gears_strict_package | True | 0 |  |

## 论文意义

这一步把 GEARS 从“存在旧结果”推进到“可审计、可复现、可合并进入 SafeConf 协议”的状态。它不能替代更大规模外部验证，但它解决了真实模型基线进入主证据链前最容易被审稿人质疑的合同和 provenance 问题。
