# E202 控制扰动幅度后的模型失败诊断

- 主结论：**NOT SUPPORTED**。
- 主 partial Spearman：`-0.0680` （95% bootstrap 区间 `-0.1522` 至 `0.0191`）。
- 幅度五分位内中心化描述性 Spearman：`-0.0911`。

## 主结局上的风险成分

| predictor | raw_spearman | partial_spearman | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- |
| training_delta_dispersion | 0.0798 | -0.0680 | -0.1522 | 0.0191 |
| transfer_risk | -0.5473 | -0.6505 | -0.7097 | -0.5844 |
| model_baseline_gap | -0.5889 | -0.6032 | -0.6707 | -0.5293 |
| negative_log_train_cells | -0.6718 | -0.7136 | -0.7614 | -0.6611 |
| support_context_deficit | -0.5284 | -0.5260 | -0.5931 | -0.4514 |

## 五个评价端点

| endpoint | raw_spearman | partial_spearman | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- |
| mse | -0.1399 | -0.1130 | -0.1970 | -0.0243 |
| pearson_pert | 0.1275 | 0.0824 | -0.0023 | 0.1631 |
| rank | 0.1530 | -0.0079 | -0.0952 | 0.0767 |
| energy_distance_pca_k=50 | 0.0315 | -0.0255 | -0.1179 | 0.0660 |
| de_auprc | 0.2496 | 0.1317 | 0.0434 | 0.2174 |

## 解释边界

E202 是 E200 之后登记的 K562 诊断。主门槛只判断训练背景分歧在控制预测幅度后，是否仍与 GAT 相对 general baseline 的额外 RMSE 正相关。它不构成其他细胞系、模型家族或数据集上的确认性证据。
