# E200 预测封存

TxPert 官方 K562 cross-cell GAT 和 general baseline 均完成全量预测。两套结果各含 150,472 个 K562 扰动细胞、3,352 个基因、1,087 个官方测试任务和 48 个批次。

行、基因、任务和批次顺序一致；两套真值矩阵逐元素相同。两套 control 的最大差为 `9.5367431640625e-07`，504,382,144 个值中无一超过 `1e-6`，按 `atol=1e-6, rtol=0` 判定为数值等价。

原始结果位于 `DATA/txpert_official_20260802/e200/predictions/`，不进入 Git。文件字节数与 SHA-256 见 `../tables/E200_PREDICTION_HASHES.csv`。截至本封存点，尚未计算 GAT 误差、评价端点、风险相关性或路由效用。
