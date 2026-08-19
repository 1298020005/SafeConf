# Existing GEARS Supplement Status

## 结论

已有 GEARS 输出可以作为 supplement（补充），但不能替代 7 主表证据，也不能冒充 Cui 上的第三 predictor（预测器）。

## 当前已有

- GEARS run status file exists: True
- GEARS eval file exists: True
- GEARS uncertainty proxy file exists: True
- native uncertainty present: False
- total GEARS run rows: 9
- total GEARS eval rows: 5
- total GEARS uncertainty/proxy rows: 12

## Dataset-level GEARS confidence signals

- adamson: `gears_prediction_magnitude_risk` aligned rho = 0.422, n = 21
- dixit: `gears_prediction_magnitude_risk` aligned rho = 0.500, n = 3
- norman: `gears_prediction_magnitude_risk` aligned rho = 0.624, n = 30

## 解释

这些 GEARS 结果来自 Norman / Adamson / Dixit 的 gene perturbation（基因扰动）场景。它们说明 SafeConf 可以读取 GEARS per-prediction records（逐条预测记录），但样本量小、split（切分）不是当前 7 主表的 cross-context task（跨背景任务），且 native uncertainty（原生不确定性）仍未导出。

建议论文写成：

> GEARS supplement demonstrates adapter compatibility, while the main claims remain based on the seven-dataset cross-context benchmark.
