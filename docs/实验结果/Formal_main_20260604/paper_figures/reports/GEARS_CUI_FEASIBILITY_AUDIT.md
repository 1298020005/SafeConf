# GEARS on Cui Feasibility Audit

## 结论

不建议在 CuiHacohen2023 上硬跑 GEARS。

原因很简单：GEARS（图神经网络扰动预测模型）主要面向 gene perturbation（基因扰动），而 CuiHacohen2023 的 perturbation_type 是 `cytokines`，也就是 cytokine（细胞因子/刺激物）处理。

## 证据

- 非 control perturbations: 86
- 与 var_names 精确匹配的基因名: 1
- 拆分后与 var_names 匹配的基因名: 1
- exact overlap examples: Flt3l

如果把这些 cytokine 名称强行当作 GEARS 的 gene perturbation，会变成伪实验。

## 建议

1. 不要把“GEARS on Cui”写进执行计划。
2. 如果要证明 predictor-agnostic（不绑定预测器），优先用已有 Norman/Adamson/Dixit 的 GEARS supplement。
3. 如果一定要在 Cui 这类 cytokine 数据上接第三 predictor，应考虑 cytokine/drug response 兼容模型，而不是 GEARS。
