# E142｜Frangieh RNA→CITE-seq 封存式跨模态验证

## 预注册 gate：未通过

同批 **218,331** 个细胞、20 个生物表面蛋白、837 个正式测试任务。蛋白矩阵在风险分数和分析合同冻结后才打开。

| endpoint | SafeConf fold-macro ρ | 95% CI | Δ vs magnitude (95% CI) | Δ vs disagreement (95% CI) |
|---|---:|---:|---:|---:|
| protein RMSE | 0.220 | [0.162, 0.278] | +0.008 [-0.056, +0.077] | +0.021 [-0.043, +0.089] |
| protein cosine error | 0.146 | [0.093, 0.197] | +0.137 [+0.063, +0.209] | +0.140 [+0.069, +0.211] |

## RNA→protein decoder 可用性

输入真实 RNA 效应时，decoder 相对训练蛋白均值基线的平均 RMSE 优势为 **-0.0112**（正值表示 decoder 更好）。这一步检查蛋白误差是否只是不可预测的翻译噪声。

## 生物标志物

CD117, CD119, CD140a, CD140b, CD172a, CD184, CD202b, CD274, CD29, CD309, CD44, CD47, CD49f, CD58, CD59, CD61, HLA_A, HLA_E, CD9, CD279

## 边界

这是同一公开实验中的正交蛋白读出，不是新采集湿实验。RNA→protein decoder 只在外层 train/val 任务拟合，但蛋白丰度仍受翻译后调控和抗体噪声影响；失败 setting 和更强基线必须逐项保留。
