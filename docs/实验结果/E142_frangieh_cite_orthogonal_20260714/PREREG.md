# E142 预注册｜Frangieh CITE-seq 封存式跨模态验证

E108 风险分数、scGPT/GEARS RNA 预测和切分已经完成；冻结本合同后才打开同批细胞的蛋白矩阵。

## 固定处理

- 采用原论文的 isotype normalization：`max(ln((antibody+1)/(matched IgG+1)), 0)`。
- 4 个 IgG isotype 只作归一化/负控，20 个生物表面蛋白构成评价轴。
- 同一 context 内，protein effect = perturbation mean − control mean；扰动标签去掉 E108 的 `+ctrl` 后匹配。
- 每个外层 fold 仅用 train RNA 真值与 train 蛋白效应拟合 decoder；alpha 在 val 蛋白上从固定网格选择，再用 train+val 重拟合。
- decoder 分别接收 scGPT 与 GEARS 的测试 RNA 预测，目标蛋白从不进入 SafeConf 评分。

## 主终点与 gate

主终点是两预测器平均 protein RMSE 与 protein cosine error。每折算 Spearman 后等权平均，按 perturbation 整簇 bootstrap 3,000 次。Gate 要求两个相关方向均为正、至少一个 95% CI 下界大于 0，且 true-RNA decoder 的 protein RMSE 优于 train-mean baseline。所有 predicted magnitude、disagreement、isotype 负控和失败 setting 原样保留。
