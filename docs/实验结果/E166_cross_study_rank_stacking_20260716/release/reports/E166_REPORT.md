# E166｜跨研究折内秩组合结果

## 主结果

八研究等权平均 `Δrho(stack−magnitude)=0.0324`，两层 bootstrap 95% CI `[-0.0283, 0.0877]`；点估计为正的研究为 `5/8`。预设严格 gate：`FAIL`。

| dataset | rho_rank_stack_lodo | rho_magnitude | delta_stack_minus_magnitude | delta_vs_magnitude_ci95_low | delta_vs_magnitude_ci95_high |
|---|---|---|---|---|---|
| Frangieh | 0.2352 | 0.1481 | 0.0870 | 0.0531 | 0.1216 |
| Lara_exvivo | 0.2778 | 0.1475 | 0.1303 | 0.0508 | 0.2030 |
| Liang | 0.1625 | 0.0742 | 0.0883 | -0.0006 | 0.1836 |
| Nadig_two_cellline | 0.3214 | 0.4026 | -0.0811 | -0.1791 | 0.0201 |
| Replogle_two_cellline | 0.1556 | 0.2143 | -0.0588 | -0.1709 | 0.0502 |
| Santinha | -0.0680 | -0.0893 | 0.0213 | -0.1048 | 0.1167 |
| Shifrut | 0.2073 | 0.2087 | -0.0014 | -0.1168 | 0.1464 |
| Tian_CRISPRi | 0.1401 | 0.0667 | 0.0734 | 0.0196 | 0.1309 |

## 每个留出研究对应的训练权重

| heldout_dataset | weight_magnitude | weight_disagreement | weight_safeconf |
|---|---|---|---|
| Frangieh | 0.3718 | 0.1490 | 0.4792 |
| Lara_exvivo | 0.3805 | 0.1498 | 0.4697 |
| Liang | 0.3756 | 0.1449 | 0.4794 |
| Nadig_two_cellline | 0.3284 | 0.1422 | 0.5294 |
| Replogle_two_cellline | 0.3304 | 0.1659 | 0.5037 |
| Santinha | 0.3777 | 0.1437 | 0.4787 |
| Shifrut | 0.3656 | 0.1013 | 0.5331 |
| Tian_CRISPRi | 0.3754 | 0.1021 | 0.5224 |

## 审计与边界

每一轮权重只由另外七个研究的折内风险秩和误差秩拟合；留出研究真值用于评价的行数不进入权重拟合。输入为 E153 已公开任务快照，因此 E166 是 post-hoc 方法开发和跨研究交叉验证，不是新的独立确认。折内秩分数适合一批任务的相对质检，不能直接解释为单任务绝对失败概率。
