# SafeConf 7 主表正式审计结果

更新时间：2026-06-04

## 一句话结论

7 个主表数据集都跑通了。  
按冻结 `protocol_v0_2_family_confidence（v0.2 规则可信度分数）` 看：

- aligned rho（方向对齐相关）过 0.20：6/7
- partial rho（控制 effect magnitude/效应大小后的相关）过 0.10：6/7
- risk-coverage@80%（只保留高可信 80% 后误差下降）：7/7 都是正的
- 失败边界：McFarlandTsherniak2020 drug-only

这说明 SafeConf 不是只在小数据上有信号，已经有正式主表基础。  
但还不能说“一区稳了”，因为 McFarland 这个大药物数据集是负结果，而且所有数据集都有很强的 effect magnitude confounding（效应大小混杂）。

## 主表结果

| 数据集 | 线 | n test | aligned rho | partial rho | magnitude-only rho | RC@80% |
|---|---|---:|---:|---:|---:|---:|
| CuiHacohen2023 | gene_main | 2506 | 0.445 | 0.328 | 0.736 | 21.59% |
| Frangieh | gene_main | 1266 | 0.583 | 0.474 | 0.797 | 5.03% |
| Lara exvivo | gene_main | 662 | 0.561 | 0.430 | 0.513 | 56.12% |
| Lara invivo | gene_main | 780 | 0.413 | 0.358 | 0.639 | 12.83% |
| McFarland drug-only | chem_robust | 2326 | -0.086 | -0.061 | 0.795 | 3.95% |
| SantinhaPlatt2023 | chem_robust | 566 | 0.206 | 0.224 | 0.824 | 2.08% |
| Srivatsan sciplex3 | chem_robust | 1128 | 0.428 | 0.629 | 0.740 | 15.10% |

## 怎么理解

- `aligned rho（方向对齐相关）`：可信度分数越高，真实误差越低，算好信号。
- `partial rho（控制效应大小后的相关）`：排除“只是效应本来就大/小”的假象后，分数还有没有用。
- `magnitude-only rho（只看效应大小）`：如果这个很高，说明误差很容易被 effect magnitude（效应大小）影响。
- `RC@80%`：只保留分数认为比较可信的 80% 预测，看平均误差有没有下降。

## 现在能说什么

- 可以说：7 个主表数据集已经完成 formal audit（正式审计）。
- 可以说：6/7 数据集在 partial rho 上仍有正信号，说明不是完全靠效应大小。
- 可以说：McFarland 是当前最重要失败边界，需要单独诊断。

## 现在不能说什么

- 不能说：SafeConf 已经一区稳了。
- 不能说：所有药物数据都成功。
- 不能只报 aligned rho，不报 magnitude-only 和 partial rho。
- 不能把 learned_risk_score 当主方法标题；主表仍以冻结 protocol v0.2 为主。

## 关键文件

- 主表：`tables/FORMAL_MAIN_TABLE.csv`
- 每折结果：`tables/FORMAL_PER_FOLD_RHO.csv`
- 每 predictor（预测器）结果：`tables/FORMAL_PER_PREDICTOR_RHO.csv`
- 输入状态：`tables/FORMAL_INPUT_STATUS.csv`
- 完整报告：`FORMAL_MAIN_AUDIT_REPORT.md`

## 2026-06-06 新增：7 主表信号有效性审计

新增目录：

```text
signal_validity_7main_20260606/
```

这次补的是正式 7 主表版本，不再把旧 v6 signal validity 当主表证据。

重点输出：

- `tables/SIGNAL_VALIDITY_7MAIN_MAIN_SCORE.csv`
- `tables/PARTIAL_AND_WITHIN_STRATUM_7MAIN.csv`
- `tables/MAGNITUDE_BASELINE_7MAIN.csv`
- `reports/SIGNAL_VALIDITY_7MAIN_REPORT.md`

一句话结论：effect magnitude（效应大小）混杂很强，但 6/7 数据集控制效应大小后仍有正信号；McFarland 继续作为失败边界。
