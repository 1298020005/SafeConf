# 7 主表 signal validity audit（信号有效性审计）

一句话：这次专门检查 SafeConf 的 confidence score（可信度分数）是不是只是在偷看 effect magnitude（效应大小）。

## 这次做了什么

输入不是旧 v6 结果，而是正式主表：

```text
code/20260426_154505_perturb_transport_final_push/
outputs/safeconf_formal_main_20260604/formal_audit/tables/FORMAL_SCORED_RECORDS.csv
```

也就是说，这次审计对应当前 7 个正式主表数据集。

## 主要输出

```text
tables/SIGNAL_VALIDITY_7MAIN_MAIN_SCORE.csv
tables/SIGNAL_VALIDITY_7MAIN_SUMMARY.csv
tables/PARTIAL_AND_WITHIN_STRATUM_7MAIN.csv
tables/MAGNITUDE_BASELINE_7MAIN.csv
reports/SIGNAL_VALIDITY_7MAIN_REPORT.md
```

## 大白话结论

- magnitude-only（只看效应大小）确实很强，说明这个问题里有明显“幅度混杂”。
- 但 6/7 个数据集在 partial ρ（控制效应大小后的相关）上仍然为正。
- gene_main（基因主线）4/4 为正。
- chem_robust（化学线）里 Srivatsan 和 Santinha 为正，McFarland 仍然失败。
- McFarland 继续作为 failure boundary（失败边界），不改公式硬救。

## 主分数结果

| 数据集 | raw ρ | magnitude-only ρ | partial ρ | within-perturbation ρ | within-context ρ |
|---|---:|---:|---:|---:|---:|
| CuiHacohen2023 | 0.445 | 0.736 | 0.328 | 0.380 | 0.247 |
| Frangieh | 0.583 | 0.797 | 0.474 | 0.634 | 0.597 |
| Lara exvivo | 0.561 | 0.513 | 0.430 | 0.514 | 0.253 |
| Lara invivo | 0.413 | 0.639 | 0.358 | 0.226 | 0.404 |
| McFarland | -0.086 | 0.795 | -0.061 | 0.057 | -0.167 |
| Santinha | 0.206 | 0.824 | 0.224 | 0.019 | 0.374 |
| Srivatsan sciplex3 | 0.428 | 0.740 | 0.629 | 0.074 | 0.538 |

## 不能夸大的地方

- 不能说 SafeConf 完全摆脱 effect magnitude（效应大小）影响。
- 不能说 chemical perturbation（化学扰动）全面稳定，因为 McFarland 明确失败。
- 不能把旧 v6 signal validity 当成 7 主表正式证据；这次才是对应 7 主表的审计。

