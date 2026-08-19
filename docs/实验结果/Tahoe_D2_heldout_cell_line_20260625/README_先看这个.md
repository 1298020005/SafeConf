# Tahoe D2 held-out-cell-line sensitivity

日期：2026-06-25
结论：`PARTIAL`（统计信号强，但只覆盖 supported tasks）

## 一句话

把 Tahoe 的测试集改成“整条 cell line 留出”后，SafeConf chemical risk 在可评估任务内仍然很强：

```text
partial rho = 0.470
task-cluster 95% CI = [0.457, 0.482]
```

但不是所有 held-out-cell-line 任务都能评估。9000 个原始测试任务里，8253 个有训练侧 perturbation 支持：

```text
applicability coverage = 91.7%
```

所以这不是完整的新细胞系覆盖结论，而是：

```text
supported held-out-cell-line tasks 上的 sensitivity evidence
```

## 数据规模

| 项目 | 数值 |
|---|---:|
| split mode | heldout_context |
| raw test tasks | 9000 |
| evaluated test task clusters | 8253 |
| applicability coverage | 91.7% |
| contexts | 25 |
| drug-dose perturbations | 1028 |
| prediction records | 80988 |

## 正式结果

| 层级 | aligned rho | partial rho | partial 95% CI | RC@80 |
|---|---:|---:|---:|---:|
| overall | 0.405 | **0.470** | **[0.457, 0.482]** | 4.40% |
| V0DrugMeanAcrossDose | 0.351 | 0.321 | [0.302, 0.336] | 4.38% |
| V0ExactDoseMean | 0.479 | 0.611 | [0.596, 0.625] | 4.40% |

## Leakage / support audit

| 检查 | 结果 |
|---|---:|
| test context seen in train | 0 |
| test pair leakage | 0 |
| raw test perturbation missing rows | 147 |
| evaluated task coverage | 91.7% |

## 允许写

```text
In a held-out-cell-line Tahoe sensitivity analysis, SafeConf risk remained
associated with error within the supported applicability domain.
```

## 不允许写

```text
all held-out-cell-line tasks are covered
full new-cell-line generalization
held-out-drug generalization
```

## 文件

| 文件 | 用途 |
|---|---|
| `RUN_STATUS.json` | 主运行规模、split、点估计 |
| `TAHOE_D2_FINAL_STATUS.json` | 最终 gate 与 applicability coverage |
| `tables/TAHOE_D1_FORMAL_SUMMARY.csv` | task-cluster CI |
| `tables/TAHOE_D2_APPLICABILITY_COVERAGE.csv` | 每个 held-out context 的覆盖率 |

大型 arrays、完整 records 和 bootstrap draws 保存在：

```text
/home/yyf/safeconf_runtime/outputs/tahoe_d2_heldout_cell_line_20260625
/home/yyf/safeconf_runtime/outputs/tahoe_d2_heldout_cell_line_taskcluster_20260625
```
