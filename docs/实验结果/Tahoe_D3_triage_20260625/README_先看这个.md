# Tahoe D3 prediction-triage audit

日期：2026-06-25
结论：`PASS_MAGNITUDE_STRONGER`

## 一句话

在 Tahoe chemical test records 上，SafeConf 可以明显优于随机地抓出高误差预测：

```text
SafeConf top-10 enrichment = 4.88x
random top-10 enrichment = 1.05x
```

但 chemical Tahoe 里，预测幅度 baseline 更强：

```text
predicted-magnitude top-10 enrichment = 6.49x
SafeConf - magnitude 95% CI = [-1.93, -1.38]
```

所以这部分不能写成 SafeConf 击败 magnitude。正确用法是：

```text
SafeConf has practical triage value on Tahoe, while magnitude remains a strong
chemical baseline and should be reported side by side.
```

## Top-k enrichment

| score | top 5% | top 10% | top 20% |
|---|---:|---:|---:|
| SafeConf full | 9.48x | **4.88x** | 2.42x |
| predicted magnitude | 14.49x | **6.49x** | 2.93x |
| disagreement only | 11.76x | 5.37x | 2.62x |
| support only | 1.84x | 2.01x | 1.75x |
| random | 0.94x | 1.05x | 1.00x |
| oracle error | 19.99x | 10.00x | 5.00x |

## 关键解释

| 问题 | 结论 |
|---|---|
| SafeConf 有没有实际分诊价值？ | 有，top 10% 风险能抓到 4.88x 高误差预测 |
| SafeConf 是否赢过 magnitude？ | 没有，Tahoe chemical 上 magnitude 更强 |
| 哪个特征最像主信号？ | disagreement-only 很强，support-only 较弱 |
| 论文里怎么放？ | 可作 chemical practical-value supplement，必须带 magnitude caveat |

## 允许写

```text
Tahoe confirms that SafeConf can prioritize high-error chemical predictions,
but also shows that magnitude is a strong chemical baseline.
```

## 不允许写

```text
SafeConf outperforms magnitude on Tahoe
SafeConf is the best chemical triage score
```

## 文件

| 文件 | 用途 |
|---|---|
| `TAHOE_D3_STATUS.json` | gate 和核心数字 |
| `tables/TAHOE_D3_TRIAGE_POINT_SUMMARY.csv` | top 5/10/20 enrichment |
| `tables/TAHOE_D3_TOP10_TASK_CLUSTER_CI.csv` | task-cluster bootstrap CI |
| `reports/TAHOE_D3_TRIAGE_REPORT.md` | 简短报告 |

大型中间表保存在：

```text
/home/yyf/safeconf_runtime/outputs/tahoe_d3_triage_20260625
```
