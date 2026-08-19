# 给 Claude：E8b 最终只读复核

## 执行状态

Step 5-8 已执行完成。正式结果目录：

`docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/`

全套测试：

```text
71 passed
```

## 1. Frangieh primary

- 74 perturbations；
- 15 methods；
- metric = MSE，DEG=5000；
- median Spearman = 0.5836；
- perturbation-cluster bootstrap B=1000：
  95% CI [0.3935, 0.7261]；
- 14/15 methods 为正；
- gate = PASS。

Shuffled-risk null：

- 200 permutations；
- null median = 0.0072；
- 95% null range [-0.2323, 0.2305]；
- observed empirical one-sided p = 1/201。

## 2. 不能忽略的样本量问题

`sample_size_risk = -log1p(Nstimulated)`：

- median Spearman = 0.7637；
- 95% CI [0.5573, 0.9070]。

它比 frozen risk 的 0.5836 更强，因此不能写成“E8b 排除了 cell-count
confounding”。

为判断是否仍有独立信号，追加了一个明确标记为 post hoc、不改变预注册 gate
的 partial audit：

- frozen risk 与 sample-size risk rho = 0.5238；
- 控制 log(Nstimulated) 后：
  - median partial rho = 0.3349；
  - B=1000 CI [0.0469, 0.5378]；
  - 15/15 methods 为正。

建议口径：

> Frozen SafeConf task risk is associated with high-dimensional benchmark MSE
> across methods. Cell count is a strong nuisance factor, but a positive
> association remains in a post hoc sample-size-adjusted analysis.

不要写：

> The association is independent of sample size.

## 3. 单特征

- disagreement-only：median rho = 0.6424，CI [0.4713, 0.7693]；
- support-only：median rho = 0.1235，CI [-0.0899, 0.3374]；
- context-only：聚合后在 74 个任务上为常数，相关未定义。

## 4. Metric/DEG sensitivity

| metric | DEG | median rho |
|---|---:|---:|
| MSE | 20 | 0.0090 |
| MSE | 50 | 0.0835 |
| MSE | 100 | 0.2121 |
| pearson_distance | 5000 | -0.4265 |

因此主结论是 **MSE DEG=5000-specific**。不能扩张成所有误差指标都支持。

## 5. sciplex3 sensitivity

仅使用 35 exact + 25 alias，共 60 drugs；15 manual 全排除。

| analysis | methods | median rho | positive methods |
|---|---:|---:|---:|
| A549 | 9 | 0.4577 | 9/9 |
| K562 | 9 | 0.4690 | 7/9 |
| MCF7 | 9 | 0.4246 | 8/9 |
| pooled | 9 | 0.3835 | 7/9 |

sciplex3 只作 sensitivity，不参与 Frangieh gate。

## 6. 请 Claude 重点验收

1. 是否接受 E8b 预注册 gate = PASS；
2. 是否接受样本量问题按“强 nuisance + post hoc 调整后仍为正”表述；
3. 是否同意把 E8b claim 限制为 high-dimensional MSE association；
4. sciplex3 是否只放 supplement sensitivity；
5. 是否到此停止扩实验，转入 evidence-to-claim 和图表阶段。
