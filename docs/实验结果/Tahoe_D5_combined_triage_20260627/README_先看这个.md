# Tahoe D5: fixed SafeConf + magnitude triage

## 一句话结论

Tahoe 上的固定 SafeConf + magnitude 联合分诊比 SafeConf 单独更强，但没有超过 magnitude-only。

## 主结果

| 分数 | top-10% enrichment | aligned rho | 解读 |
|---|---:|---:|---|
| SafeConf | 4.880x | 0.399 | 有用，但弱于 magnitude |
| magnitude-only | 6.486x | 0.485 | Tahoe 上的最强单一基线 |
| 50% SafeConf + 50% magnitude | 5.730x | 0.489 | 比 SafeConf 强，但显著弱于 magnitude |
| 25% SafeConf + 75% magnitude | 6.263x | 0.503 | rho 略升，top-10 仍未超过 magnitude |
| 75% SafeConf + 25% magnitude | 5.240x | 0.453 | 弱于上述两种 |

Primary 50/50 combination:

```text
top-10 enrichment = 5.730x
95% CI            = [5.451, 5.922]
combined - magnitude 95% CI = [-1.017, -0.608]
gate = PASS_USEFUL_NOT_BETTER
```

## 结论边界

- 可以说：在 Tahoe 上，固定联合分数仍是有效的高错误分诊信号。
- 可以说：SafeConf 的加入让 aligned rho 略有上升，但没有改善 top-10 检索。
- 不能说：SafeConf 与 magnitude 联合后在 Tahoe 上超过 magnitude-only。
- 不能继续试更多权重，然后只报最好的一个；那会变成 test-set 调参。
- D5 是 D3 后预注册的敏感性分析，不是 frozen v0.2 的一部分。

## 为什么这个负结果仍有价值

```text
原先的可能性：SafeConf 和 magnitude 一合并就能更强
                       ↓ D5 否定
Tahoe 事实：magnitude 已捕获大部分最顶端错误排序
                       ↓
论文应该说：SafeConf 提供 magnitude 之外的 partial-rho 信号，
但在 Tahoe 的 top-10 实用检索上，magnitude 仍然更强。
```

## 文件

- `PREREGISTRATION.md`: 计算前冻结的权重、指标和 gate。
- `TAHOE_D5_STATUS.json`: 主状态。
- `tables/TAHOE_D5_POINT_SUMMARY.csv`: 各分数在 top 5%/10%/20% 的结果。
- `tables/TAHOE_D5_TOP10_TASK_CLUSTER_CI.csv`: 1,000 次 task-cluster bootstrap CI。

## 大型输出位置

```text
/home/yyf/safeconf_runtime/outputs/tahoe_d5_combined_triage_20260627/
```
