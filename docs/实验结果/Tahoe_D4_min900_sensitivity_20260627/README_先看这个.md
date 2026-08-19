# Tahoe D4: min-900 gene-completeness sensitivity

## 一句话结论

D1 的 Tahoe chemical 结论不依赖 `min_genes_per_task=850`。将门槛严格提高到 900 后，SafeConf 仍对测试任务误差保持稳定正相关。

## D1 与 D4 对照

| 项目 | D1 主分析 | D4 严格敏感性 | 解读 |
|---|---:|---:|---|
| 每任务最少非空基因 | 850 | 900 | D4 更严格 |
| 任务数 | 9,000 | 9,000 | 样本量不变 |
| context | 25 | 25 | 覆盖不变 |
| perturbation | 1,028 | 1,023 | 只减少 5 个 |
| test task clusters | 8,057 | 7,997 | 减少 0.7% |
| aligned rho | 0.399 | 0.371 | 略下降，仍稳定为正 |
| partial rho | 0.453 | 0.461 | 控制 magnitude 后不降反升 |
| partial rho 95% CI | [0.441, 0.467] | [0.449, 0.472] | 两者都明显高于 0 |
| RC@80 improvement | 4.28% | 4.06% | 小幅下降 |
| Gate | PASS | PASS | 敏感性通过 |

## 可以说什么

- Tahoe 的 chemical task-risk 信号对基因完整度门槛不敏感。
- 控制真实 effect magnitude 后，partial rho 仍为 0.461，95% CI 不跨 0。
- D4 支持 D1 的稳健性，不取代 D1 主分析。

## 不能说什么

- 不能说 SafeConf 在 Tahoe 上比 magnitude-only 更强；D3 显示 magnitude-only 的 top-10% enrichment 更高。
- 不能说是 held-out-drug 验证；当前 V0-family predictor 需要同药物历史支持。
- 不能说是完全独立的深度学习预测器验证。

## 文件

- `D4_D1_COMPARISON.csv`: D1/D4 核心数字对照。
- `RUN_STATUS.json`: D4 主程序状态。
- `TAHOE_D1_POSTPROCESS_STATUS.json`: task-cluster bootstrap 状态。
- `tables/TAHOE_D1_FORMAL_SUMMARY.csv`: overall/fold/predictor 点估计。
- `tables/TAHOE_D1_TASK_CLUSTER_BOOTSTRAP_CI.csv`: 1,000 次 task-cluster bootstrap CI。

## 大型输出位置

```text
/home/yyf/safeconf_runtime/outputs/tahoe_d4_min900_pair_20260627/
/home/yyf/safeconf_runtime/outputs/tahoe_d4_min900_pair_taskcluster_20260627/
```
