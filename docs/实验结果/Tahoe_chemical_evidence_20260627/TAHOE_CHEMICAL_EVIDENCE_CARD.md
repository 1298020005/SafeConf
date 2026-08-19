# Tahoe chemical evidence card

## 现在到了哪里

```text
Tahoe 数据是否值得做？
        │
        └─ D0 数据地图审计：值得，与主表药物重叠 <8%
                    │
                    ├─ D1 pair holdout：风险信号明显为正
                    │
                    ├─ D2 held-out cell line：跨细胞系仍为正
                    │
                    ├─ D3 practical triage：能抓坏预测，但 magnitude 更强
                    │
                    ├─ D4 stricter gene threshold：主结论不依赖 850 门槛
                    │
                    └─ D5 fixed combination：联合后仍未超过 magnitude
```

## 证据表

| 实验 | 它回答什么 | 核心数字 | 判定 | 论文角色 |
|---|---|---|---|---|
| D0 | Tahoe 是否是值得做的外部 chemical 资源 | 与既有数据药物重叠约 7% | PASS_WITH_CAVEAT | 数据独立性审计 |
| D1 | held-out pair 上 SafeConf 是否有信号 | partial rho 0.453, CI [0.441, 0.467] | PASS | Tahoe chemical 主证据 |
| D2 | 留出整个 cell line 后是否仍有信号 | partial rho 0.470, CI lower 0.457; applicability 91.7% | PARTIAL | 跨细胞系敏感性 |
| D3 | 实际分诊能抓多少坏预测 | SafeConf 4.88x; magnitude 6.49x | PASS_MAGNITUDE_STRONGER | practical value + 诚实边界 |
| D4 | 结论是否依赖 850-gene 门槛 | min-900 partial rho 0.461, CI [0.449, 0.472] | PASS | 稳健性附录 |
| D5 | 固定联合 SafeConf+magnitude 能否超过 magnitude | combined 5.73x vs magnitude 6.49x | PASS_USEFUL_NOT_BETTER | 负结果边界 |

## 最简单的人话

| 问题 | 回答 |
|---|---|
| chemical 数据太少吗？ | 现在不再只靠 Srivatsan/McFarland。Tahoe 提供 9,000 任务、25 个 context、约 1,000 个 perturbation 的大型补充。 |
| Tahoe 支持 SafeConf 吗？ | 支持“存在稳定 task-risk 信号”，partial rho 约 0.45–0.47。 |
| SafeConf 比 magnitude 强吗？ | 不是。Tahoe 的 top-10 检索中 magnitude 更强。 |
| 联合两者会不会更强？ | 固定 50/50 联合比 SafeConf 强，但仍弱于 magnitude。 |
| 这会不会让论文变差？ | 不会。它让论文的边界更可信：SafeConf 不是 magnitude 的普遍替代品。 |

## 正文可用的最小结论

> In the large Tahoe chemical perturbation resource, frozen SafeConf retained a positive magnitude-controlled association with prediction error under both held-out-pair and held-out-cell-line analyses. The signal was robust to a stricter gene-completeness threshold. However, predicted magnitude remained stronger for top-decile error retrieval, and a fixed rank combination did not surpass magnitude alone.

## 不要再跑什么

| 想法 | 为什么现在不跑 |
|---|---|
| 继续搜索最佳 SafeConf/magnitude 权重 | 会在 test set 上调参 |
| 继续改 gene threshold 到 875/925/950 | D4 已回答门槛稳健性，再跑边际收益很低 |
| 把 held-out drug 强行当主验证 | 当前 V0 predictor 需要同药物历史支持，问题定义不成立 |
| 强行调公式救 Tahoe top-10 | 会破坏 frozen v0.2 和预注册口径 |

## 下一步

```text
实验线：暂停。D1-D5 已闭环。
       ↓
只读复核：让 Claude 核对 D1-D5 数字和 claim boundary。
       ↓
叙事决策：老师是否接受“task-risk triage”主线。
       ↓
写作：把 Tahoe 放为 large chemical supplement，不写成 magnitude 胜利实验。
```
