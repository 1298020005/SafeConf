# Phase 5a-0 Cost-effectiveness / Prediction Triage Report

日期：2026-06-16

## 1. 这一步做了什么

本阶段没有跑新实验，只把已有 bad-prediction retrieval 结果整理成
Fig 5 和 supplement risk-coverage 的 figure-ready tables。

新增输出：

- `figure_ready_tables/FIG5_COST_EFFECTIVENESS.csv`
- `figure_ready_tables/FIG5_COST_EFFECTIVENESS_MACRO_TOP10.csv`
- `figure_ready_tables/FIG5_COST_EFFECTIVENESS_HEATMAP.csv`
- `figure_ready_tables/SFIG_RISK_COVERAGE.csv`
- `figures/FIG5_cost_effectiveness.png`
- `figures/FIG5_cost_effectiveness.svg`
- `figures/SFIG5_cost_effectiveness_thresholds.png`
- `figures/SFIG5_cost_effectiveness_thresholds.svg`

## 2. 数据来源

Fig 5：

- `docs/实验结果/Task_risk_audit_20260611/tables/B1_bad_prediction_retrieval.csv`

Supplement risk-coverage：

- `docs/实验结果/Formal_main_20260604/paper_figures/tables/PAPER_RISK_COVERAGE_CURVES.csv`

`__macro_mean__` 行表示 7 个真实数据集的 macro mean，不是第 8 个数据集。

## 3. Top 10% macro enrichment

| Strategy | Score name | Enrichment |
|---|---|---:|
| Random | `random` | 1.11x |
| Magnitude-only | `predicted_magnitude` | 3.30x |
| Frozen v0.2 | `protocol_v0_2_family_confidence` | 3.35x |
| LODO risk | `safeconf_lodo_risk` | 2.31x |
| Per-dataset risk | `safeconf_perdataset_risk` | 5.36x |
| Oracle | `oracle_magnitude_diagnostic` | 8.21x |

可写核心句（含 per-dataset heterogeneity caveat）：

> At the top 10% risk threshold, frozen SafeConf identifies 3.35x more
> high-error predictions than random selection, comparable to the magnitude-only
> baseline (3.30x) in macro average. However, per-dataset patterns diverge
> substantially: frozen scoring outperforms magnitude on datasets where effect
> magnitude poorly predicts error (Lara ex vivo: 7.80x vs. 0.46x), whereas
> magnitude dominates on datasets with strong magnitude-error coupling
> (Frangieh: 8.08x vs. 2.90x), suggesting the two signals are complementary
> rather than redundant.

## 4. Top 10% per-dataset heterogeneity

| Dataset | Frozen v0.2 | Magnitude-only | Pattern |
|---|---:|---:|---|
| Cui | 3.30x | 2.55x | frozen higher |
| Frangieh | 2.90x | 8.08x | magnitude higher |
| Lara ex vivo | 7.80x | 0.46x | frozen much higher |
| Lara in vivo | 2.53x | 2.27x | similar |
| McFarland | 0.90x | 2.57x | magnitude higher |
| Santinha | 1.08x | 0.90x | both weak |
| Srivatsan | 4.95x | 6.27x | magnitude higher |

## 5. Fig 5 草图结构

Panel A：

- top 10% macro-mean enrichment bar chart；
- 包含 Random、Magnitude-only、Frozen v0.2、LODO risk、Per-dataset risk、Oracle。
- Oracle 左侧加虚线分隔，并标注为 non-deployable reference。

Panel B：

- 7 个真实数据集的 top 10% enrichment heatmap；
- strategies = Random、Magnitude-only、Frozen v0.2、LODO risk、Per-dataset risk；
- 不画 Oracle，避免颜色范围被上界压缩；
- 完整 top 5% / 10% / 20% 三阈值版本放入 supplement figure：
  `SFIG5_cost_effectiveness_thresholds`.

## 6. 口径边界

- Fig 5 不能写成 frozen v0.2 全面超过 magnitude-only。
  正确口径是：在 top 10% high-error retrieval 这个 practical triage 指标上，
  frozen v0.2 与 magnitude-only macro 平均几乎持平，但 per-dataset 强弱互补。
- 不能写成 "frozen v0.2 matches magnitude in cost-effectiveness"。
  正确措辞是 "comparable in macro-averaged enrichment, with complementary
  per-dataset strengths."
- `Per-dataset risk` 是 within-dataset training/reference upper bound，
  不能写成 deployable frozen protocol。
- `Oracle` 使用非部署式真实效应诊断，只能作为参考上界。
- LODO risk 在 macro top 10% 高于 random，但低于 magnitude-only 和 frozen v0.2；
  不应写成跨数据集 risk 排序优于 magnitude。
- 主文 Panel B 显示 top 10% per-dataset heterogeneity，正文应避免“一刀切”叙述。

## 7. 给 Claude 的验收点

请重点判断：

1. Main Fig 5 的 top10-only Panel B 是否已经足够清晰；
2. Oracle 的 non-deployable reference 分隔是否合适；
3. 完整三阈值版本作为 supplement 是否接受；
4. 上面的 heterogeneity 核心句是否可以进入 Results 2.7；
5. 是否接受 `prediction triage / high-error retrieval` 作为 practical-value 段落。
