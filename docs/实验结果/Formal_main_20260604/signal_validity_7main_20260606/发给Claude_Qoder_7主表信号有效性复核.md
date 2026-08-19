# 请 Claude / Qoder 复核：7 主表 signal validity audit

这次请只审计这一个问题：

> SafeConf 的 confidence score（可信度分数）是不是只是 effect magnitude（效应大小）的伪相关？

## 本次新增内容

Codex 已补跑 7 主表正式版本的 signal validity audit（信号有效性审计）。

注意：这不是旧 v6 结果。  
输入来自：

```text
Formal_main_20260604/formal_audit/tables/FORMAL_SCORED_RECORDS.csv
```

输出目录：

```text
docs/实验结果/Formal_main_20260604/signal_validity_7main_20260606/
```

## 关键表

```text
tables/SIGNAL_VALIDITY_7MAIN_MAIN_SCORE.csv
tables/PARTIAL_AND_WITHIN_STRATUM_7MAIN.csv
tables/MAGNITUDE_BASELINE_7MAIN.csv
reports/SIGNAL_VALIDITY_7MAIN_REPORT.md
```

## 目前 Codex 的解释

1. magnitude-only（只看效应大小）在多数数据集上很强，所以 effect magnitude confounding（效应大小混杂）必须正面承认。
2. 但是 6/7 数据集的 partial ρ（控制效应大小后的相关）仍为正，说明 SafeConf 不只是幅度伪相关。
3. gene_main（基因主线）4 个数据集全部 partial ρ 为正。
4. chem_robust（化学线）里 Srivatsan 和 Santinha 为正，但 McFarland 失败。
5. McFarland 继续作为 failure boundary（失败边界），不改 frozen protocol v0.2。

## 请你们重点挑刺

1. `within-perturbation ρ` 和 `within-context ρ` 是否足以回答“不是假信号”？
2. Santinha 的 within-perturbation ρ 只有约 0.019，这是否削弱 chemical line（化学线）？
3. Srivatsan 的 within-perturbation ρ 也只有约 0.074，但 within-context ρ 约 0.538，这应该怎么解释？
4. 是否应该把主 claim 收缩成：  
   “gene_main 稳定，chemical line 有正信号但存在失败边界”？
5. 这套 signal validity 结果是否足以支持二区投稿的 Methods/Results，还是还必须补别的控制实验？

## 禁止误读

- 不要把 Tahoe 写成 7 主表正式证据。
- 不要把 GEARS alignment 写成已经成功。
- 不要说 model_disagreement（模型分歧）是唯一最稳定信号。
- 不要把旧 v6 signal validity 当成本次正式 7 主表审计。

