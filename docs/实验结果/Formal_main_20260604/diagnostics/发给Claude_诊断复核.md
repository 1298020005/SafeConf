# 发给 Claude：Formal main 后诊断复核

请你按审稿人视角复核，不要默认同意 Codex，也不要因为用户想冲高目标就放宽标准。

## 你先读哪些文件

本目录：

```text
docs/实验结果/Formal_main_20260604/diagnostics/
```

请按顺序读：

1. `reports/FORMAL_MAIN_DECISION.md`
2. `reports/McFarland_failure_diagnosis.md`
3. `reports/Tahoe_eligibility_audit.md`
4. `tables/McFarland_single_feature_diagnostics.csv`
5. `tables/McFarland_per_dose_rho.csv`
6. `tables/McFarland_per_time_rho.csv`
7. `tables/FEATURE_CORRELATION_TOP_PAIRS.csv`
8. `tables/Tahoe_drug_cell_line_matrix.csv`

## Codex 这次做了什么

新增代码：

```text
code/20260426_154505_perturb_transport_final_push/safetrans_confidence/cli/run_post_formal_diagnostics.py
```

它只做诊断，不训练模型、不改公式、不下载新数据。

输出：

```text
code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_20260604/post_formal_diagnostics/
docs/实验结果/Formal_main_20260604/diagnostics/
```

## 关键结果

### McFarland

- v0.2 主公式 aligned rho = -0.086
- v0.2 主公式 partial rho = -0.061
- `learned_risk_score` aligned rho = 0.587
- `historical_residual_risk` aligned rho = 0.426
- `support_count_score` aligned rho = -0.148
- `context_similarity_score` aligned rho = -0.078
- h5ad metadata 中 observed non-control cell_line × drug pairs = 1175
- formal held-out test pairs = 1163
- 有 7 个 dose value 和 3 类 time label

Codex 当前判断：

> McFarland 不应删，也不应为了它改冻结公式；它应该作为 chem_robust 的 failure boundary。若后续要救，应先把 task 改成 drug-dose-time，而不是调 v0.2 权重。

### Tahoe

- 已下载约 71.6GB
- obs metadata rows = 100,648,790
- obs 中 drug × cell_line pairs = 19,000
- 至少 20 cells 的 pairs = 18,999
- pseudobulk sample 含 `log2FoldChange`、`n_cells_trt`、`n_cells_ctrl`、`drug`、`concentration`、cell-line identifiers

Codex 当前判断：

> Tahoe 不进当前 7 主表，但非常适合下一步 external mega-scale validation。下一步应写 pseudobulk adapter + leakage audit。

## 请你回答

1. 你是否同意 McFarland 应保留为 failure boundary，而不是从主表删除？
2. 你是否同意不为 McFarland 修改 frozen protocol v0.2？
3. McFarland 上 `learned_risk_score` 和 `historical_residual_risk` 有强信号，这在论文里应写成 ablation、future work，还是可以作为 McFarland-specific rescue？
4. Tahoe 是否值得现在优先接 pseudobulk adapter？还是应先跑 supplement 数据集 / GEARS？
5. `FEATURE_CORRELATION_TOP_PAIRS.csv` 显示一些特征高度冗余，你建议论文主文保留哪些特征，哪些放附录？
6. 如果目标是稳二区、冲一区，你认为下一步最应该做哪 2 件事？

请直接给结论和理由。不要写泛泛鼓励。
