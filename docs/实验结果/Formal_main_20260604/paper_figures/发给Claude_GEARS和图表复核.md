# 发给 Claude：GEARS-Cui 可行性 + 论文图表复核

请继续按审稿人视角复核，不要默认同意 Codex。

## 先读目录

```text
docs/实验结果/Formal_main_20260604/paper_figures/
```

请按顺序读：

1. `README_先看这个.md`
2. `reports/GEARS_CUI_FEASIBILITY_AUDIT.md`
3. `reports/GEARS_EXISTING_SUPPLEMENT_STATUS.md`
4. `figures/F1_per_dataset_rho_bars.png`
5. `figures/F2_risk_coverage_curves.png`
6. `figures/F3_mcfarland_dose_partial_rho.png`

同时看诊断目录里的新表：

```text
docs/实验结果/Formal_main_20260604/diagnostics/tables/
```

重点：

1. `McFarland_leave_one_dose_out_rho.csv`
2. `McFarland_time_label_audit.csv`

## Codex 新发现

### 1. GEARS on Cui 不成立

CuiHacohen2023 的 perturbation_type 是 `cytokines（细胞因子）`。

非 control perturbation = 86 个。

其中和基因名精确匹配的只有 1 个：`Flt3l`。

因此 Codex 判断：

> 不能在 Cui 上硬跑 GEARS。GEARS 是 gene perturbation（基因扰动）模型，把 cytokine 名称当作 gene perturbation 会变成伪实验。

请你确认：

Q1. 你是否同意撤销“GEARS on Cui”这个执行建议？
Q2. 如果不同意，请说明 GEARS 如何合法处理 cytokine perturbation。

### 2. 已有 GEARS 只能作 supplement

已有 GEARS formal 输出来自 Norman / Adamson / Dixit。

现状：

- 9 个 seed run
- 54 条 GEARS prediction records
- native uncertainty（原生不确定性）没有导出
- 主要可评估的是 prediction magnitude risk

Codex 判断：

> 这只能说明 SafeConf adapter 能读 GEARS per-prediction records，不能作为当前 7 主表的第三 predictor 主证据。

请你确认：

Q3. GEARS supplement 应该怎么写进论文？主文一段，还是附录？
Q4. 这是否足以回应“你只用了玩具 predictor”的审稿质疑？如果不够，下一步应该接哪个更合适的 predictor？

### 3. McFarland dose=2.5 补诊断

per-dose 表显示 dose=2.5 的 partial rho = -0.533，确实严重反向。

但 Codex 补了 leave-one-dose-out：

- 去掉 dose=2.5 后 aligned rho = -0.013
- 去掉 dose=2.5 后 partial rho = 0.011

也就是说，去掉最坏剂量后只是接近 0，没有真正回正。

请你确认：

Q5. McFarland 失败是否仍应写成整体 task structure 问题，而不是“只有 dose=2.5 拖垮”？

### 4. 论文图草稿

Codex 生成了：

- `F1_per_dataset_rho_bars.png`
- `F2_risk_coverage_curves.png`
- `F3_mcfarland_dose_partial_rho.png`

请你回答：

Q6. 这三张图是否适合做论文 Figure 初稿？
Q7. 还缺哪 2 张最必要的图？

## 请给最终下一步建议

请在下面三个选项里排序：

1. 跑 8 个 supplement datasets
2. 整理已有 GEARS supplement + 写进附录
3. 继续论文图表和主文 Results 初稿
4. Tahoe pseudobulk adapter

请给出排序和理由。
