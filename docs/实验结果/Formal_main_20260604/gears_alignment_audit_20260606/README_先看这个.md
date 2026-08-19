# GEARS 与主表 Frangieh 对齐审计

更新时间：2026-06-06

## 一句话结论

Qoder 建议“把 GEARS predicted_effect（预测效应）接入 model_disagreement（模型分歧）”，方向是对的，但当前文件不能直接这么做。

本次审计结论：

> `FAIL_FOR_DIRECT_DISAGREEMENT`：当前 GEARS 输出不能直接和 V0/ContextSim 输出相减计算分歧。

## 为什么不通过

| 项目 | 结果 | 判断 |
|---|---:|---|
| perturbation（扰动）对齐 | GEARS 58/58 个扰动都能在主表 Frangieh 找到 | 通过 |
| context（背景）对齐 | 主表是 `Co-culture / Control / IFNγ`，GEARS 是 `GEARS_single_heldout` | 不通过 |
| prediction array（预测向量）维度 | 主表 5000 genes，GEARS 3000 genes | 不通过 |
| selected gene order（筛选基因顺序） | 当前输出没有导出统一 gene order | 不通过 |

## 通俗解释

扰动名字能对上，说明“题目名字”差不多。

但两边真正预测的不是同一张考卷：

- 主表 Frangieh 是跨 `context（背景）` 的 `(context, perturbation)` 预测；
- GEARS 是 single-gene heldout（留出单基因扰动）预测；
- 主表预测 5000 个基因；
- GEARS 预测 3000 个基因；
- 当前输出没有一个共同的 gene order（基因顺序）文件。

所以如果现在硬算 `GEARS vs V0`，会变成伪比较。

## 这对项目意味着什么

这不是 GEARS 失败，而是说明：

1. GEARS 目前适合作为 supplement（补充）里的 adapter feasibility（适配器可行性）；
2. GEARS native uncertainty（原生不确定性）弱，可以作为“为什么需要外部 SafeConf 打分”的动机；
3. 不应在当前 sprint（短冲刺）继续硬接 GEARS-vs-V0/ContextSim disagreement；
4. 若以后要做，必须从头统一 task definition（任务定义）、context（背景）、split（切分）、gene panel（基因集合）和 gene order（基因顺序）。

## 关键文件

- 审计报告：`reports/GEARS_ALIGNMENT_AUDIT.md`
- 扰动重叠：`tables/GEARS_MAIN_PERTURBATION_OVERLAP.csv`
- 背景审计：`tables/GEARS_MAIN_CONTEXT_AUDIT.csv`
- 数组维度审计：`tables/GEARS_MAIN_ARRAY_SPACE_AUDIT.csv`
- 基因空间审计：`tables/GEARS_MAIN_GENE_SPACE_AUDIT.csv`
- 状态：`RUN_STATUS.json`

## 下一步

按照 Qoder 建议，GEARS 不再硬接主表分歧。下一步回到两件更有收益的事：

1. feature ablation（特征消融）bootstrap 提到 1000；
2. McFarland failure mode（失败模式）和主表 claim（主张）收缩。

