# 下一阶段安排：GEARS 先过关，再谈更大模型

更新时间：2026-06-05

## 一句话

现在不继续下载 Tahoe 337GB，也不急着写初稿。下一阶段先把 GEARS（主流扰动预测模型）接入这件事做扎实。

## 当前已经完成

1. 7 主表 formal audit（正式审计）已完成。
2. Tahoe（超大药物扰动数据）sampled formal v1 已完成，可以放 supplement（补充结果）。
3. GEARS Frangieh adapter smoke 已完成：
   - GEARS 能跑。
   - Frangieh 数据能接入。
   - PredictionRecord（预测记录）能导出。
   - 但只有 62 条 test record，还不是 formal validation（正式验证）。

## 当前正在跑

已完成：

```text
GEARS uncertainty probe
output: code/20260426_154505_perturb_transport_final_push/outputs/gears_frangieh_uncertainty_probe_20260605/
docs: docs/实验结果/Formal_main_20260604/gears_frangieh_uncertainty_probe/
```

目的：

> 用 `--uncertainty` 跑一次 GEARS，检查它能不能导出 native uncertainty（原生不确定性）。

结果：

> 可以导出。21/21 条 PredictionRecord 有非空 `gears_uncertainty_logvar_mean`。

这一步很关键，因为如果 GEARS 自己能给 uncertainty，我们就可以公平比较：

- GEARS native uncertainty（模型自己的不确定性）
- SafeConf score（我们外部打分）

如果导不出来，就不能硬说“比较了 GEARS uncertainty”，只能用 seed ensemble proxy（多随机种子分歧代理）。

## 下一阶段优先级

### Step 1：GEARS uncertainty probe

已完成，`--uncertainty` 可以产生非空 `gears_uncertainty_logvar_mean`。

结果分两种：

| 情况 | 下一步 |
|---|---|
| 有 native uncertainty | 已发生。下一步做 GEARS native uncertainty vs true error 的正式评估 |
| 仍然为空 | 停止追 native uncertainty，改用 seed ensemble proxy 或只写 adapter smoke |

### Step 2：GEARS split compatibility audit

当前 GEARS smoke 用的是 GEARS 自己的 `single` split（按扰动留出），不是 SafeConf 的 held-out pair split（背景×扰动留出）。

必须先审：

- 它能不能回答 cross-context confidence scoring（跨背景可信度打分）？
- 如果不能，只能作为 supplement probe，不能进主表。

### Step 3：GEARS formal probe

只有 Step 1/2 过关，才扩大 GEARS：

- Frangieh 作为首选数据集。
- 目标不是刷 GEARS MSE，而是导出足够多 PredictionRecord。
- 输出 GEARS predictor 下的 SafeConf score 与 true error 的相关。

### Step 4：再考虑 supplement / 更大数据

Tahoe 已够 supplement，不继续下 337GB。

其他 supplement 数据集可以排队，但优先级低于 GEARS。

## 现在不做什么

- 不下载 Tahoe 337GB raw expression。
- 不为了 McFarland 改 frozen protocol v0.2。
- 不把 GEARS smoke 写成 formal 结果。
- 不继续堆新模型名字。

## 给你的通俗解释

你现在的论文如果想更有说服力，最大短板不是“没有更多数据”，而是：

> 现在主要 predictor（预测器）还是 V0 / ContextSim 这种简单方法。审稿人会问：对 GEARS 这种真正模型还管用吗？

所以我现在优先处理 GEARS。
