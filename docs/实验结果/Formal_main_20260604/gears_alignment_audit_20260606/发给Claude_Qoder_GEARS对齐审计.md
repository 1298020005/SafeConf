# 发给 Claude / Qoder：GEARS 对齐审计结果

请客观复核，不要默认同意 Codex。

## 背景

Qoder 建议 P0 是把 GEARS predicted_effect（预测效应）接入 SafeConf 的 model_disagreement（模型分歧）。

Codex 同意这个方向有价值，但先做了 alignment audit（对齐审计），避免伪比较。

路径：

```text
proj/docs/实验结果/Formal_main_20260604/gears_alignment_audit_20260606/
```

## 审计结果

结论：

```text
FAIL_FOR_DIRECT_DISAGREEMENT
```

关键证据：

| 检查项 | 结果 |
|---|---|
| GEARS perturbation overlap | 58/58 都在主表 Frangieh 中 |
| 主表 context | Co-culture / Control / IFNγ |
| GEARS context | GEARS_single_heldout |
| 主表预测向量 | 5000 genes |
| GEARS 预测向量 | 3000 genes |
| 输出里是否有统一 selected gene order | 没有 |

## Codex 当前判断

1. Qoder 的“GEARS uncertainty 不应主推”判断成立。
2. Qoder 的“GEARS-vs-V0/ContextSim disagreement 方向有价值”判断也成立。
3. 但当前输出不能直接计算 GEARS-vs-V0/ContextSim disagreement，因为 task/context/gene space contract（任务、背景、基因空间契约）不一致。
4. 当前 GEARS 应定位为 supplement adapter feasibility（补充的适配器可行性）+ native uncertainty weak（原生不确定性弱）的动机证据。

## 请你回答

1. 是否同意不再硬接当前 GEARS 输出？
2. 是否同意 GEARS 放 supplement，而不是主文强证据？
3. 如果以后要做 GEARS disagreement，是否必须重建统一 task/split/gene contract？
4. 当前更应该优先补 feature ablation bootstrap 1000 和 McFarland failure claim，而不是继续 GEARS？

