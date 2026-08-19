# E8b scPerturBench aggregate-error association - 冻结预注册

日期：2026-06-15

决策者：Claude（architecture review）

执行者：Codex（implementation and audit）

状态：**FROZEN BEFORE ASSOCIATION ANALYSIS**

## 1. 目标

验证 SafeConf frozen v0.2 task-risk score 是否与 scPerturBench 官方
benchmark 中各方法的 per-perturbation error 正相关。

本实验的 claim boundary 冻结为：

> external benchmark method-error association on shared biological datasets

本实验不能称为：

- 独立生物数据集外部验证；
- 对 scPerturBench 方法运行完整 SafeConf；
- 完整的跨架构向量级评分；
- 27 种架构验证。

## 2. 数据来源与版本

| 来源 | 固定版本 |
|---|---|
| scPerturBench genetic aggregate CSV | `6e24e7a9827e55d4567d2139427be9af0d1e7a6c` |
| scPerturBench chemical aggregate CSV | `6e24e7a9827e55d4567d2139427be9af0d1e7a6c` |
| SafeConf fold-safe E1-E4 evidence | `efa95556efe7c7ab1ca66a43ac438da286fcabc0` |
| Frozen v0.2 scorer | 当前 codebase；不得修改公式 |

scPerturBench 原始 CSV 存放在：

```text
/home/yyf/data/scperturbench/
```

原始 CSV 不进入 Git。

## 3. 分析层次

| 层次 | 数据集 | 角色 |
|---|---|---|
| Primary | Frangieh | 主分析 |
| Sensitivity | sciplex3，不含 `sciplex3_comb` | 敏感性分析 |

Frangieh 有 74 个 benchmark perturbations，均能与 SafeConf 基因符号精确
匹配。sciplex3 只能在显式 drug alias mapping 审批后运行。

## 4. Primary metric

主指标：

```text
metric = mse
DEG = 5000
```

选择依据：

- SafeConf Frangieh effect vector 为 5000 维；
- SafeConf Frangieh 使用
  `official_generalization/Frangieh.h5ad` 的 5000 HVG；
- scPerturBench 使用 `DEG_hvg5000` 计算该指标；
- MSE 与 RMSE 平方单调对应，不改变 Spearman 排序。

以上说明两边均使用 5000-gene 级别的评估，但不额外声称两个基因列表已经逐项
证明完全相同。

敏感性指标：

- `pearson_distance, DEG=5000`；
- `mse, DEG=20`；
- `mse, DEG=50`；
- `mse, DEG=100`。

sciplex3 的 SafeConf 与 benchmark gene panel 不视为完全相同，因此只作
sensitivity analysis。

## 5. Score 定义与聚合

必须调用真实 frozen v0.2 scorer，不得在 E8b 中重新实现或修改公式。

Frozen formulas：

- Frangieh `gene_main`：
  `z(context_similarity_max) + z(log_support) - z(model_disagreement_rmse)`；
- sciplex3 `chem_robust`：
  `z(log_support) - z(model_disagreement_rmse)`。

其中每个 fold 的 median/IQR 标准化只能使用该 fold 的 train rows。

聚合顺序冻结如下：

1. 每个 fold 使用该 fold train rows 构造 median/IQR；
2. scorer 可以计算 train+val，但不得使用 test rows 形成最终外部 task score；
3. 每个 `(fold, context, perturbation)` 对
   V0StrongBaseline 和 ContextSimBaseline 的 confidence 取 median；
4. 每个 `(context, perturbation)` 跨 folds 取 median；
5. Frangieh 每个 perturbation 再跨三个 contexts 取 median；
6. `risk = -confidence`。

`model_disagreement_rmse` 依赖 SafeConf 的 V0/ContextSim reference
predictions。因此该分数可以称为 retrieval-reference task risk，不能称为
完全不运行任何参考预测器的纯 metadata-only score。

## 6. Benchmark error 聚合与 seed coverage

每个 `(method, perturbation)` 对可获得的 benchmark seeds 取 median。

不得写成每个 perturbation 都具有完整三个 seed。已知 Frangieh
`mse, DEG=5000` 的 seed coverage 为：

- 57/74 perturbations：1 个 seed；
- 15/74 perturbations：2 个 seeds；
- 2/74 perturbations：3 个 seeds。

正式输出必须包含 seed coverage audit，并保留每个 perturbation 实际可用的
seed 数。

## 7. 对齐规则

### Frangieh

- 使用 SafeConf `perturbation` 列与 scPerturBench `perturb` 列精确匹配；
- 预期覆盖 74/74；
- SafeConf 的三个 contexts 聚合到 perturbation-only 粒度；
- `task_key` 是内部序号，不得用于外部 join。

### sciplex3

- 排除 `sciplex3_comb`；
- context mapping 冻结为：

```text
alveolar basal epithelial cells -> A549
lymphoblasts -> K562
mammary epithelial cells -> MCF7
```

- 只使用 Claude 审批后的 explicit alias table；
- 正式 sensitivity 仅使用 `exact + alias`；
- benchmark 四个 doses 先按 `(context, drug)` 取 median；
- 不使用模糊字符串匹配；
- manual 条目超过 10 个时全部舍弃，不硬凑。

## 8. 负对照与 nuisance baselines

Frangieh 主分析同时报告：

1. frozen v0.2 risk；
2. sample-size baseline：
   `sample_size_risk = -log1p(Nstimulated)`；
3. shuffled-risk permutation null：
   200 permutations，seed = 5201；
4. frozen score 组成特征的单独外部关联：
   - `-z(context_similarity_max)`；
   - `-z(log_support)`；
   - `+z(model_disagreement_rmse)`。

`Nstimulated` 的准确来源和聚合方式必须在 dry audit 中记录。若使用
scPerturBench aggregate CSV 的 `Nstimulated`，必须明确它是 benchmark
task metadata，而不是 SafeConf feature matrix 字段。

## 9. Per-method inference

每个 method 输出：

- aligned perturbation 数；
- Spearman rho；
- raw p-value；
- Benjamini-Hochberg q-value。

Methods 共享同一批 perturbations，不能当成独立生物重复。单方法 p/q 值只作
描述，正式 gate 不依赖单方法显著性。

## 10. Perturbation-cluster bootstrap

- bootstrap 次数：1000；
- 重采样单位：perturbation；
- seed：5201。

每次 bootstrap：

1. 对 perturbations 有放回重采样；
2. 在相同重采样任务上重新计算每个 method 的 Spearman；
3. 取 across-method median Spearman；
4. 保存该 median。

1000 个 median Spearman 的 2.5% 和 97.5% 分位数构成 95% CI。

## 11. Gate

### PASS

- across-method median Spearman 的 bootstrap CI lower > 0；
- 且至少 60% methods 的 rho > 0。

### PARTIAL

- median Spearman > 0 但 CI 跨 0；
- 或 methods 正相关比例处于 40%-60%。

### FAIL

- median Spearman <= 0。

若多个条件冲突，采用更保守等级。sciplex3 sensitivity 不参与 primary gate。

## 12. 计划输出

所有紧凑证据保存到：

```text
docs/实验结果/E1_E4_preregistered_20260614/E8b_scperturbench_association/
```

| 文件 | 内容 |
|---|---|
| `E8b_PREREGISTRATION.md` | 本文件 |
| `E8b_FRANGIEH_PER_METHOD.csv` | method-level rho、raw p、BH q |
| `E8b_FRANGIEH_CONTROLS.csv` | nuisance 与 negative controls |
| `E8b_FRANGIEH_PERFEATURE.csv` | frozen component associations |
| `E8b_FRANGIEH_SUMMARY.json` | primary result 与 gate |
| `E8b_FRANGIEH_SEED_COVERAGE.csv` | available-seed coverage |
| `E8b_SENSITIVITY_DEG.csv` | Frangieh metric/DEG sensitivity |
| `E8b_SCIPLEX3_PER_METHOD.csv` | sciplex3 sensitivity |
| `E8b_SCIPLEX3_DRUG_ALIAS.csv` | 经审批的映射表副本 |

## 13. 不做的事

- 不修改 frozen v0.2；
- 不使用 learned model；
- 不下载额外 H5AD、PKL 或 Podman；
- 不使用 SafeConf 或 scPerturBench test error 训练分数；
- 不把 scPerturBench 原始 CSV 放入 Git；
- 不声称 27 种架构验证；
- 不在 alias audit 批准前运行 sciplex3 sensitivity。

## 14. 数据文件哈希

哈希算法：SHA-256。

```text
f8f5a2a7700075ce92458ccd7bac16302ebd26d748e2fef28c6424596ac087fa  all_dataset_genetic.csv
541452ad9b3c3f0635cf7d6df6611909d3ebd2c29de7ad85f761fda09a9195b0  all_dataset_chemical.csv
```

文件大小：

```text
all_dataset_genetic.csv   55,119,282 bytes
all_dataset_chemical.csv  14,773,597 bytes
```

在本预注册冻结后，不得根据观察到的 E8b 关联结果修改 primary dataset、
primary metric、score direction、aggregation order、bootstrap gate 或 claim
boundary。
