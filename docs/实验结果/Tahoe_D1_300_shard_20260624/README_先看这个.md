# Tahoe D1 300-shard sampled chemical validation

日期：2026-06-24
结论：`PASS`

## 一句话

SafeConf frozen-style chemical risk score 在 Tahoe-100M pseudobulk 的
9000 个 cell-line × drug-dose 任务上仍能预测误差：

```text
partial rho = 0.453
task-cluster 95% CI = [0.441, 0.467]
```

## 数据规模

| 项目 | 数值 |
|---|---:|
| 候选 Tahoe shards | 300 |
| 实际读取 shards | 145 |
| tasks | 9000 |
| contexts | 25 |
| drug-dose perturbations | 1028 |
| prediction records | 80812 |
| test task clusters | 8057 |
| genes per vector | 1000 |

达到 9000 tasks 后停止，因此没有继续读取剩余候选 shards。

## 正式结果

| 层级 | aligned rho | partial rho | partial 95% CI | RC@80 |
|---|---:|---:|---:|---:|
| overall | 0.399 | **0.453** | **[0.441, 0.467]** | 4.27% |
| V0DrugMeanAcrossDose | 0.337 | 0.330 | [0.313, 0.346] | 4.29% |
| V0ExactDoseMean | 0.469 | 0.601 | [0.587, 0.616] | 4.27% |

Bootstrap：

```text
B = 1000
unit = task_key
同一 task 的两个 predictor 行一起重采样
```

## Leakage audit

| 检查 | 结果 |
|---|---:|
| held-out pair 出现在 train | 0 |
| test context 无训练支持 | 0 |
| test perturbation 无训练支持 | 0 |
| test plate 在 train 出现比例 | 100% |
| same-drug other-dose support | 98.9% |

最后两项是 caveat：当前结果不是 held-out plate，也不是 held-out drug。

## 与旧 100-shard smoke 的关系

| 项目 | 100-shard | D1 |
|---|---:|---:|
| tasks | 3000 | 9000 |
| test task clusters | 2066 | 8057 |
| partial rho | 0.293 | 0.453 |
| partial CI lower | 0.256 | 0.441 |

旧 3000 个 task 中有 2973 个进入 D1；D1 新增 6027 个 task。

## 参数纠错

原预注册把“1000 维向量”误写为“每任务必须有 1000 个非空 logFC”。
Tahoe 的向量固定有 1000 个基因行，但 DESeq2 的部分 logFC 为 NaN。

旧 100-shard 输出中：

```text
min non-null logFC = 850
```

使用 `min_genes_per_task=850` 后，旧 100-shard 的 task 集合、record 集合和
所有正式指标均逐项零差异复现。因此 D1 沿用可复现门槛 850。

## 允许写

```text
SafeConf risk remained associated with prediction error in a sampled
Tahoe-100M chemical pseudobulk validation.
```

## 不允许写

```text
full Tahoe validation
held-out-drug generalization
all chemical perturbations
all prediction architectures
```

## 文件

| 文件 | 用途 |
|---|---|
| `RUN_STATUS.json` | 主运行规模、leakage、点估计 |
| `TAHOE_D1_POSTPROCESS_STATUS.json` | task-cluster gate |
| `tables/TAHOE_D1_FORMAL_SUMMARY.csv` | 正式结果和 CI |
| `tables/TAHOE_SHARD_SELECTION_SMOKE.csv` | 300 个候选 shard 清单 |
| `tables/D0_TAHOE_DRUG_OVERLAP_AUDIT.csv` | 与已有 chemical 数据的 drug overlap |

大型 arrays、完整 records 和 bootstrap draws 保存在：

```text
/home/yyf/safeconf_runtime/outputs/tahoe_300_shard_nested_formal_20260624
/home/yyf/safeconf_runtime/outputs/tahoe_300_shard_nested_taskcluster_20260624
```
