# E8b Step 2-4 audit report

日期：2026-06-15

状态：**PAUSED BEFORE PRIMARY ASSOCIATION**

本轮完成 Step 2（sciplex3 alias audit）、Step 3（实现与测试）和 Step 4
（Frangieh dry audit）。没有计算真实数据上的 Spearman、bootstrap CI、
permutation null 或 gate。

## 1. sciplex3 drug alias audit

完整候选表：

`E8b_SCIPLEX3_DRUG_ALIAS.csv`

统计：

| match_type | 数量 |
|---|---:|
| exact | 35 |
| alias | 25 |
| manual | 15 |
| unmatched | 0 |
| exact + alias 可用 | 60 |

`exact` 只允许大小写、空格和标点格式归一化；`alias` 只允许去掉 SafeConf
药名中的括号别名后匹配。

15 条 manual proposal 是：

| scPerturBench | SafeConf candidate |
|---|---|
| AZ | AZ 960 |
| Alendronate | Alendronate sodium trihydrate |
| Alvespimycin | Alvespimycin (17-DMAG) HCl |
| Epothilone | Epothilone A |
| Fasudil | Fasudil (HA-1077) HCl |
| Flavopiridol | Flavopiridol HCl |
| GSK | GSK J1 |
| NVPBSK805 | NVP-BSK805 2HCl |
| Obatoclax | Obatoclax Mesylate (GX15-070) |
| Quisinostat | Quisinostat (JNJ-26481585) 2HCl |
| Rucaparib | Rucaparib (AG-014699,PF-01367338) phosphate |
| SRT1720 | SRT1720 HCl |
| Tofacitinib | Tofacitinib (CP-690550) Citrate |
| Triamcinolone | Triamcinolone Acetonide |
| Tubastatin | Tubastatin A HCl |

由于 manual 数量为 15，超过预注册上限 10，所有 manual rows 当前均未批准，
也不会进入 sciplex3 sensitivity。若 Claude 不修改冻结规则，最终只使用 60 条
exact + alias 映射。

## 2. Implementation and tests

新增：

- `safetrans_confidence/cli/run_e8b_association.py`
- `safetrans_confidence/tests/test_e8b_association.py`

CLI 将 `alias-audit`、`dry-audit` 与后续 association functions 分开。
`dry-audit` 不调用任何相关、bootstrap 或 permutation 函数。

新增合成测试覆盖：

- test rows 不进入最终 task-risk 聚合；
- risk/error 方向；
- bootstrap CI；
- permutation null；
- 74 perturbation join；
- available-seed median aggregation。

测试环境：

```text
/home/yyf/.conda/envs/scgpt_env/bin/python
```

结果：

```text
68 passed, 14 warnings
```

其中同时修正了一个既有回归测试的 join key：family-level 结果具有相同的
`dataset_name=ALL`，原测试遗漏 `dataset_family` 后产生笛卡尔积假失败。
该修正没有修改 frozen scorer、公式或已有结果文件。

## 3. Frangieh dry audit

机器可读结果：

`E8b_FRANGIEH_DRY_AUDIT.json`

| 检查项 | 结果 |
|---|---:|
| benchmark perturbations | 74 |
| SafeConf scored perturbations | 211 |
| joined perturbations | 74/74 |
| joined non-null scores | 74/74 |
| seed coverage = 1 | 57 perturbations |
| seed coverage = 2 | 15 perturbations |
| seed coverage = 3 | 2 perturbations |
| Nstimulated missing rows | 0 |
| Nstimulated unique per perturbation | yes |
| association computed | no |

`Nstimulated` 来源冻结为 scPerturBench aggregate CSV 的 benchmark task
metadata。它不是 SafeConf feature matrix 字段，也没有从 H5AD 重新计算。

## 4. Pause decision

Step 2-4 已完成。按 Claude 的硬性暂停要求，未启动 Step 5。下一步只有在
Claude 验收 alias、测试和 dry audit 后，才运行 Frangieh 正式关联分析。
