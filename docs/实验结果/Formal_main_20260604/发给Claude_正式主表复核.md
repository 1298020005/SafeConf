# 发给 Claude：SafeConf 7 主表 formal audit 复核请求

请先保持客观，不要默认同意用户、Codex 或 Qoder。  
你的角色是：审稿式方案评估 + 下一步计划设计，不负责跑服务器实验。

## 0. 你先读这些文件

如果你在 Windows 本地仓库，请先 `git pull`，然后读：

```text
docs/实验结果/Formal_main_20260604/README_先看这个.md
docs/实验结果/Formal_main_20260604/tables/FORMAL_MAIN_TABLE.csv
docs/实验结果/Formal_main_20260604/tables/FORMAL_PER_FOLD_RHO.csv
docs/实验结果/Formal_main_20260604/tables/FORMAL_PER_PREDICTOR_RHO.csv
docs/03-审计报告/PhaseC-数据下载储备/TAHOE_CURRENT_STATUS_20260604.md
agents/STATE.md
agents/TASKS.md
agents/LOG.md
```

不要只看这一份摘要。  
请以 CSV 表为准，不要凭叙事判断。

## 1. 当前 git 版本变化

最近两个关键提交：

```text
a966887 results: add formal main audit summary
6c92080 safeconf: launch formal main audit
```

`6c92080` 做了：

- 统一 `chem_robust（药物稳健线）/ gene_main（基因主线）` 配置。
- 新增 McFarland drug-only 过滤，只保留 `perturbation_type == drug`。
- 新增 formal audit 脚本，输出 aligned rho、partial rho、magnitude-only rho、risk-coverage@80%、per-fold rho、Bootstrap 95% CI。

`a966887` 做了：

- 将 7 主表正式结果同步到 `docs/实验结果/Formal_main_20260604/`。
- 写入 Tahoe 当前状态：已下载约 72G，已有 859 个 pseudobulk parquet 分片，metadata 可读。
- 更新 `agents/STATE.md / TASKS.md / LOG.md`。

## 2. 当前正式主表结果

主 score：

```text
protocol_v0_2_family_confidence
```

主表：

| 数据集 | 线 | n test | aligned rho | partial rho | magnitude-only rho | RC@80% |
|---|---|---:|---:|---:|---:|---:|
| CuiHacohen2023 | gene_main | 2506 | 0.445 | 0.328 | 0.736 | 21.59% |
| Frangieh | gene_main | 1266 | 0.583 | 0.474 | 0.797 | 5.03% |
| Lara exvivo | gene_main | 662 | 0.561 | 0.430 | 0.513 | 56.12% |
| Lara invivo | gene_main | 780 | 0.413 | 0.358 | 0.639 | 12.83% |
| McFarland drug-only | chem_robust | 2326 | -0.086 | -0.061 | 0.795 | 3.95% |
| SantinhaPlatt2023 | chem_robust | 566 | 0.206 | 0.224 | 0.824 | 2.08% |
| Srivatsan sciplex3 | chem_robust | 1128 | 0.428 | 0.629 | 0.740 | 15.10% |

当前 gate：

- aligned rho > 0.20：6/7
- partial rho > 0.10：6/7
- risk-coverage@80% 改善：7/7
- 失败边界：McFarland drug-only

## 3. 当前我最担心的地方

1. `magnitude-only rho（只看效应大小的相关）` 在多数数据集很高。  
   这说明 effect magnitude confounding（效应大小混杂）仍然是最大风险。

2. McFarland drug-only 是大药物数据集，但主 score 为负。  
   这可能意味着：
   - chem_robust 公式不适合 McFarland；
   - McFarland 的 context（背景）/ dose（剂量）结构和其他药物数据不同；
   - drug-only 过滤后仍有 dose/time 等混杂；
   - 或者当前 task 定义在 McFarland 上不成立。

3. gene_main 线目前比 chem_robust 线更稳。  
   如果论文主张过宽，会被审稿人抓住 McFarland。

4. Tahoe 已下载 72G，可读字段显示有 `drug`、`cell_line`、`log2FoldChange`，但还没有真正完成 SafeConf task adapter。

## 4. 请你客观回答的问题

请按编号回答，不要泛泛而谈。

### Q1. 当前结果能不能支撑继续做 SafeConf？

请分三档判断：

- A. 可以冲主线论文；
- B. 可以做，但必须收缩 claim（主张）；
- C. 不建议继续，应该改题。

请给理由，不要只说“看起来不错”。

### Q2. 这套结果离“稳二区 / 冲一区”分别还差什么？

请明确列：

- 稳二区最低还缺什么；
- 冲一区必须补什么；
- 哪些东西是加分但非必需。

### Q3. McFarland 失败应该怎么诊断？

请给 Codex 一个可执行诊断清单。  
至少包括：

- dose（剂量）是否混杂；
- time（时间）是否混杂；
- cell_line（细胞系）支持度是否不均；
- drug（药物）是否太少或太偏；
- V0 / ContextSim / model_disagreement 单独表现；
- 是否应该把 McFarland 改成 failure boundary（失败边界）而不是硬救。

### Q4. chem_robust 公式要不要改？

当前 chem_robust 是：

```text
log_support - model_disagreement
```

请判断：

- 是否允许为了 McFarland 改公式；
- 如果允许，应该在什么 split 上调；
- 如果不允许，McFarland 应该怎么写进论文。

注意：不要让 Codex 在 test 上调参。

### Q5. Tahoe 下一步应该做什么？

当前 Tahoe：

- 下载约 72G；
- `obs_metadata.parquet` 有 100,648,790 行；
- 字段包括 `drug`、`cell_line`、`plate`；
- pseudobulk 分片含 `gene_name`、`log2FoldChange`、`pvalue`、`padj`、`drug`、`concentration`；
- 后段分片有 ERR，但已有 859 个分片。

请判断：

- 是否值得立即写 Tahoe adapter；
- Tahoe 应该作为主表、外部验证，还是 supplement；
- 需要先做哪些 eligibility audit（可用性审计）。

### Q6. 下一步 Codex 应该做什么？

请给一个 3-5 步计划。  
每一步要有明确输出文件，例如：

```text
McFarland_failure_diagnosis.csv
Tahoe_eligibility_audit.csv
FORMAL_MAIN_DECISION.md
```

不要只写“继续优化”。

### Q7. 下一步用户应该做什么？

用户是单细胞小白，不应该手动改代码。  
请明确告诉用户：

- 需要把哪份文件发给你；
- 是否需要问导师；
- 是否需要继续下载数据；
- 是否需要暂停等 Codex 结果。

## 5. 请你不要做的事

- 不要说“一区稳了”。
- 不要只看 6/7 过线而忽略 McFarland。
- 不要把 learned_risk_score 当主方法标题。
- 不要建议盲目下载更多 scPerturb 数据。
- 不要让用户手动调代码。
- 不要只复述 Codex 的结论；必须挑刺。

## 6. 期望你输出的格式

请用下面格式回复：

```text
一、总判断
二、当前结果能说什么 / 不能说什么
三、McFarland 失败诊断计划
四、Tahoe 下一步计划
五、给 Codex 的执行清单
六、给用户的一句话建议
```

请尽量通俗，但每个关键英文第一次出现要带中文解释。
