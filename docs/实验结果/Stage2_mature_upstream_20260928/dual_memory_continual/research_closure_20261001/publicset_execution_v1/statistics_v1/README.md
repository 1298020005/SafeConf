# PublicSet 完整五折真实统计

采用决定：**STOP_PUBLICSET_ARCHITECTURE_RETAIN_B1**；保留 `B1_HGB`。

这是 DEV/SEEN 新架构比较，不能作为未参与设计的独立确认。所有数值来自完整运行，未新增上游训练、未读取新 TEST 真值。

输入运行：`docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/full_v2_resume`；训练 PID：`1575043`；统计 PID：`1591704`。

主指标先分别计算三个神经种子的指标，在各 context × outer fold 内平均，再对有效分层等权平均。B0/B1/简单规则只有 seed0。主结果不使用平均风险分数排序；未把两个 Source upstream 作为独立生物样本。

配对 bootstrap 5000 次：Source 全体 575 gene 的同一个重采样同步应用到所有上游、种子和 outer folds；McFaline 377 gene 独立重采样。gene 重复次数被完整保留。60 个含并列、部分 top-k 和小分层的显式复制检查通过。

全臂共同任务覆盖：542–1808 条；各 scope 完整数量和缺失见 `COMMON_TASK_COVERAGE.csv`。

采用规则同时要求 ΔU20 ≥ 0.005、CI 下界 ≥ −0.005、≥60% 有效分层非负、≥80% 分层有效；macro AURC/error@10/20/50 相对恶化 ≤5%，高风险漏检率增量 ≤0.02。每条门限分别保存，不以“不显著”代替收益或稳定性。分层安全变化另表报告。跨域采用须在每个上游/域对 B0/B1/Magnitude/HistorySupport 及预登记 NearestControl/SameContextSupport/SameConditionSupport 都通过；DeepSets 集合贡献还须对 Pointwise 通过结构门。

U20 和低风险 error@ 的任务数均使用 ceil；并列按任务 ID。AURC沿用原定义（累计平均误差的平均，未归一）。漏检按重复任务副本的 top20 交集计算。抽样后不足80%计划分层有效的 macro draw 记 NA；置信区间为有效配对draw的百分位区间。

| Scope | Readout | Comparison | ΔU20 | 95% CI | Nonnegative strata | Gate |
|---|---|---|---:|---|---:|---|
| Source/TxPert_GAT | risk | B2_Pointwise−B1_HGB | 0.04017 | [-0.00436, 0.08673] | 65.0% | True |
| Source/TxPert_GAT | risk | B3_DeepSets−B1_HGB | 0.04213 | [-0.00620, 0.08720] | 75.0% | False |
| Source/TxPert_GAT | risk | B3_DeepSets−B2_Pointwise | 0.00195 | [-0.01488, 0.01423] | 55.0% | False |
| Source/TxPert_GAT | direct_risk | B2_Pointwise−B1_HGB | 0.22436 | [0.16222, 0.31368] | 80.0% | True |
| Source/TxPert_GAT | direct_risk | B3_DeepSets−B1_HGB | 0.23214 | [0.16298, 0.31485] | 90.0% | True |
| Source/TxPert_GAT | direct_risk | B3_DeepSets−B2_Pointwise | 0.00779 | [-0.01624, 0.01830] | 65.0% | False |
| Source/TxPert_Exphormer | risk | B2_Pointwise−B1_HGB | 0.04180 | [-0.00223, 0.08918] | 75.0% | True |
| Source/TxPert_Exphormer | risk | B3_DeepSets−B1_HGB | 0.03514 | [-0.00119, 0.08779] | 65.0% | True |
| Source/TxPert_Exphormer | risk | B3_DeepSets−B2_Pointwise | -0.00665 | [-0.01537, 0.01530] | 40.0% | False |
| Source/TxPert_Exphormer | direct_risk | B2_Pointwise−B1_HGB | 0.21075 | [0.13589, 0.27524] | 95.0% | True |
| Source/TxPert_Exphormer | direct_risk | B3_DeepSets−B1_HGB | 0.20463 | [0.13272, 0.27318] | 95.0% | True |
| Source/TxPert_Exphormer | direct_risk | B3_DeepSets−B2_Pointwise | -0.00612 | [-0.01973, 0.01552] | 35.0% | False |
| McFaline/DecoderOnly | risk | B2_Pointwise−B1_HGB | 0.04286 | [-0.01319, 0.05225] | 80.0% | False |
| McFaline/DecoderOnly | risk | B3_DeepSets−B1_HGB | 0.03778 | [-0.01644, 0.05322] | 73.3% | False |
| McFaline/DecoderOnly | risk | B3_DeepSets−B2_Pointwise | -0.00508 | [-0.00995, 0.00897] | 66.7% | False |
| McFaline/DecoderOnly | direct_risk | B2_Pointwise−B1_HGB | 0.01665 | [-0.03771, 0.04725] | 80.0% | False |
| McFaline/DecoderOnly | direct_risk | B3_DeepSets−B1_HGB | 0.02616 | [-0.03965, 0.05232] | 80.0% | False |
| McFaline/DecoderOnly | direct_risk | B3_DeepSets−B2_Pointwise | 0.00951 | [-0.01569, 0.01548] | 86.7% | False |

对 B0/Magnitude/HistorySupport 的全部比较及实际收益门见 `PAIRED_COMPARISONS.csv`；三个固定匹配参照另列 `STRONG_SIMPLE_COMPARISONS.csv`，规则不按评价结果选择。单种子结果见 `SEED_METRICS.csv`。McFaline cell-weighted B0−old-guide B0 仅为数据对齐变化，见 `MC_DATA_ALIGNMENT_COMPARISONS.csv`，不作为学习架构增益。

辅助读出依赖状态：`False`；仅 direct 通过的候选：`[]`。辅助读出结果不得替代主读出宣布成功。

本轮具体处理：停止把本轮 Pointwise/DeepSets 作为整体替换候选，继续使用 B1；已计算的凸组合重建诊断只解释生物重建空间。继续完成合法 Source 错误输入资格、同预算风险层和反馈工作。

文件：`MACRO_METRIC_INTERVALS.csv`、`STRATUM_METRICS.csv`、`SEED_METRICS.csv`、`PAIRED_COMPARISONS.csv`、`PAIRED_STRATUM_DELTAS.csv`、`STRONG_SIMPLE_COMPARISONS.csv`、`STRONG_SIMPLE_STRATUM_DELTAS.csv`、`COMMON_TASK_COVERAGE.csv`、`MC_DATA_ALIGNMENT_COMPARISONS.csv`、`DECISIONS.csv`、`DECISION.json`、`BOOTSTRAP_GENE_MULTIPLICITIES.npz`、`BOOTSTRAP_MACRO_DRAWS.npz`、`BOOTSTRAP_SEED_MACRO_DRAWS.npz`、三张 `FIG*.pdf/png`、`RUN_STATUS.json`。

复现：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/miniconda/bin/python /home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/analyze_safeconf_publicset_v1.py --run-dir docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/full_v2_resume --output-dir docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/statistics_v1 --replicates 5000 --threads 4 --max-seconds 600
```
