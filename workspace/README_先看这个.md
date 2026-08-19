# SafeConf 当前工作入口

> 2026-08-15 请先看：
> `docs/投稿准备/Q1_PATH_SEMANTIC_AUDIT_20260815.md`
> 和 `docs/实验结果/CURRENT_RESEARCH_STATUS_20260815.md`。

**更新日期：2026-08-14**

当前优先打开：

```text
docs/实验结果/CURRENT_RESEARCH_STATUS_20260814.md
docs/实验结果/NEXT_PHASE_PLAN_20260814.md
docs/实验结果/E201_txpert_multitarget_retraining_20260802/EXECUTION_CHECKPOINT_20260814.md
workspace/group_meeting_20260813/README_先看这个.md
```

老师原话和 7 月拆解仍在：

```text
workspace/group_meeting_20260709_MAINLINE_WHITE/周老师聊天记录_要求拆解与实验设计_20260709.md
```

`workspace/group_meeting_20260709_FINAL/` 和 `GATE_STATUS_20260714.md` 只作历史追溯。

## 目前最重要的结论

SafeConf 现阶段是一个**任务风险分诊协议**：它判断哪些预测任务更容易出错、更应优先复核或实验验证。

周老师这次聊天里的重点不是继续堆普通相关性结果，而是补更硬的实验 setting：

- 查清每个 score 的输入来源，尤其区分 `true effect magnitude` 和 `predicted magnitude`；
- 把当前 random held-out pair 扩成小矩阵 / 低覆盖度实验；
- 做整行、整列 holdout；
- 尝试跨数据集预测；
- gene perturbation 和 chemical perturbation 分开报告。

上述计算 setting 已完成。当前新增实验的优先级是 E143：至少一个全新细胞背景、扰动前冻结、双背景盲法 CRISPRi。E141 只有较弱通路支持；E142 蛋白正交和 E144 STRING/靶基因自身审计均未通过主 gate，不再用同一批真值调分数挽救。

## 目录纪律

- `workspace/` 根目录不再堆放中间版。
- 六月中间推演与 7 月 2 日旧汇报已转入 `/home/yyf/archive/safeconf/workspace_history/`。
- 归档内容仅供追溯，不代表当前结论。
