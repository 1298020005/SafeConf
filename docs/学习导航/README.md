# SafeConf 学习导航

更新时间：2026-08-14

事实基线：E181–E186 完成注册模型家族证书、外部评价、理论定位和完整性审计；
E189 补齐周老师要求的小矩阵与行列缺失，E190 完成 Adamson→Replogle 真实预测器
迁移，E191 完成有限复核预算收益，E192 完成 RPE1 锁定确认并返回 `ABSTAIN`，
E193/E194 完成多几何证书和 family 治理压力测试。E198–E200 已完成公开 TxPert
模型的协议校准、K562 未见扰动和单一完整背景留出；E201 正在进行四背景、四种子盲
训练，尚未释放 target 真值。当前汇报状态以
`docs/实验结果/CURRENT_RESEARCH_STATUS_20260814.md` 为准；E181–E197 历史证据线
以 `docs/实验结果/GATE_STATUS_20260729.md` 为准。

这里是 SafeConf 给人和外部 Agent 共用的学习目录。根目录唯一的 Agent 总入口是：

```text
START_HERE_FOR_AGENTS.md
```

不要从 `docs/实验结果/` 随机挑一个编号开始读。SafeConf 经历了多轮实验升级，早期文件在当时是正确的，但可能已经被后续正式实验更新。

## 本目录的五份文件

| 文件 | 解决的问题 |
|---|---|
| [01_目录与权威等级.md](./01_目录与权威等级.md) | 仓库每个目录放什么，发生冲突时相信谁 |
| [02_当前证据链与实验谱系.md](./02_当前证据链与实验谱系.md) | 当前做到什么程度，E1 到 E132 怎样分层理解 |
| [03_Agent学习任务与验收.md](./03_Agent学习任务与验收.md) | Qoder、Gemini、Claude 或其他 Agent 应该读什么、交什么 |
| [04_论文创作接力说明.md](./04_论文创作接力说明.md) | 当前论文正文缺什么，下一位 Agent 如何继续写而不混淆旧结果 |
| [05_E133_E140方向风险与七数据更新.md](./05_E133_E140方向风险与七数据更新.md) | E133–E140 最新七数据、Systema 方向误差和第七数据确认；覆盖 02 中的六数据口径 |
| [06_E141_E144机制审计与前瞻湿实验.md](./06_E141_E144机制审计与前瞻湿实验.md) | E141 通路弱正结果、E142/E144 失败边界和 E143 前瞻湿实验交接 |

最新汇报结论直接从 [当前研究状态](../实验结果/CURRENT_RESEARCH_STATUS_20260814.md)、
[下一部分安排](../实验结果/NEXT_PHASE_PLAN_20260814.md)、
[当前 gate（E181–E197）](../实验结果/GATE_STATUS_20260729.md)、
[E194 family 治理](../实验结果/E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md)、
[E193 多几何证书](../实验结果/E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md)、
[E192 RPE1 锁定确认](../实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md)、
[E191 决策收益](../实验结果/E191_certificate_decision_utility_20260729/reports/E191_INTERPRETATION.md)、
[E190 跨研究迁移](../实验结果/E190_adamson_to_replogle_direct_transfer_20260729/final_evaluation/reports/E190_INTERPRETATION.md)、
[E189 笛卡尔缺失实验](../实验结果/E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md)
和 E186–E181 主证书证据进入。

## 按目的选择入口

| 目的 | 第一入口 | 随后阅读 |
|---|---|---|
| 10 分钟了解项目 | `START_HERE_FOR_AGENTS.md` | `GATE_STATUS_20260729.md`、E194、E193、E192、E191、E190、E189、E186、E181 |
| 从零学习 SafeConf | `docs/SafeConf_完整项目讲解/index.html` | 本目录 01、02 |
| 审核实验是否可信 | 当前 gate、E185、E183、E182、E181，再读 02 | E180、E179、E178、E177、E176、E173 |
| 接着做实验 | 当前 gate 的“投稿前剩余硬任务” | 不再用已开真值数据调公式；E143 物理实验另行交接 |
| 接着写论文 | 当前 gate、E184、E183、E182、E181 和 04 | E176–E185；旧排序降为诊断 |
| 修改代码或复跑 | `code/README.md` | `docs/代码设计/PROTOCOL.md`、对应 E 编号报告 |
| 准备组会 | `workspace/group_meeting_20260709_MAINLINE_WHITE/README_先看这个.md` | 当前 gate 和项目总账 |

## 当前事实入口

以下文件发生冲突时，按顺序以前面的为准：

1. `docs/实验结果/CURRENT_RESEARCH_STATUS_20260813.md`（当前汇报口径；明确标记
   E201 为未开真值的运行中实验）
2. E198–E200 的正式 report 和 table；E201 的训练冻结、盲视图审计和执行检查点
3. `docs/实验结果/GATE_STATUS_20260729.md`（E181–E197 历史主证据）
4. E194、E193、E192、E191、E190、E189 的正式 report 和 table
5. `docs/实验结果/E186_presubmission_integrity_audit_20260724/reports/E186_REPORT.md`
6. `docs/实验结果/E185_minimal_release_validation_20260724/reports/E185_REPORT.md`
7. `docs/实验结果/E184_direct_competitor_positioning_20260724/reports/E184_REPORT.md`
8. E183–E173 的正式报告
9. 历史投稿总账和具体实验目录

`agents/`、`workspace/`、早期教程和旧草稿不能覆盖上面的事实层。

## 维护规则

- 每次正式 gate 更新时，同时更新本目录、`START_HERE_FOR_AGENTS.md` 和 `START_HERE_FOR_GPT.md`。
- 新 Agent 的原始意见放 `agents/<agent-name>/`，经复核后才能写入 `docs/`。
- 学习资料可以更通俗，但数字、数据集数量和实验状态必须与当前 gate 一致。
- 论文草稿引用实验时必须写出证据路径，不能只写“已有实验表明”。
- `/home/yyf/archive/safeconf/`、`待整理/` 和私人聊天只用于追溯，不进入公开事实链。
