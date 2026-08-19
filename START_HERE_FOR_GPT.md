# SafeConf 给 AI / 网页 GPT 的第一入口

你现在看到的是 `/home/yyf/proj`，这是用户唯一想看的服务器主项目目录。

当前分支：

```text
exp/task-risk-audit-20260611
```

一句话定位：

> SafeConf 给已有单细胞扰动预测做可靠性审计：注册预测家族提供两个确定性误差下界，独立校准得到的参考质心上界可严格搬移到正式家族。

## 先读顺序

| 顺序 | 文件/目录 | 用途 |
|---|---|---|
| 0 | `docs/实验结果/CURRENT_RESEARCH_STATUS_20260815.md` | 当前 E201 进度与一区口径 |
| 0b | `docs/实验结果/NEXT_PHASE_PLAN_20260814.md` | 历史审核和下一部分完整顺序 |
| 0a | `REMOTE_CODEX_INIT.md` | 远程电脑 / 新 Codex 初始化说明 |
| 1 | `START_HERE_FOR_AGENTS.md` | 给 Qoder / Gemini / Claude / 新 Codex 的完整学习地图 |
| 2 | `docs/学习导航/README.md` | 目录权威、实验谱系和论文接力 |
| 3 | `INDEX.md` | 项目总入口 |
| 4 | `docs/投稿准备/录用判断与项目总账_20260713/index.html` | 周老师问题、当前证据、投稿定位和录用边界 |
| 5 | `docs/实验结果/GATE_STATUS_20260729.md` | 当前实验事实总入口 |
| 6 | `docs/实验结果/E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md` | family 治理与防操纵边界 |
| 7 | `docs/实验结果/E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md` | 多几何证书与方向排序边界 |
| 8 | `docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md` | RPE1 锁定确认与 ABSTAIN 裁决 |
| 9 | `docs/实验结果/E191_certificate_decision_utility_20260729/reports/E191_INTERPRETATION.md` | 有限复核预算收益与失败边界 |
| 10 | `docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729/final_evaluation/reports/E190_INTERPRETATION.md` | 基因侧真实预测器跨研究迁移 |
| 11 | `docs/实验结果/E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md` | 小矩阵和四类缺失 setting |
| 12 | `docs/实验结果/E186_presubmission_integrity_audit_20260724/reports/E186_REPORT.md` | 投稿前证据链、口径、复现和远端完整性审计 |
| 13 | `docs/实验结果/E185_minimal_release_validation_20260724/reports/E185_REPORT.md` | 一条命令重算主数字与发布物完整性 |
| 14 | `docs/实验结果/E184_direct_competitor_positioning_20260724/reports/E184_REPORT.md` | 直接竞品、经典理论来源和贡献边界 |
| 15 | `docs/实验结果/E183_all_study_family_synthesis_20260724/reports/E183_SYNTHESIS_REPORT.md` | 四项研究合并审计与有限校准波动解释 |
| 16 | `docs/实验结果/E182_gse225807_registered_family_20260724/final_evaluation/reports/E182_FINAL_REPORT.md` | GSE225807 事前冻结评价 |
| 17 | `docs/实验结果/E181_registered_family_hilbert_certificate_20260724/reports/E181_REPORT.md` | 注册家族双侧证书与五张总图 |
| 18 | `workspace/README_先看这个.md` | 当前科研推进入口 |
| 19 | `workspace/group_meeting_20260709_MAINLINE_WHITE/周老师聊天记录_要求拆解与实验设计_20260709.md` | 老师原话与问题拆解 |
| 20 | `docs/学习导航/05_E133_E140方向风险与七数据更新.md` | 历史七数据与双风险头更新 |
| 21 | `agents/README.md` | 多 AI 协作规则；原始代理意见不得覆盖当前 gate |

## 目录边界

| 目录 | 怎么理解 |
|---|---|
| `code/` | SafeConf 正式代码 |
| `docs/` | 稳定文档、实验结果、小白学习资料 |
| `docs/学习导航/` | 给人和 Agent 的当前学习路线、证据谱系和论文接力 |
| `agents/` | 当前 AI 协作状态和原始输出 |
| `workspace/` | 当前日常工作区 |
| `tools/` | 下载、复跑、资源清单、环境脚本 |
| `runtime/` | 临时输出入口说明 |
| `/home/yyf/archive/safeconf/` | 集中历史库，不作为当前结论 |

## 当前结论边界

- E176 在四个完整留出供体、640 个评价靶点上得到下界零违反与 90.47% 常数上界靶点簇覆盖；它仍是同一研究内确认。
- E177 在独立公开处理数据上得到下界零违例与 88% 靶点簇覆盖；置信区间包含 90%，但点估计未达目标，只能写边界结果。
- E178 表明 scGPT 与 GEARS 的误差高度同步，当前主要是共享任务难度；模型分歧不能指出哪个模型错。
- E180 在 XuCao 的 73 个 guide 任务上再次得到下界零违反，常数上界覆盖 27/27 靶点；ExtraTrees 自适应上界更宽，已停止。
- E181 将 scGPT 五种子和 GEARS 五种子组成注册家族，三套数据合计 2,393 个任务的家族 RMS 与最坏成员下界均为零违反。
- E182 在此前未使用的 GSE225807 上完成完整事前冻结评价：40 个 guide 任务的两类下界仍为零违反；靶点簇上界覆盖 16/20，未达到注册的 17/20 门槛，必须写成 FAIL。
- E183 统一审计四项研究的 2,433 个任务和 737 个靶点簇：下界零违反，家族上界靶点簇覆盖 666/737=90.37%。这是描述性合并，不能把 E182 改判为通过。
- E185 用标准库验证器从提交的任务级证书重算全部主数字，12,033 项检查零失败；它不替代端到端模型重训。
- E189 统一比较小矩阵、随机缺一格、整行、整列和双未见。random pair 为正相关，整行接近零，双未见为负相关；随机拆分确实偏乐观。
- E190 在 Adamson 训练并直接预测 Replogle 的 47 个共同基因、692 个任务。family 相对 zero 的基因簇 95% CI 跨 0，必须保留为边界结果。
- E191 在固定 10%/20%/30% 复核预算下检验实际收益。diversity 对 magnitude 为 8 胜 8 负，double unseen 排序会产生负收益。
- E192 在未参与排序开发的 RPE1 上得到 20% 预算 utility=0.696，但相关区间跨 0；按事前双 gate 保持 `ABSTAIN`，不能事后改阈值。
- E193 在 absolute RMSE、effect cosine 和 effect Pearson 中保持下界零违例，但
  方向型经验排序没有跨细胞系运输。
- E194 的 310 个 family 场景与 492 项不变量检查说明：证书属于预注册加权 family；
  复制或加入成员会改变对象，必须按 prediction hash、lineage 和 architecture 治理。
- E168/E172 已否定 fixed SafeConf 稳定超过 predicted magnitude 的旧主张。当前论文主线是双边证书与 fail-closed 审计，排序只是诊断。
- Frozen v0.2 是可解释的固定规则；McFarland 仍是 frozen v0.2 的 failure boundary（失败边界）。
- Learned reliability layer（学习型可靠性层）是补强实验，不能写成 frozen v0.2 本身成功。
- E8b 是 external benchmark method-error association（外部 benchmark 方法误差关联），不是 27 个模型的逐预测 SafeConf 验证。
- 不要声称 SafeConf 已经证明对 GEARS、CPA、scGPT 等所有深度模型普适。
- E108/E112 已经完成正式 context-aware scGPT–GEARS 验证；E104 和 E25–E32 中“正式双模型尚未完成”的说法属于历史状态。
- E140 支持七套正式数据 absolute RMSE 总体超过 disagreement；相对 magnitude 的区间跨 0，Nadig 是明确反例。
- E141 只有较弱通路误差证据；E142 蛋白正交 gate 失败，E144 STRING/靶基因自身 gate 失败。
- E143 的计算、功效、盲法和模板完成，物理湿实验尚未执行；不能提前写成验证成功。
- E139 支持冻结的 Directional-SafeConf 在 Nadig 第七数据确认通过；它不能覆盖原 SafeConf 的方向负结果或 Nadig absolute 负结果。
- E132 只支持完整 risk–coverage 区间相对 disagreement 的稳定改善；固定 top-20% 错误捕获增益未闭环。
- E111 表明风险信号主要对应 GEARS 误差，不能改写成模型无关的通用置信度。
- E114 的上界覆盖充分但偏保守；E117 紧化后覆盖失败，不能写成无条件、紧致的误差保证。
- 证据达到投稿竞争力不等于期刊百分之百录用。当前完整解释见 `docs/投稿准备/录用判断与项目总账_20260713/`。
- `/home/yyf/archive/safeconf/` 里的历史材料可以查证，但不能直接当当前结论。

## 回答风格

用户正在学习项目。请用通俗中文解释。

每个英文词第一次出现时，请写中文含义，并说明它在 SafeConf 里的作用。

优先用：

```text
流程图
表格
具体例子
一句话结论
```

少用空泛口号。
