# SafeConf 外部 Agent 总入口

更新时间：2026-08-13

当前公开事实基线：E181–E186 完成注册模型家族证书、外部评价、理论定位和完整性
审计；E189 完成小矩阵及随机/整行/整列/双未见同合同实验，E190 完成
Adamson→Replogle 真实预测器迁移，E191 完成有限复核预算收益，E192 完成未参与
排序开发的 RPE1 锁定确认并按预注册 gate 返回 `ABSTAIN`，E193/E194 完成多几何
证书与 family 治理压力测试。最终数值以 `GATE_STATUS_20260729.md` 为准。

当前分支：`exp/task-risk-audit-20260611`

## 2026-08-12 的当前状态

E201 正在用公开 TxPert STRING-GAT 执行四个细胞背景、四个种子的完整背景留出。
它目前仍是盲训练阶段：目标扰动真值没有打开，不能写成新结论。先读
`docs/项目交接_20260812.md` 与
`docs/实验结果/E201_txpert_multitarget_retraining_20260802/EXECUTION_CHECKPOINT_20260813.md`，
再阅读下方的历史证据链。

当前事实入口：`docs/实验结果/CURRENT_RESEARCH_STATUS_20260815.md`。
一区路径与聊天审核：`docs/投稿准备/Q1_PATH_SEMANTIC_AUDIT_20260815.md`。
不要把 E201 写成已完成结果。

本文件是 Qoder、Gemini、Claude、网页 GPT、新 Codex 和其他 Agent 的唯一根入口。不要从旧实验目录、Agent 原始输出或 archive 随机开始学习。

## 0. 一句话定位

SafeConf 不是新的单细胞扰动预测模型。当前主线也不再是“得到一个比 magnitude 更强的排序分数”。

```text
scGPT / GEARS 多种子先预测扰动效应
                 ↓
注册预测家族给出两类确定性误差下界
独立校准数据给参考质心建立 conformal 上界
                 ↓
通过质心距离严格搬移，输出双侧证书
```

最稳的中文说法：

> SafeConf 是单细胞扰动预测后的可靠性审计框架：它冻结多个模型和随机种子的预测，用模型家族的几何关系给出误差下界，再把独立校准得到的参考质心上界严格搬移到正式家族。

## 1. 第一次进入必须读什么

按顺序读，不要跳：

| 顺序 | 文件 | 目的 |
|---:|---|---|
| 0 | `docs/项目交接_20260812.md` | 当前服务器状态、Git/数据盘边界、远程接手方式 |
| 0a | `docs/实验结果/E201_txpert_multitarget_retraining_20260802/EXECUTION_CHECKPOINT_20260812.md` | E201 盲训练实时检查点 |
| 1 | `docs/学习导航/README.md` | 选择学习路线 |
| 2 | `docs/学习导航/01_目录与权威等级.md` | 知道冲突时相信谁 |
| 3 | `docs/实验结果/GATE_STATUS_20260729.md` | 当前最高事实入口 |
| 4 | `docs/实验结果/E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md` | family 成员、权重和防操纵边界 |
| 5 | `docs/实验结果/E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md` | RMSE/cosine/Pearson 多几何证书 |
| 6 | `docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md` | RPE1 锁定确认与 ABSTAIN 裁决 |
| 7 | `docs/实验结果/E191_certificate_decision_utility_20260729/reports/E191_INTERPRETATION.md` | 固定复核预算下的实际收益与失败 setting |
| 8 | `docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729/final_evaluation/reports/E190_INTERPRETATION.md` | Adamson→Replogle 真实预测器跨研究迁移 |
| 9 | `docs/实验结果/E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md` | 小矩阵、随机、整行、整列和双未见 |
| 10 | `docs/实验结果/E186_presubmission_integrity_audit_20260724/reports/E186_REPORT.md` | 投稿前证据链、口径、复现和远端完整性审计 |
| 11 | `docs/实验结果/E185_minimal_release_validation_20260724/reports/E185_REPORT.md` | 一条命令重算主数字与发布物完整性 |
| 12 | `docs/实验结果/E184_direct_competitor_positioning_20260724/reports/E184_REPORT.md` | 直接竞品、经典理论来源和可写贡献边界 |
| 13 | `docs/实验结果/E183_all_study_family_synthesis_20260724/reports/E183_SYNTHESIS_REPORT.md` | 四项研究统一审计与有限校准波动解释 |
| 14 | `docs/实验结果/E182_gse225807_registered_family_20260724/final_evaluation/reports/E182_FINAL_REPORT.md` | GSE225807 事前冻结评价与诚实 FAIL |
| 15 | `docs/实验结果/E181_registered_family_hilbert_certificate_20260724/reports/E181_REPORT.md` | 注册家族方法、主表和五张总图 |
| 16 | `docs/实验结果/E178_crossstudy_bilateral_certificate_audit_20260722/reports/E178_REPORT.md` | 老师问题与模型特异性 |
| 17 | `agents/README.md` | 多 Agent 协作规则 |

准备继续写论文，再读：

```text
docs/学习导航/04_论文创作接力说明.md
```

准备修改代码或复跑，再读：

```text
code/README.md
docs/代码设计/PROTOCOL.md
```

## 2. 当前项目事实

```text
E176：4 个完整留出供体，640 个评价靶点，1,920 个任务
E177：独立公开处理数据，50 个评价靶点，400 个任务
E180：XuCao 独立 CRISPRi，27 个评价靶点，73 个 guide 任务
E182：GSE225807 独立 RBP CRISPRi，20 个评价靶点，40 个 guide 任务
E183 合并：737 个靶点簇，2,433 个任务
四项研究家族 RMS / 最坏成员下界违反：0 / 0
E189：4 donor × 4 支持量 × 4 缺失 setting，13,440 个任务实例
E190：Adamson→Replogle，47 个共同基因、692 个跨研究任务
E191：10% / 20% / 30% 固定复核预算，210 个注册比较行
E192：Adamson K562→Replogle RPE1，21 个基因、175 个锁定任务
E193：867 个任务 × 3 种几何，确定性下界 0 违例
E194：310 个 family 场景、134,385 条逐任务记录、492/492 治理检查通过
```

- E176 常数上界靶点簇同时覆盖：579/640=90.47%，达到预设 90% 目标。
- E177 外部靶点簇覆盖：44/50=88.0%，95% CI 包含 90%，属于边界结果。
- E180 常数上界靶点簇覆盖：27/27=100%；学习型 ExtraTrees 上界反而更宽，该支线已停止。
- E181 的 10 模型家族在三套数据全部任务上均比两个架构均值获得更高的家族下界紧致度。
- E182 再次得到两类确定性下界零违反；上界只覆盖 16/20 个靶点，低于注册的 17/20 门槛，结论是 FAIL。
- E183 保留 E182 的失败裁决。四项研究合并后，家族上界任务覆盖 2,331/2,433=95.81%，靶点簇同时覆盖 666/737=90.37%；后一个数是描述性汇总，不是新 conformal 保证。
- E185 用仓库内证书表重算上述全部主数字，12,033 项检查零失败；它是发布物复现，不是端到端重训。
- E189 的 random pair 有正相关，整行接近零，双未见为负相关；随机缺一格确实会高估可用性。
- E190 中 diversity 与 family error 的相关为 0.424，但 predicted magnitude 为 0.420；分歧有信号，没有独特优势。
- E191 中 16 个 `support×setting` 层对 magnitude 为 8 胜 8 负；双未见排序产生负收益，必须返回“排序未验证”。
- E192 的 20% 预算 diversity utility 为 0.696，但相关 95% CI 跨 0，按冻结双 gate 返回 `ABSTAIN`；predicted magnitude 的 utility 为 0.725。
- E193 的数学证书在 RMSE、effect cosine 和 effect Pearson 中均成立，但方向排序
  不从 K562 运输到 RPE1。
- E194 证明证书严格属于预注册加权 family；复制成员只有在 lineage/architecture
  权重治理下才不改变 A0，合成成员可在质心不变时人为放大 diversity。
- E178 中 scGPT 与 GEARS 的误差相关为 0.975/0.992，主要反映共享任务难度。
- E168 与 E172 两次全新靶点实验都没有确认 fixed SafeConf 相对 predicted magnitude 的排序增量，该主张已停止。
- Directional-SafeConf 的 Nadig 第七数据确认仍是有效的独立支线，不能与 absolute-RMSE 证书混成一个分数。

## 3. 五类对象不要混称

| 名称 | 含义 | 当前地位 |
|---|---|---|
| calibrated SafeConf | fold 内 source validation 校准后的 pair risk | 七数据 absolute 主结果；对 magnitude 区间跨 0 |
| frozen SafeConf v0.2 | 预先固定、可解释的规则 | 固定协议和稳健性对照 |
| Directional-SafeConf | 四部署特征 Ridge 风险头 | 六数据探索后冻结，Nadig 第七数据确认通过 |
| learned router / reliability layer | 后续学习型风险模型 | 补充或负结果，不能改写 frozen 成功率 |
| pair bilateral certificate | `d(p1,p2)/2` 下界 + split-conformal 上界 | E173–E178 方法形成 |
| registered-family certificate | 预注册加权 family 的多样性/直径下界 + 可搬移 conformal 上界 | E181 定义；E182 前瞻检验；E183 统一审计；E193 多几何；E194 治理 |

任何总结都要说明使用的是哪一个版本。

## 4. 当前主证据怎么组织

| 问题 | 主证据 | 当前结论 |
|---|---|---|
| 注册家族下界是否成立 | E181–E183 | 四项研究 2,433 个任务的家族 RMS 和最坏成员下界均零违反 |
| 校准上界是否覆盖 | E176、E177、E180、E182 | 前三项为 90.47%、88%、100%；E182 为 16/20，注册门槛 FAIL |
| 四项结果合起来怎样 | E183 | 666/737=90.37%，只作描述性合并，不替代各研究 gate |
| 学习型上界能否更紧 | E179、E180 | 开发增益很小，独立确认失败，正式停止 |
| 分数针对单模型还是任务 | E178 | 两模型误差高度同步；当前证书属于模型对层面 |
| 七数据 absolute 是否有效 | E140、E168、E172 | 历史总体超过 disagreement；全新靶点未确认对 magnitude 增量 |
| direction 是否有效 | E135、E139 | 六数据探索后冻结，Nadig 第七数据确认通过 |
| 是否能帮助分诊 | E132 | 完整 risk–coverage 区间改善；固定 top-20% 收益有限 |
| 风险识别谁的错误 | E111 | GEARS 信号明显强于 scGPT，不能写模型无关 |
| 为什么有些任务更难 | E116 | 背景新颖度是最清楚的失效来源，通路富集未显著 |
| 是否有误差上界 | E114、E117 | 保守上界覆盖充分；紧化后覆盖失败 |
| chemical 是否成功 | E118 | 正式 CPA chemical 合同中 magnitude 更强 |
| 学习型路由器是否确认 | E126、E130 | 未通过预设门槛，停止事后调参 |

## 5. 必须保留的边界

- Santinha 是弱复制；Shifrut 和 Nadig absolute 没有超过 magnitude；Tian 含负 fold。
- 原 SafeConf 的方向误差相关接近 0；Directional-SafeConf 是单独风险头。
- Tian 的 context 是技术批次，不是新的生物细胞类型。
- SafeConf 对 GEARS 错误更敏感，对 scGPT 较弱。
- chemical 不能与 gene 主结果混成跨模态成功。
- 早期 E114 的 90% split-conformal 边际覆盖偏高且上界较宽；最新主结果应引用 E176/E177 的靶点簇同时覆盖。
- 高风险基因通路没有通过预设显著性门槛。
- 具备投稿竞争力不等于任何期刊保证录用。
- E177 的 `gem_group` 只是技术组，不能写成供体、患者或生物学背景。
- 两模型误差高度同步不代表预测可靠，只说明它们面对相似的任务难度。
- 四项研究的覆盖率必须先分别报告；E183 的描述性合计不能生成新的 conformal 保证。
- E181 是打开真值后的方法整合，不得写成新的事前确认。
- E182 的 16/20 必须写成注册门槛 FAIL；有限样本解释不能把它改判为通过。
- E183 在 E182 解封后完成，只能统一审计，不能冒充前瞻确认。

## 6. 目录怎么理解

| 目录 | 作用 |
|---|---|
| `docs/学习导航/` | 给人和 Agent 的统一学习体系 |
| `docs/实验结果/` | 正式结果和当前事实 |
| `docs/投稿准备/` | 项目总账、投稿定位和录用边界 |
| `docs/SafeConf_完整项目讲解/` | 从零到完整项目的十二章教程 |
| `docs/小白科普/` | 专题解释和真实数据样例 |
| `docs/代码设计/` | 协议、合同和系统设计 |
| `code/` | 正式代码、兼容实现和测试 |
| `tools/` | E 编号运行脚本和维护工具 |
| `workspace/` | 当前工作与组会材料 |
| `agents/` | 多 Agent 原始意见、状态和交接 |
| `runtime/` | 大型临时输出入口说明 |
| `待整理/` | 私有或未分类材料，不进入公开事实链 |

更完整的权威等级见 `docs/学习导航/01_目录与权威等级.md`。

## 7. 按目的选择学习路线

### 从零学习

```text
docs/SafeConf_完整项目讲解/index.html
docs/学习导航/01_目录与权威等级.md
docs/学习导航/02_当前证据链与实验谱系.md
```

### 审核当前研究

```text
docs/实验结果/GATE_STATUS_20260729.md
E192 / E191 / E190 / E189 / E186 / E185 / E184 / E183 / E182 / E181 / E178
```

### 接着写论文

```text
docs/学习导航/04_论文创作接力说明.md
```

注意：`PHASE5A1_METHODS_DRAFT.md` 与 `PHASE5A2_RESULTS_DRAFT.md` 写于 2026-06-16，仍是旧七数据集和 V0/ContextSim 主线，不是最新六数据集正式正文。

### 准备组会

```text
docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md
docs/实验结果/E191_certificate_decision_utility_20260729/reports/E191_INTERPRETATION.md
docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729/final_evaluation/reports/E190_INTERPRETATION.md
docs/实验结果/E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md
docs/实验结果/E183_all_study_family_synthesis_20260724/reports/E183_SYNTHESIS_REPORT.md
docs/实验结果/E182_gse225807_registered_family_20260724/final_evaluation/reports/E182_FINAL_REPORT.md
docs/实验结果/E181_registered_family_hilbert_certificate_20260724/reports/E181_REPORT.md
docs/实验结果/GATE_STATUS_20260729.md
```

## 8. 直接交给外部 Agent 的提示词

```text
你现在在 SafeConf 仓库中。先不要修改文件，也不要启动实验。

请按顺序阅读：
1. START_HERE_FOR_AGENTS.md
2. docs/学习导航/README.md
3. docs/学习导航/01_目录与权威等级.md
4. docs/学习导航/02_当前证据链与实验谱系.md
5. docs/实验结果/GATE_STATUS_20260729.md
6. docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md
7. docs/实验结果/E191_certificate_decision_utility_20260729/reports/E191_INTERPRETATION.md
8. docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729/final_evaluation/reports/E190_INTERPRETATION.md
9. docs/实验结果/E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md
10. docs/实验结果/E186_presubmission_integrity_audit_20260724/reports/E186_REPORT.md
11. docs/实验结果/E185_minimal_release_validation_20260724/reports/E185_REPORT.md
12. docs/实验结果/E184_direct_competitor_positioning_20260724/reports/E184_REPORT.md
13. docs/实验结果/E183_all_study_family_synthesis_20260724/reports/E183_SYNTHESIS_REPORT.md
14. docs/实验结果/E182_gse225807_registered_family_20260724/final_evaluation/reports/E182_FINAL_REPORT.md
15. docs/实验结果/E181_registered_family_hilbert_certificate_20260724/reports/E181_REPORT.md

读完后用通俗中文输出：
- SafeConf 一句话定位；
- 当前注册模型家族证书和 E176/E177/E180/E182 结果；
- E189 怎样回答小矩阵与行列双未见，E190 怎样完成跨研究预测；
- E191 哪些 setting 有复核收益，为什么 double unseen 必须停止排序；
- E192 为什么预算收益为正仍按事前双 gate 返回 ABSTAIN；
- E182 为什么判 FAIL，E183 为什么不能把它改判为通过；
- E184 怎样界定经典理论来源，E185 实际复核了哪些内容；
- pair certificate、registered-family certificate 和旧排序分数的区别；
- 当前最强证据、失败边界和论文缺口；
- 每个结论对应的仓库证据路径。

不得把 smoke、历史 gate、旧论文草稿或 Agent 原始意见写成当前正式结论。
```

完整 Agent 分工和验收标准见 `docs/学习导航/03_Agent学习任务与验收.md`。

## 9. 修改仓库前

1. 先运行 `git status --short --branch` 和 `git log -1 --oneline`。
2. 说明准备修改哪些文件以及为什么。
3. 结论必须附正式证据路径。
4. 不清理用户私有材料，不覆盖 dirty 工作区。
5. 不读取或提交 `.ssh`、`.codex`、`.claude`、`.qoder`、token、key、credential。
6. 新实验必须写输入、输出、通过标准、停止规则和复现命令。

## 10. 学习完成标准

新 Agent 必须能正确回答：

```text
SafeConf 为什么不是预测模型？
E176、E177、E180 与 E182 的校准单位和评价单位分别是什么？
家族多样性为什么能给家族 RMS 误差下界？
模型直径为什么能给最坏成员误差下界，却不能判断哪个模型错？
为什么四项研究的覆盖率不能合并成新的 conformal 保证？
为什么 E177 是边界结果？
为什么旧排序优势必须删除？
为什么 E180 使学习型上界支线停止？
E181 为什么属于方法整合而不是新的前瞻性验证？
E182 为什么仍是 FAIL？E183 为什么不能将它改判为通过？
chemical、directional 和 absolute-RMSE 三条证据为什么不能混写？
```

答不清这些问题时，不应开始修改论文或设计新实验。
