# SafeConf 项目总入口

先别全盘乱扫。按下面顺序看。

## 1. 今天要看什么

| 你想做什么 | 打开 |
|---|---|
| 看一区路径、聊天审核和下一阶段 | `docs/投稿准备/Q1_PATH_SEMANTIC_AUDIT_20260815.md` |
| 看 2026-08-15 当前状态 | `docs/实验结果/CURRENT_RESEARCH_STATUS_20260815.md` |
| 准备 2026-08-13 中午组会（已结束，只作讲稿） | `workspace/group_meeting_20260813/README_先看这个.md` |
| 在另一台电脑接手、看当前 E201 进度和数据边界 | `docs/项目交接_20260812.md` |
| 在一个页面看全部进度、问题和文档 | `SafeConf_统一研究工作台.html` |
| 看当前项目进度 | `workspace/README_先看这个.md` |
| 看 2026-07-09 组会汇报 | `workspace/group_meeting_20260709_FINAL/README_只看这个.md` |
| 从零学完整项目（推荐） | `docs/SafeConf_完整项目讲解/index.html` |
| 查 h5ad 和真实数据专题 | `docs/小白科普/index.html` |
| 看正式实验结果 | `docs/实验结果/` |
| 看当前最高事实、E201 进度和下一步 | `docs/实验结果/CURRENT_RESEARCH_STATUS_20260814.md` |
| 看 E181–E197 历史总账（不是今天主入口） | `docs/实验结果/GATE_STATUS_20260729.md` |
| 看预测家族治理、防复制和合成攻击结果 | `docs/实验结果/E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md` |
| 看 RMSE、cosine、Pearson 多几何证书 | `docs/实验结果/E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md` |
| 看当前期刊梯队、近邻论文和实验阻断项 | `docs/投稿准备/期刊与文献定位_20260729/README.md` |
| 一条命令复核当前证书 | `REPRODUCE_CURRENT_RELEASE.md` |
| 看投稿前完整性对抗审计 | `docs/实验结果/E186_presubmission_integrity_audit_20260724/reports/E186_REPORT.md` |
| 看最小复现实际运行结果 | `docs/实验结果/E185_minimal_release_validation_20260724/reports/E185_REPORT.md` |
| 看直接竞品、理论来源和可写贡献边界 | `docs/实验结果/E184_direct_competitor_positioning_20260724/reports/E184_REPORT.md` |
| 看四项研究统一审计与有限校准波动 | `docs/实验结果/E183_all_study_family_synthesis_20260724/reports/E183_SYNTHESIS_REPORT.md` |
| 看 GSE225807 事前冻结评价与注册门槛 FAIL | `docs/实验结果/E182_gse225807_registered_family_20260724/final_evaluation/reports/E182_FINAL_REPORT.md` |
| 看注册模型家族双侧证书和五张白底图 | `docs/实验结果/E181_registered_family_hilbert_certificate_20260724/README_先看这个.md` |
| 看 XuCao 一次性独立评价 | `docs/实验结果/E180_xucao_fresh_guide_certificate_20260723/final_evaluation/reports/E180_FINAL_REPORT.md` |
| 看周老师问题和模型特异性审计 | `docs/实验结果/E178_crossstudy_bilateral_certificate_audit_20260722/reports/E178_REPORT.md` |
| 接着做前瞻湿实验 | `docs/实验结果/E143_prospective_wetlab_validation_20260714/README_先看这个.md` |
| 看周老师的问题、投稿定位和录用边界 | `docs/投稿准备/录用判断与项目总账_20260713/index.html` |
| 让 Qoder/Gemini/外部 agent 完整学习项目 | `START_HERE_FOR_AGENTS.md` |
| 查看 Agent 学习目录、证据谱系和论文接力 | `docs/学习导航/README.md` |
| 让 Claude/GPT 快速理解项目 | `START_HERE_FOR_GPT.md` |
| 让多个 AI 协作 | `agents/README.md` |

## 2. 当前项目定位

```text
SafeConf = registered-family reliability certificate
中文：注册预测家族的可靠性误差证书
```

它做的不是：

```text
输入扰动 -> 预测细胞变化
```

它做的是：

```text
多个冻结预测结果 + 独立校准数据
  -> 给出家族平均误差和最坏成员误差的下界
  -> 给出继承 conformal 覆盖事件的经验上界
```

## 3. 当前目录规则

| 目录 | 状态 |
|---|---|
| `code/` | 正式代码 |
| `docs/` | 学习导航、完整项目讲解、稳定文档、实验结果和专题科普 |
| `agents/` | AI 协作当前状态和原始输出 |
| `workspace/` | 当前科研推进 |
| `tools/` | 工具、脚本、服务器环境文档 |
| `runtime/` | 临时输出说明 |
| `/home/yyf/archive/safeconf/` | 集中历史库，不属于当前 Git 工作区 |

## 4. 禁止误读

- 不要把 `/home/yyf/archive/safeconf/` 当当前结论。
- 不要把 `agents/qoder/` 的原始输出当最终学习资料。
- 不要把 E8b 写成“27 个模型逐预测验证”。
- 不要说 SafeConf 已经证明对所有深度模型普适。
- 不要说 McFarland frozen v0.2 成功；它仍是失败边界。

## 5. 给 AI 的规矩

1. 先读 `START_HERE_FOR_AGENTS.md` 和 `docs/学习导航/README.md`。
2. 结论必须给证据路径。
3. 新实验必须写输入、输出、通过标准。
4. 用户是小白，英文第一次出现要写中文解释。
