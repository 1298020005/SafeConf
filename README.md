# SafeConf

SafeConf 是一个单细胞扰动预测结果的可靠性审计项目。

一句话：

```text
SafeConf 不预测细胞会怎么变；
SafeConf 给冻结预测家族建立可复核的误差下界和上界。
```

## 现在只看这些入口

| 入口 | 用途 |
|---|---|
| `SafeConf_统一研究工作台.html` | 统一可跳转导航：当前问题、证据、组会、学习、代码和归档 |
| `INDEX.md` | 项目总入口，新 AI 和人都先看这里 |
| `START_HERE_FOR_AGENTS.md` | 给 Qoder / Gemini / Claude / 新 Codex 的完整学习地图 |
| `docs/学习导航/README.md` | Agent 学习、目录权威、证据谱系和论文接力的统一目录 |
| `START_HERE_FOR_GPT.md` | 给网页 GPT / Claude 的第一入口 |
| `workspace/` | 当前组会和近期工作材料 |
| `docs/实验结果/` | 已冻结实验结果和论文证据 |
| `docs/实验结果/GATE_STATUS_20260729.md` | E194 后的最高事实入口：老师问题、外部验证、family 治理和投稿阻断项 |
| `docs/投稿准备/期刊与文献定位_20260729/` | 当前期刊梯队、2024–2026 近邻论文与实验修正 |
| `docs/实验结果/E194_family_governance_stress_20260729/` | family 重复、失衡、遗漏与合成攻击治理 |
| `docs/实验结果/E193_multigeometry_certificate_robustness_20260729/` | RMSE、cosine、Pearson 多几何证书 |
| `REPRODUCE_CURRENT_RELEASE.md` | 一条命令复核当前证书主数字，不依赖 GPU 或原始数据 |
| `docs/实验结果/E186_presubmission_integrity_audit_20260724/` | 投稿前 18 项完整性对抗审计、0 失败 |
| `docs/实验结果/E185_minimal_release_validation_20260724/` | 12,033 项发布物复现检查、0 失败 |
| `docs/实验结果/E184_direct_competitor_positioning_20260724/` | 直接竞品、经典理论来源、可写贡献边界和白底定位图 |
| `docs/实验结果/E183_all_study_family_synthesis_20260724/` | 四项研究 2,433 个任务、737 个靶点簇的统一审计和三张白底图 |
| `docs/实验结果/E182_gse225807_registered_family_20260724/` | GSE225807 完整事前冻结流程、下界通过和上界注册门槛 FAIL |
| `docs/实验结果/E181_registered_family_hilbert_certificate_20260724/` | 2,393 个任务的统一证书、五张白底图和复现表 |
| `docs/实验结果/E180_xucao_fresh_guide_certificate_20260723/` | XuCao 一次性独立评价和自适应上界负结果 |
| `docs/实验结果/E143_prospective_wetlab_validation_20260714/` | 前瞻湿实验的功效、候选、盲法、QC、图和交接模板 |
| `docs/投稿准备/录用判断与项目总账_20260713/index.html` | 周老师问题、当前证据、投稿定位和不能保证录用的原因 |
| `docs/SafeConf_完整项目讲解/index.html` | 从零读懂整个项目：生物、数据、代码、实验、结果和下一阶段 |
| `docs/小白科普/` | 早期专题学习材料与真实数据样例 |
| `agents/` | Codex / Qoder / Grok 等 AI 协作状态和原始输出 |
| `code/` | SafeConf 正式代码 |
| `tools/` | 下载、复跑、资源清单、Codex 环境等维护工具 |
| `runtime/` | 临时输出说明；大型运行输出不进 Git |
| `/home/yyf/archive/safeconf/` | 集中历史库，不在 Git 工作区内 |

## 顶层纪律

顶层只保留当前工作。旧讨论、旧草稿、旧汇报和旧计划统一迁往 `/home/yyf/archive/safeconf/`。

```text
日常工作：workspace/
Agent 学习：START_HERE_FOR_AGENTS.md -> docs/学习导航/
完整学习：docs/SafeConf_完整项目讲解/index.html
专题样例：docs/小白科普/
论文证据：docs/实验结果/
AI 协作：agents/
代码运行：code/ + tools/
旧东西：/home/yyf/archive/safeconf/
```

## 同步

服务器主项目：

```bash
cd /home/yyf/proj
git status --short --branch
```

远端：

```text
GitHub: git@github.com:1298020005/SafeConf.git
Gitee:  https://gitee.com/librety/safe-conf.git
```

本分支：

```text
exp/task-risk-audit-20260611
```
