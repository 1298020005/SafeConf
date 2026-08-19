# SafeConf Agent 学习任务与验收

更新时间：2026-07-29

## 1. 所有 Agent 的共同规则

第一次进入仓库时先只读，不改代码、不启动大实验。按顺序阅读：

1. `START_HERE_FOR_AGENTS.md`
2. `docs/学习导航/README.md`
3. `docs/学习导航/01_目录与权威等级.md`
4. `docs/学习导航/02_当前证据链与实验谱系.md`
5. `docs/实验结果/GATE_STATUS_20260729.md`
6. `docs/实验结果/E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md`
7. `docs/实验结果/E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md`
8. E192–E176 以及任务相关的正式报告

读完后，先交学习报告，再接任务。

## 2. 通用学习提示词

```text
你现在在 SafeConf 仓库中。先不要修改文件，也不要启动实验。

请按顺序阅读：
1. START_HERE_FOR_AGENTS.md
2. docs/学习导航/README.md
3. docs/学习导航/01_目录与权威等级.md
4. docs/学习导航/02_当前证据链与实验谱系.md
5. docs/实验结果/GATE_STATUS_20260729.md
6. docs/实验结果/E194_family_governance_stress_20260729/reports/E194_INTERPRETATION.md
7. docs/实验结果/E193_multigeometry_certificate_robustness_20260729/reports/E193_INTERPRETATION.md
8. docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729/final_evaluation/reports/E192_INTERPRETATION.md

阅读后用通俗中文输出：
1. SafeConf 一句话定位；
2. 当前正式实验合同；
3. E176/E177 双边证书主结果；
4. calibrated、frozen、Directional 和 bilateral certificate 的区别；
5. 当前最强证据、失败边界和论文缺口；
6. 你引用的每个结论对应的仓库路径。

要求：英文词第一次出现时给中文解释；不得把 smoke、历史 gate 或 Agent 原始意见写成当前正式结论。
```

## 3. 不同 Agent 的建议分工

| Agent | 优先任务 | 必读补充文件 | 不应擅自做的事 |
|---|---|---|---|
| Qoder | 代码、数据合同、复现入口和表格追踪 | `code/README.md`、`docs/代码设计/PROTOCOL.md`、RUN_STATUS | 不能只根据脚本名判断实验已完成 |
| Gemini | 领域定位、文献对照、论文故事和通俗解释 | 项目总账、E116、E118、本目录 04 | 不能用外部文献覆盖本项目真实数字 |
| Claude | 对抗式审稿、claim 边界和统计表达 | E131、E132、E111、E114、E117、E118 | 不能把建议直接写成已完成结果 |
| Codex | 实施、版本控制、验证和跨文件一致性 | 当前 gate、任务对应脚本和测试 | 不能覆盖用户 dirty 文件或私有材料 |
| 其他 Agent | 先完成通用学习报告 | 本目录全部文件 | 未通过验收前不接大任务 |

这些角色是建议，不是权限边界。任何 Agent 的结论都要由证据路径复核。

## 4. 学习验收清单

新 Agent 至少应正确回答：

- SafeConf 为什么不是扰动预测模型？
- 一个 task、一个 PredictionRecord 和一个 outer fold 分别是什么？
- E176 与 E177 分别支持什么、不能支持什么？
- 为什么 calibrated SafeConf 和 frozen v0.2 不能混称？
- 模型分歧为什么能给下界，却不能成为单模型置信度？
- 为什么 Tian 不是新的生物细胞类型？
- chemical 线为什么属于边界？
- 哪个文件是当前最高事实入口？
- 旧 Methods/Results 草稿为什么不能直接继续投稿？

如果答错其中任一关键边界，应回到 `GATE_STATUS_20260729.md`、E194、E193 和本目录
02 重读。

## 5. Agent 输出格式

```text
# 学习结论
- 一句话定位：
- 当前主证据：
- 主要边界：

# 证据路径
- 结论 A -> 路径
- 结论 B -> 路径

# 我发现的冲突或疑问
- 文件 A 与文件 B 的冲突：
- 我的判断依据：

# 建议的下一步
- 必须做：
- 可选做：
- 不建议做：
```

## 6. 修改前检查

```text
1. git status --short --branch
2. git log -1 --oneline
3. 说明准备修改的文件和目的
4. 确认没有把待整理、凭据或私人聊天带入提交
5. 修改后运行路径、schema、泄漏或文档链接检查
```

禁止读取或提交 `.ssh`、`.codex`、`.claude`、`.qoder`、token、key、credential。大型 row-level runtime 输出默认不进 Git。
