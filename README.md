# SafeConf

SafeConf 研究单细胞扰动预测完成以后，怎样在不知道真实答案时识别高风险任务，以及能否把风险信息用于改进模型训练。

当前分支：`exp/task-risk-audit-20260611`

事实截止：2026-09-06

## 第一次进入先读这两份

1. [小白审核与从零入口](docs/学习导航/07_小白审核与从零入口_20260906.md)：核对当前数字能否当真，并用一道真题从零讲完项目。
2. [当前项目手把手学习](docs/学习导航/00_当前项目手把手学习_20260905.md)：完整教程，从单细胞、扰动、细胞系讲到公式、E199–E205、代码和自测题。

文中所有仓库文件都使用相对链接，下载到其他电脑后仍可跳转。

## 当前进度

| 工作 | 状态 | 当前结论 |
| --- | --- | --- |
| E199：K562 未见基因 | 完成 | 三个公开 TxPert 模型（GAT / Exphormer / Exphormer-MG）的分歧能识别一部分难任务 |
| E200：整个 K562 背景留出 | 完成 | 预测幅度明显强于固定风险分 |
| E201：四背景 × 四种子盲测 | 完成 | SafeConf 与误差稳定正相关，也有幅度之外的信息；但单独排序弱于幅度 |
| E202：相对基线失败归因 | 完成，主门失败 | 当前 source dispersion 不能解释 GAT 相对简单基线多犯的错 |
| E204：风险指导训练 | 四个背景的工程验收通过 | 权重确实进入训练且无覆盖缺口；正式 80 轮对照尚未运行完 |
| E205：跨架构分歧 | 协议已设计 | 尚未运行，当前四个成员只是同架构四个随机种子 |

详细数字以 [当前研究判断](docs/实验结果/CURRENT_RESEARCH_DECISION_20260905.md)、[E201 正式报告](docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md) 和 [E204 工程验收](docs/实验结果/E204_risk_guided_training_20260830/PROFILE_ACCEPTANCE_20260905.md) 为准。

## 目录

| 路径 | 内容 |
| --- | --- |
| `code/` | SafeConf 正式代码和测试 |
| `tools/` | E 编号实验运行脚本、审计和维护工具 |
| `docs/实验结果/` | 冻结协议、结果表、报告和负结果 |
| `docs/学习导航/` | 当前教程和历史学习材料 |
| `workspace/` | 近期汇报与工作材料，不高于正式实验报告 |
| `agents/` | Grok、GLM、Qoder 等原始意见，不是事实来源 |
| `/home/yyf/archive/safeconf/` | 服务器历史归档，不属于 Git 仓库 |

大型 H5AD、模型权重和运行缓存不提交 Git；仓库主要保存代码、协议、精简结果和可核验记录。

## 远程同步

```text
GitHub: https://github.com/1298020005/SafeConf
Gitee:  https://gitee.com/librety/safe-conf
分支:   exp/task-risk-audit-20260611
```

完整拉取和新 Codex 初始化见 [REMOTE_CODEX_INIT.md](REMOTE_CODEX_INIT.md)。
