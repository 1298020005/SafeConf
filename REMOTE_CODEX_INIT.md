# SafeConf 远程电脑与新 Codex 初始化

更新时间：2026-09-05

当前分支：`exp/task-risk-audit-20260611`

## 1. 仓库地址

GitHub：

```text
https://github.com/1298020005/SafeConf
https://github.com/1298020005/SafeConf/tree/exp/task-risk-audit-20260611
```

Gitee：

```text
https://gitee.com/librety/safe-conf
https://gitee.com/librety/safe-conf/tree/exp/task-risk-audit-20260611
```

两端的事实分支都是 `exp/task-risk-audit-20260611`。不要默认 `main` 或 `master` 已包含相同进度。

## 2. 第一次下载

优先使用 GitHub：

```bash
git clone https://github.com/1298020005/SafeConf.git safeconf
cd safeconf
git fetch --all --prune
git checkout exp/task-risk-audit-20260611
git pull --ff-only
git log -1 --oneline
```

GitHub 访问不稳定时使用 Gitee：

```bash
git clone https://gitee.com/librety/safe-conf.git safeconf
cd safeconf
git fetch --all --prune
git checkout exp/task-risk-audit-20260611
git pull --ff-only
git log -1 --oneline
```

仓库已经存在时：

```bash
cd safeconf
git status --short
git fetch --all --prune
git checkout exp/task-risk-audit-20260611
git pull --ff-only
git log -1 --oneline
```

若 `git status --short` 有本地修改，先保存并确认归属，不要直接覆盖。

## 3. 拉取后先检查

```bash
test -f docs/学习导航/00_当前项目手把手学习_20260905.md
test -f docs/实验结果/CURRENT_RESEARCH_DECISION_20260905.md
test -f docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md
test -f docs/实验结果/E204_risk_guided_training_20260830/PROFILE_ACCEPTANCE_20260905.md
```

四条命令没有输出且返回成功，说明当前学习入口和核心报告都已拉到。

## 4. 人工学习顺序

第一遍只读：

```text
docs/学习导航/00_当前项目手把手学习_20260905.md
```

按它的第 14 节分四遍学习：

1. 先理解生物问题、task、source、target 和三种留出；
2. 再理解五个风险成分、预测幅度、RMSE、Spearman 和置信区间；
3. 接着对照一行真实数据和 E201/E204 代码链；
4. 最后回答第 18 节十道自测题。

核对数字时再打开：

```text
docs/实验结果/CURRENT_RESEARCH_DECISION_20260905.md
docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md
docs/实验结果/E204_risk_guided_training_20260830/PROFILE_ACCEPTANCE_20260905.md
docs/实验结果/E204_risk_guided_training_20260830/IMPLEMENTATION_AMENDMENT_20260905.md
```

## 5. 让远程 Codex 先学习，不改代码

在仓库根目录打开 Codex，发送下面这段：

```text
你现在位于 SafeConf 仓库的 exp/task-risk-audit-20260611 分支。

先不要改文件、运行实验或设计新结论。请依次阅读：
1. docs/学习导航/00_当前项目手把手学习_20260905.md
2. docs/实验结果/CURRENT_RESEARCH_DECISION_20260905.md
3. docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md
4. docs/实验结果/E204_risk_guided_training_20260830/IMPLEMENTATION_AMENDMENT_20260905.md
5. docs/实验结果/E204_risk_guided_training_20260830/PROFILE_ACCEPTANCE_20260905.md

读完后用通俗中文完成三件事：
A. 回答学习稿第 18 节的 10 个自测题；
B. 分开列出已经完成的结果、负结果、工程验收和下一步计划；
C. 说明周老师每个问题由哪个实验回答，并给相对路径。

不能把四个随机种子说成四种模型，不能说 SafeConf 单独超过预测幅度，
不能把 E204 profile PASS 写成正式性能提升，也不要从 agents 原始意见推断事实。
```

## 6. Git 中有什么，没有什么

Git 中保存：

- 代码和测试；
- 冻结协议、运行状态和哈希；
- 精简结果表、正式报告和论文图；
- 学习导航与复现说明。

Git 中通常不保存：

- 大型 H5AD 单细胞表达文件；
- 完整模型权重和 GPU 缓存；
- 可由正式脚本重新生成的大型中间数组。

因此，在普通电脑上可以完整学习项目、查看结果和审核代码，但重新训练 E201/E204 需要服务器数据盘和对应环境。

## 7. 当前状态核对

拉取后的 Codex 必须知道：

```text
E201 已正式完成，不再是盲训练中。
E202 主门失败，是需要保留的负结果。
E204 四个 target 的工程 profile 已通过，正式 80 轮效果尚未产生。
E205 只是协议，尚未运行。
```

如果旧文件仍写“E201 未解封”，把它当历史时间点，不要覆盖当前状态。
