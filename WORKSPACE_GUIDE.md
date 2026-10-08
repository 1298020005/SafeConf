# SafeConf 当前工作区与实验室接手

更新：2026-10-03。本文整理服务器入口、Git 分支及历史恢复，不改变科研方法或实验记录。

## 当前研究从这里读

当前研究分支是 `exp/e220-reviewer-closure-20260921`。服务器的实际工作目录为 `/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921`，共享 Git 对象库为 `/home/yyf/proj/.git`。`proj` 的旧事实分支及旧教程保留。

最近一轮实验的直接入口：

| 要核对什么 | 文件 |
| --- | --- |
| 实验收口与模型采用 | [FINAL_EXPERIMENTAL_DECISION.md](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/FINAL_EXPERIMENTAL_DECISION.md) |
| 简明完成状态 | [METHOD_STATUS.md](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/METHOD_STATUS.md) |
| 机器可读的模型设置 | [FINAL_MODEL_SETTINGS.json](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/FINAL_MODEL_SETTINGS.json) |
| 完成回执与文件绑定 | [FINAL_EXPERIMENT_MANIFEST.json](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/FINAL_EXPERIMENT_MANIFEST.json) |
| 各阶段结果清单 | [FINAL_RESULT_INDEX.csv](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/FINAL_RESULT_INDEX.csv) |
| 较早 Orion 阶段的边界 | [METHOD_DECISION.md](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/METHOD_DECISION.md) |
| 复现说明 | [REPRODUCIBILITY_REPORT.md](docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/REPRODUCIBILITY_REPORT.md) |
| 新电脑初始化 | [REMOTE_CODEX_INIT.md](REMOTE_CODEX_INIT.md) 的当前分支段落 |

本轮公共参照、联合风险、内容置乱与受影响反馈实验已经完成，正文/PDF 依用户最新指示延后。当前采用及限定范围见最终决定，不以维护说明推断科学结论。九月的 INDEX、START_HERE、旧教程和早先状态是历史时间点；不能覆盖更晚的最终回执。下一轮仍先核对用户最新要求和实际运行状态。

## 会话与并行工作

lab-168 当前主会话标题为“论文学习”，ID `01a0e117-aaf2-7371-8710-9954cecfe72e`；它从旧主会话 `019f139e-0161-7483-aed1-857a3ad57666` 续接。更早的“论文创作”和改名会话也保留，不能凭同名判断重复。

Windows 的“论文学习 / 论文创作”是独立审阅会话，不替代 lab-168 主会话。网页版意见需由主会话对照真实资产判断。原始聊天记录保留在应用管理目录，未上传到 Git。

多个会话可能共享仓库。修改前看工作树状态和进程归属，不把其他会话的 dirty/untracked 文件打包提交，不切换或重置其工作分支。当前科研任务继续沿用用户授权和具体实验合同。

## 服务器资产地图

| 服务器路径 | 内容 |
| --- | --- |
| `/home/yyf/INDEX.md`、`WORKSPACE_GUIDE.md`、`AGENTS.md` | 本机新会话总入口和接手说明 |
| `/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921` | 当前研究代码、协议、报告及未提交工作 |
| `/home/yyf/runtime_worktrees/e205_postprocess_20260915` | E208 后台监督任务的工作目录 |
| `/home/yyf/proj` | 共享主仓库及旧事实分支；原位保留 |
| `/home/yyf/data` | 原始/处理数据、模型和冻结资产；约 1013.4 GiB（整理前快照） |
| `/home/yyf/datasets` | 指向 data 的兼容软链接，不是第二份数据 |
| `/home/yyf/runtime_artifacts` | 近期大型数组、模型和中间产物；约 143.2 GiB |
| `/home/yyf/safeconf_runtime` | 较早实验运行产物；约 18.0 GiB |
| `/home/yyf/Desktop`、`师姐论文` | 阅读、汇报和论文材料，保留原路径 |
| `/home/yyf/archive` | 历史项目、独有 Git 提交、资料与备份 |
| `/home/yyf/archive/audits/20261002` | 本轮完整空间/Git/环境/会话审计、校验及恢复说明；仅本机保存 |
| `/home/yyf/archive/compressed/20261002` | 无损压缩的五月封存旧数据 |

普通 Git clone 只带回代码、协议和精简证据，完整训练需要服务器数据与环境。

## 运行环境与历史维护

本轮观察到两张 Quadro RTX 6000 各 24 GiB，GPU 空闲；两个 E208 Python 监督进程仍运行，使用 `.venvs/perturbench-c84038bc`、`data/perturbench_e208` 及 e205 工作树。tmux 会话为 `safeconf_e208_risk_seal_20260921` 和 `safeconf_e208_finalize_20260921`。维护时重新核对进程、打开文件和锁；不能以 GPU 空闲推断任务已结束。

`.venvs`、`.conda/envs` 和 `.local/lib/python3.12` 保留；部分环境继承全局包。基础解释器在 `/home/miniconda`。Codex JSONL、SQLite/WAL、Cursor snapshots 与账号/网络/SSH 配置保持原位。

本轮空间处理仅限五月已封存且未发现当前引用的 `archive/code/20260519_0958_home_cleanup/moved_dataset_old`：26 个文件的完整原文保存在 tar.zst 中，释放原副本前完整解压逐文件 SHA-256 验证，并重新核对源内容与元数据；原位置留恢复说明。准确压缩大小、最终校验结果和恢复命令见服务器本轮报告。

其余数据、checkpoint、当日 incomplete 目录和科研历史保留。若同 SHA-256 checkpoint 以后需要去重，还须核对冻结清单、当前依赖和恢复方式；本轮只登记候选。旧 Git 工作树占用较小，部分有独有提交或未提交修改，保留原位。

GitHub remote 为 `github`，Gitee remote 为 `origin`。2026-10-03 已将 Gitee e220 分支安全快进到主会话已提交且已发布到 GitHub 的 `f185e07ab5a77c61449cb9ecfb2738dca1cd914c`；未改动当前工作树或未提交内容。后续以实际 `git ls-remote` 回执为准，镜像并非自动持续同步。
