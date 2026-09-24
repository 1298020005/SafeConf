# E233 第 1 阶段结局：上游能力门未通过

记录：2026-09-24；只涉及预先允许读取的**验证**扰动表达。测试扰动真实表达读取为 0 行。来源是运行时 `E233_STAGE1_SUPERVISOR_STATUS.json` 和两种候选的逐任务／逐背景 CSV；正式外部风险评价 E235 已自动阻断。此记录不把失败改写成其他参数下的成功。

| 同一验证任务与表达尺度 | 任务/背景 | 平均 MSE | 比无变化预测的相对改善 | 背景不劣于无变化 | 预定能力门 |
|---|---:|---:|---:|---:|---|
| 不变预测（对照表达） | 216 / 12 | 0.000440907154 | 0 | — | 参照 |
| `matched_control_softplus` | 216 / 12 | 0.012659460907 | −27.7123（明显变差） | 0 / 12 | FAIL |
| `matched_control_linear` | 216 / 12 | 0.000683254621 | −0.5497（变差） | 0 / 12 | FAIL |

选择规则是“候选先过平均表现和至少 8/12 背景门，再从合格者中选验证 MSE 最低”；本次没有合格者，`selected_variant=null`。监督状态为 `STAGE1_COMPETENCE_BLOCKED`，E235 预真值流程状态为 `BLOCKED_UPSTREAM_COMPETENCE`，最终评价状态为 `BLOCKED_UPSTREAM`。不能在这里换成 IFNG 等局部验证结果后称 E235 正式通过。后续独立实验需预先登记新上游模型/数据合同，不复用 E235 测试集真值选模型。

运行时证据路径（不纳入 Git 的大文件目录）：

- `/home/yyf/data/perturbench_e233/stage1_validation_20260923/E233_STAGE1_SUPERVISOR_STATUS.json`
- 同目录 `E233_matched_control_softplus_TASKS.csv`、`E233_matched_control_softplus_CONTEXTS.csv`、`E233_matched_control_linear_TASKS.csv`、`E233_matched_control_linear_CONTEXTS.csv`
- `/home/yyf/data/perturbench_e235/pretruth_supervisor_20260924/E235_PRETRUTH_SUPERVISOR_STATUS.json`
- `/home/yyf/data/perturbench_e235/finalization_20260924/E235_FINALIZER_STATUS.json`

论文候选的统一贡献、数据范围与下一轮实验在[当前决策表](../../方法设计/20260922_条件风险模型/20260924_三项贡献与实验范围.md)。
