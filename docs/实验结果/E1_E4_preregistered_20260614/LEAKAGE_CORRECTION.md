# Outer-fold 泄漏发现与修正

发现日期：2026-06-14

## 问题

初版 learned LOPO 将 5 个 outer folds 的 train+val 行合并训练，再统一预测所有
fold 的 test 行。每个 `task_key` 在五个 fold 中各出现一次，因此一个 fold 的 test
task 会在其他 fold 以 train/val 行出现。

矩阵审计显示：

- 7/7 数据集的每个 PertMean test task 都能在其他 fold 的 V0/ContextSim
  train+val source 中找到同一 task；
- 这是任务误差标签的跨 fold 泄漏；
- 仅检查 test 是否参与 normalization 不能发现该问题。

## 影响

- 初次运行的 E1、E3、E4 learned-model 数字无效；
- 2026-06-13 的 learned LOPO/LODO×LOPO 结果需要 fold-safe 重跑；
- E2 本来就显式限定同一个 `fold_id`，其外层 train/test 结构不受该问题影响。

无效结果没有作为正式结果提交。现有证据以修正后的 fold-safe rerun 为准。

## 修正

learned model 现在：

1. 每个 outer fold 独立拟合；
2. source 只取当前 fold 的 V0/ContextSim train+val；
3. target 只取当前 fold 的 PertMean test；
4. LODO×LOPO 额外排除 held-out dataset；
5. 运行时断言 source 与 target 的 `(dataset_name, task_key)` 交集为 0；
6. provenance 记录 `fold_id` 和 `source_target_task_overlap=0`。

新增两 fold 回归测试，防止后续再次把 folds 纵向合并训练。
