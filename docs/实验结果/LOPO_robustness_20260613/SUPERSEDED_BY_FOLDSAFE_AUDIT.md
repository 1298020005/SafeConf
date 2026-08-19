# Superseded notice

2026-06-14 审计发现：该目录中的 learned LOPO 训练将五个 outer folds 合并，
导致 test task 的误差标签可从其他 fold 的同 task train/val 行进入训练。

因此：

- learned LOPO 与 learned LODO×LOPO 数字暂不作为正式证据；
- random、magnitude、frozen protocol v0.2 等非 learned baseline 不受该训练问题影响；
- 以 `E1_E4_preregistered_20260614` 的 fold-safe rerun 和随后生成的新版 LOPO
  证据为准。

修复要求每个 outer fold 单独拟合，并强制
`source_target_task_overlap = 0`。
