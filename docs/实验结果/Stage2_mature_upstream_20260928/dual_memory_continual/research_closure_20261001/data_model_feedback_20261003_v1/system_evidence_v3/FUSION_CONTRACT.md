# SafeConf 统一评分与融合合同

## 评分入口

`score_task(task, frozen_config)` is the single dispatch interface. It does not fit a model, read target truth, or change a frozen configuration.

Rule candidates remain label-free: amplitude and public-history scores are independently transformed with training-only empirical CDFs, and missing public history falls back to amplitude with an explicit availability flag.

## Supervision sources

- `Public_Ridge_source_supervised` is trained only on E201 Source error midrank labels. McFaline target errors are evaluation-only.
- `TargetRidge_fusion` is trained only on the registered McFaline DEV errors for each feedback budget. The 212-task holdout errors are evaluation-only.
- Source and target error labels are not pooled.

## Leakage controls

Shared and target base scores supplied to Ridge are generated fold-out-of-fold. Channel-score CDFs and error-label CDFs are audited separately. The original public rule is retained as an independent baseline and does not pass through Ridge.

Legacy matrices are used only after exact task and truth-contract alignment. The current audit rejected the legacy Matrix/PertEMA files because their `true_error_rmse` differs from the current frozen target contract; those rows remain in the alignment audit and are excluded from current same-task metrics.
