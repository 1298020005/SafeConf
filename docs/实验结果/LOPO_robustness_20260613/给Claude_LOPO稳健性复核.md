# 给 Claude：LOPO 稳健性实验只读复核

请只读检查本目录，不修改代码、不启动新大实验。

先读：

1. `README_先看这个.md`
2. `LOPO_BOOTSTRAP_CI.csv`
3. `LOPO_TRAINING_PROVENANCE.csv`
4. `LOPO_NORMALIZATION_AUDIT.csv`
5. `LOPO_PREDICTOR_DIVERSITY.csv`
6. `LOPO_BAD_RETRIEVAL.csv`

## 请重点回答

1. `PertMean pre_model_task_only` 的 6/7 positive partial rho，是否足以表述为
   “target-predictor-output-free task-risk evidence”？
2. `pre_model_task_only` 含 fold-train label-derived historical features。
   目前“目标预测器输出无关，但需要历史实验效应”的口径是否准确？
3. `LODO x LOPO full` 的 7/7 partial rho CI 下界大于 0，是否足以支持
   “cross-dataset transfer within retrieval-based predictors”？
4. PertMean 在 Frangieh、Srivatsan 与 ContextSim 有超过 50% 向量完全相同，
   且多个数据集 error Spearman 很高。是否需要进一步降低
   “third predictor independence”的措辞？
5. LODO x LOPO 的 top-10 enrichment 只有 4/7 的 CI 下界大于 1。
   是否应明确区分“整体风险排序迁移”和“极端坏预测检索迁移”？
6. Santinha 的 partial rho CI 为正，但 aligned rho 和 AURC reduction CI 不稳定。
   应定位为弱支持还是仅敏感性结果？
7. 请复核 normalization/provenance 表，检查是否存在任何 test label、
   third-predictor error 或 held-out dataset error 泄漏。

## 不应使用的表述

- SafeConf 已证明适用于任意预测器。
- 已完成深度学习预测器验证。
- pre_model_task_only 不需要历史实验数据。
- LODO x LOPO 在所有数据集都能稳定检索 top-10% 最差预测。
- Control1NN 是与 ContextSim 完全独立的预测器。

## 当前建议口径

> 在七个数据集上，使用 V0 和 ContextSim 误差训练的风险模型能够迁移到
> held-out retrieval predictor。更严格的 dataset-and-predictor double holdout
> 中，magnitude-controlled partial correlation 在 7/7 数据集为正且
> bootstrap CI 下界高于 0。该证据支持检索型预测器家族内的任务风险迁移，
> 但不等同于对深度学习预测器的普适验证。
