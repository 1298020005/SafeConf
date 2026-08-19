# 给我的中文总览

生成时间：2026-05-21 16:43:37

## 1. 当前服务器硬件资源如何？

服务器资源够用：96 线程 CPU、约 125GiB 内存、2 张 Quadro RTX 6000 24GB GPU。现在 GPU 基本空着。做 confidence scoring 的 MVP 完全不需要 GPU，CPU 就够。

## 2. 当前有哪些可用数据？

最适合先用的是：`KaggleCrossCell`、`Haber`、`Parekh`、`KaggleCrossPatient`。它们有 control、context、perturbation，比较适合计算 `true_effect = perturbed_mean - control_mean`。

大数据如 Replogle、Tian、TCDD 可以后面扩展验证，不建议现在一上来就跑。

## 3. 当前代码有什么能复用？

能复用不少：`build_context_splits.py` 负责构造 task；`transport_models.py` 有 V0 和 ContextSimBaseline；`safetrans_models.py` 有 PolicySafeTransPT、confidence、unsafe_flag；`risk_coverage.py` 已有 risk-coverage；`run_safety_abstention_evidence.py` 已有 safety 相关输出。

## 4. 当前已有结果能不能支持 confidence scoring？

能支持第一版，但不够完整。已有结果里有 `rmse`、`confidence`、`unsafe_flag` 和 `RISK_COVERAGE.csv`。但是缺每个 task 的 `context`、`perturbation`、完整 `PredictionRecord`、多种 confidence baseline 公平对比、held-out context-perturbation pair split。

## 5. 最小 MVP 怎么做？

一句话：先不训练大模型，先证明“我给的 confidence 分数真的能预测模型会不会错”。

具体做法：用已有 `SAFETY_TASK_METRICS.csv` 先算 confidence 和 RMSE 的相关性；再用 `KaggleCrossCell` 生成完整 PredictionRecord；加一个 pair split；跑 V0 和 ContextSimBaseline；比较 support_count、context_similarity、perturbation_stability、expert_disagreement 哪个最能预测 error。

## 6. 现在最不应该做什么？

最不应该继续盲目换模型、盲目下载大数据、盲目占 GPU。导师现在关心的是：这个新 task 怎么定义，怎么评估，baseline 是什么，confidence 分数和真实 error 有没有关系。

## 7. 需要你下一步确认什么？

建议你确认一句话就行：下一步先做 confidence scoring MVP，不继续训练新模型。

如果确认，下一步应该写很小的代码，不动大模型：`build_prediction_records.py`、`evaluate_confidence_scoring.py`、简单画图脚本，然后用已有数据和 V0/ContextSimBaseline 在 CPU 上快速跑出第一版结果。
