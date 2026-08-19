# E65｜scGPT 正式微调：与 E60 同一批固定任务

## 这次做了什么

E65 不使用旧的 forward-only scGPT smoke。它按 scGPT 官方 perturbation tutorial 的 GEARS `PertData` + `TransformerGenerator` + MSE fine-tuning 训练路径，把 whole-human checkpoint 的 encoder/value encoder/transformer encoder 迁移到 Adamson 训练条件，再在 E60 冻结的 24 个未见基因上测试。

共享面板有 512 个基因：Adamson 的全部扰动标签基因都保留，24 个测试基因位于前部；剩余位置从训练条件的表达量排序得到。GEARS E60 三 seed ensemble 和 scGPT 的预测、真值全都投影到同一顺序。strict PredictionRecord issue_count = 0。

## 风险审计

主目标在运行前固定为两模型的 `task_mean_rmse`。下面把 GEARS 自身、scGPT 自身、两者均值和两者最大误差一起列出；这些并列结果用于检查分歧到底在筛谁的错误。

| score | 目标误差 | ρ | bootstrap 95% CI | top20 高误差富集 |
|---|---|---:|---:|---:|
| risk_gears_scgpt_disagreement | GEARS ensemble RMSE | 0.335 | [-0.095, 0.679] | 1.377 |
| risk_gears_predicted_magnitude | GEARS ensemble RMSE | 0.076 | [-0.282, 0.438] | 0.937 |
| risk_scgpt_predicted_magnitude | GEARS ensemble RMSE | 0.143 | [-0.285, 0.516] | 1.051 |
| true_l2_diagnostic | GEARS ensemble RMSE | 0.948 | — | — |
| risk_gears_scgpt_disagreement | fine-tuned scGPT RMSE | 0.566 | [0.224, 0.767] | 1.453 |
| risk_gears_predicted_magnitude | fine-tuned scGPT RMSE | -0.139 | [-0.528, 0.281] | 0.881 |
| risk_scgpt_predicted_magnitude | fine-tuned scGPT RMSE | 0.371 | [-0.018, 0.648] | 1.229 |
| true_l2_diagnostic | fine-tuned scGPT RMSE | 0.707 | — | — |
| risk_gears_scgpt_disagreement | mean RMSE across GEARS and scGPT (pre-specified primary) **(主目标)** | 0.494 | [0.099, 0.737] | 1.418 |
| risk_gears_predicted_magnitude | mean RMSE across GEARS and scGPT (pre-specified primary) **(主目标)** | -0.038 | [-0.394, 0.338] | 0.907 |
| risk_scgpt_predicted_magnitude | mean RMSE across GEARS and scGPT (pre-specified primary) **(主目标)** | 0.284 | [-0.140, 0.613] | 1.146 |
| true_l2_diagnostic | mean RMSE across GEARS and scGPT (pre-specified primary) **(主目标)** | 0.882 | — | — |
| risk_gears_scgpt_disagreement | max RMSE across GEARS and scGPT | 0.600 | [0.267, 0.784] | 1.481 |
| risk_gears_predicted_magnitude | max RMSE across GEARS and scGPT | -0.129 | [-0.512, 0.282] | 0.879 |
| risk_scgpt_predicted_magnitude | max RMSE across GEARS and scGPT | 0.397 | [0.011, 0.680] | 1.206 |
| true_l2_diagnostic | max RMSE across GEARS and scGPT | 0.737 | — | — |

## 解释边界

这仍是一个固定 24-task、512-gene 的正式适配器实验。它可以回答同任务上的模型分歧是否有排序信息，不能代替全转录组、多数据集、多模型家族的总验证。尤其不能把“对 scGPT 或双模型平均误差有信号”偷换成“已可靠筛出 GEARS 自身错误”；两个目标必须分别阅读。真实 effect 从未进入可部署分数；`true_l2_diagnostic` 只用于检查上限。

## 文件

- 固定任务与切分：`tables/E65_FIXED_SPLIT.csv`
- 基因面板：`tables/E65_GENE_PANEL.csv`
- 严格记录：`tables/PREDICTION_RECORDS.csv`
- 任务分数：`tables/E65_TASK_RISK_TABLE.csv`
- 汇总：`tables/E65_RISK_ERROR_SUMMARY.csv`
- 图：`figures/F1_gears_scgpt_disagreement_vs_mean_error.svg`
