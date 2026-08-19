# E67｜scGPT 正式微调：与 E66 同一批固定任务

## 这次做了什么

E67 不使用旧的 forward-only scGPT smoke。它按 scGPT 官方 perturbation tutorial 的 GEARS `PertData` + `TransformerGenerator` + MSE fine-tuning 训练路径，把 whole-human checkpoint 的 encoder/value encoder/transformer encoder 迁移到 Norman 训练条件，再在 E66 冻结的 24 个未见基因上测试。

共享面板有 512 个基因：Norman 的全部扰动标签基因都保留，24 个测试基因位于前部；剩余位置从训练条件的表达量排序得到。GEARS E66 三 seed ensemble 和 scGPT 的预测、真值全都投影到同一顺序。strict PredictionRecord issue_count = 0。

## 风险审计

主目标在运行前固定为两模型的 `task_mean_rmse`。下面把 GEARS 自身、scGPT 自身、两者均值和两者最大误差一起列出；这些并列结果用于检查分歧到底在筛谁的错误。

| score | 目标误差 | ρ | bootstrap 95% CI | top20 高误差富集 |
|---|---|---:|---:|---:|
| risk_gears_scgpt_disagreement | GEARS ensemble RMSE | 0.708 | [0.420, 0.868] | 1.284 |
| risk_gears_predicted_magnitude | GEARS ensemble RMSE | 0.377 | [-0.070, 0.783] | 1.245 |
| risk_scgpt_predicted_magnitude | GEARS ensemble RMSE | 0.443 | [0.083, 0.740] | 1.238 |
| true_l2_diagnostic | GEARS ensemble RMSE | 0.675 | — | — |
| risk_gears_scgpt_disagreement | fine-tuned scGPT RMSE | 0.298 | [-0.200, 0.684] | 1.166 |
| risk_gears_predicted_magnitude | fine-tuned scGPT RMSE | 0.103 | [-0.366, 0.541] | 1.155 |
| risk_scgpt_predicted_magnitude | fine-tuned scGPT RMSE | 0.340 | [-0.136, 0.682] | 1.139 |
| true_l2_diagnostic | fine-tuned scGPT RMSE | 0.309 | — | — |
| risk_gears_scgpt_disagreement | mean RMSE across GEARS and scGPT (pre-specified primary) **(主目标)** | 0.589 | [0.196, 0.834] | 1.227 |
| risk_gears_predicted_magnitude | mean RMSE across GEARS and scGPT (pre-specified primary) **(主目标)** | 0.296 | [-0.159, 0.695] | 1.201 |
| risk_scgpt_predicted_magnitude | mean RMSE across GEARS and scGPT (pre-specified primary) **(主目标)** | 0.449 | [0.045, 0.744] | 1.191 |
| true_l2_diagnostic | mean RMSE across GEARS and scGPT (pre-specified primary) **(主目标)** | 0.554 | — | — |
| risk_gears_scgpt_disagreement | max RMSE across GEARS and scGPT | 0.596 | [0.196, 0.832] | 1.320 |
| risk_gears_predicted_magnitude | max RMSE across GEARS and scGPT | 0.248 | [-0.209, 0.650] | 1.284 |
| risk_scgpt_predicted_magnitude | max RMSE across GEARS and scGPT | 0.408 | [0.018, 0.710] | 1.278 |
| true_l2_diagnostic | max RMSE across GEARS and scGPT | 0.530 | — | — |

## 解释边界

这仍是一个固定 24-task、512-gene 的正式适配器实验。它可以回答同任务上的模型分歧是否有排序信息，不能代替全转录组、多数据集、多模型家族的总验证。尤其不能把“对 scGPT 或双模型平均误差有信号”偷换成“已可靠筛出 GEARS 自身错误”；两个目标必须分别阅读。真实 effect 从未进入可部署分数；`true_l2_diagnostic` 只用于检查上限。

## 文件

- 固定任务与切分：`tables/E67_FIXED_SPLIT.csv`
- 基因面板：`tables/E67_GENE_PANEL.csv`
- 严格记录：`tables/PREDICTION_RECORDS.csv`
- 任务分数：`tables/E67_TASK_RISK_TABLE.csv`
- 汇总：`tables/E67_RISK_ERROR_SUMMARY.csv`
- 图：`figures/F1_gears_scgpt_disagreement_vs_mean_error.svg`
