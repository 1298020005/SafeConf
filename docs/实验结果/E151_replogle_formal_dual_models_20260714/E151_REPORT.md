# E151｜Replogle K562/RPE1 正式双模型预测

两个外层 cell-line holdout folds 已完成。全部诊断测试任务 340 个，其中预注册主分析是 256 个唯一 held-out cell-line × perturbation 任务；strict PredictionRecord 680 条，issues=0。

scGPT 与 GEARS 完整沿用 E112/E138 的固定训练、验证和输出流程。目标细胞系的 control 可见，目标 perturbed expression 只在预测与风险分数固定后用于评价。因此本实验检验同一研究内的 control-observed 跨细胞系复制，不是跨研究或完全不可见目标背景的 zero-shot。
