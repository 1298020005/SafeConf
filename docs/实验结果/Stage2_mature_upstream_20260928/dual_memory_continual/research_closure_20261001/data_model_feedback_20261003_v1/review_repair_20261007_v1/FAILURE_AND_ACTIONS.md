# 修复记录

- Source v3混合不同尺度与逐行置乱：v4改为训练侧通道CDF与兼容整gene block交换，重跑。
- Jackknife额外除以m：改为sum，等权样本方差／n回归测试通过。
- 后续核查发现McFaline J沿用旧effect aggregation，而D来自cell-weighted reference：v3分别复用DEV和holdout的精确合同，并断言V匹配。
- Native61旧预测幅度与当前预测轴不同，similarity列全缺失：本轮统一到当前预测，similarity仅训练分区重建，两版本同样修正。
- XGBoost/sklearn wrapper模型保存异常：改存native Booster；不改变模型拟合、参数或预测。
- 先前未落实连续监管和补全第三项：本回执如实记录空档，不声称持续运行。
