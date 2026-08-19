# E83｜CPA-RDKit 化学四象限 pilot

E83 在 E81 的 `E81_r1_p75` 上训练官方 CPA 0.8.8。模型只读取冻结训练任务、各 context 的 vehicle control、外部 SMILES 和 log10-dose；测试任务的 perturbed expression 在预测文件落盘后才用于评价。本轮用于开发管线，run_type=`smoke`，不进入正式汇总。

- CPA 与 ridge 共享任务、1000 基因顺序和 true effect
- strict PredictionRecord：118，问题数 0
- target truth 进入 score：否
- 训练输入修正：原始 nM 会使 linear doser 数值溢出，因此预先固定为 log10-dose；该规则未读取测试误差

## CPA–ridge 分歧对 pair mean error

quadrant,score_name,target_error,n_tasks,spearman
new_context_new_perturbation,cpa_ridge_disagreement_rmse,pair_mean_rmse,8,0.7857142857142858
new_context_seen_perturbation,cpa_ridge_disagreement_rmse,pair_mean_rmse,28,0.7186644772851669
seen_context_new_perturbation,cpa_ridge_disagreement_rmse,pair_mean_rmse,12,0.1258741258741259
seen_context_seen_perturbation_pair_holdout,cpa_ridge_disagreement_rmse,pair_mean_rmse,11,0.7181818181818183


这是单 manifest pilot。测试结果已经查看，因此该 manifest 永久保留为开发集，不再通过增加 epoch 或改参数升级成正式证据。正式 E84 固定参数后只运行其余 8 个未查看 manifest。
