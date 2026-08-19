# E114｜任务误差的 split-conformal 上界

每个外层 fold 的 30 个 validation pair 按冻结哈希拆为 15 个风险拟合任务和 15 个 conformal 校准任务。设基础误差预测为 $\hat e(x)$，校准残差为 $r_i=e_i-\hat e(x_i)$；90% 上界为 $U(x)=\hat e(x)+r_{(k)}$，其中 $k=\lceil(n+1)(1-\alpha)\rceil$。在校准任务与未来任务可交换时，该上界具有有限样本边际覆盖保证。

row/column/double shift 不必满足可交换性，因此下面的经验覆盖是必要压力测试，不应写成无条件理论保证。

| setting | n | nominal | empirical | mean error | mean upper |
|---|---:|---:|---:|---:|---:|
| context_and_perturbation_unseen | 90 | 0.90 | 0.978 | 0.0555 | 0.1073 |
| context_unseen_row | 477 | 0.90 | 0.985 | 0.0558 | 0.1031 |
| perturbation_unseen_column | 180 | 0.90 | 0.961 | 0.0531 | 0.0978 |
| random_missing_pair | 90 | 0.90 | 0.989 | 0.0504 | 0.0943 |
| all_test_settings_pooled | 837 | 0.90 | 0.980 | 0.0546 | 0.1014 |

经验覆盖率为 0.980，高于名义 0.90，但平均上界 0.1014 约为平均真实误差 0.0546 的 1.86 倍。当前上界可靠但偏保守，适合做风险兜底，不适合声称已经得到紧致误差预测。
