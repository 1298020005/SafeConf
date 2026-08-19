# McFarland Failure Diagnosis

## 结论

McFarlandTsherniak2020（drug-only）不是完全没有信号，而是冻结的 `protocol_v0_2_family_confidence` 在这个数据集上方向不对。

- 主方法 aligned rho: -0.086
- 主方法 partial rho: -0.061
- 最强单项: `learned_risk_score`，aligned rho = 0.587
- `historical_residual_risk`: aligned rho = 0.426
- `support_count_score`: aligned rho = -0.148
- `context_similarity_score`: aligned rho = -0.078
- best leave-one-dose result: remove dose `2.5`, aligned rho = -0.013, partial rho = 0.011

这支持 Claude 的判断：不要为了 McFarland 修改冻结公式；应把它写成化学线的 failure boundary（失败边界），同时报告它还有 historical residual 这类替代信号。

## 数据结构

- filtered h5ad: `/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_formal_main_20260604/McFarlandTsherniak2020/input/McFarlandTsherniak2020__filtered_perturbation_type_drug.h5ad`
- cells after drug-only filter: 154,710
- cell lines: 172
- non-control drugs: 13
- observed non-control cell_line × drug pairs in h5ad metadata: 1,175
- formal held-out test cell_line × drug pairs: 1,163
- dose values: 0.0, 0.1, 0.5, 1.0, 10.0, 2.5, 5.0 
- time labels: 24, 3, 6, 12, 24, 48, 6 

## 为什么失败更可能是任务结构问题

1. McFarland 的扰动数很少，只有 13 个非 control drug，但 cell line 很多；`support_count` 会变成“覆盖多不多”的粗信号，未必代表这次预测好不好。
2. dose/time 混在同一个 drug 名下；同一 drug 在不同剂量或时间下可能不是同一种 effect（效应）。
3. `historical_residual_risk` 为正，说明可以从历史残差看到难题，但 v0.2 公式主推的 support/context/disagreement 组合在这里不合适。

## 补充诊断

- `McFarland_leave_one_dose_out_rho.csv` 检查“去掉某个 dose 后主公式是否回正”。这不是为了调公式，而是定位失败是否集中在特定 dose。
- `McFarland_time_label_audit.csv` 检查混合 time label，例如 `3, 6, 12, 24, 48` 这类标签不应被当作单一时间点解释。

## 建议口径

主表保留 McFarland，不删除；Discussion 写为：SafeConf 对基因线稳定，化学线存在强例子（Srivatsan）和失败边界（McFarland）。McFarland 后续若要救，应先重新定义 drug-dose-time task，而不是改 v0.2 公式。
