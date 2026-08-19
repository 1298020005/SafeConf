# E146｜唯一生物任务统计依赖审计合同

## 审计性质

E140 的任务误差和风险结果已经解封。本分析是在看到 E140 结果之后增加的统计依赖再审计，**不是新的预注册确认实验**，不得用于把既有证据改写成前瞻性验证。

本轮不重新拟合风险分数，不筛选任务，不更换端点。E140 冻结快照为 `docs/实验结果/E146_unique_biological_task_meta_20260714/tables/E146_E140_TASK_INPUT.csv`，SHA-256 为 `sha256:95e5c776f97efbefaaf4d312dbb906ac100701e9682480020e8dd32153188ec9`；E139 方向快照为 `docs/实验结果/E146_unique_biological_task_meta_20260714/tables/E146_E139_DIRECTIONAL_INPUT.csv`，SHA-256 为 `sha256:5c2ec6b5dfa7de56fbb873403a5910faf71c0972f188feb346fa851005517268`。

## 冻结定义

- 主端点：`error_two_predictor_mean_rmse`。
- 原 SafeConf absolute 风险：`safeconf_calibrated_pair_risk`。
- 主比较器：`risk_model_disagreement`、`baseline_predicted_magnitude`。
- E140 主估计量：保持原 fold 内 Spearman、再对 fold 等权宏平均；bootstrap 按 `(dataset, perturbation)` 整簇重抽，使同一扰动的所有 context 和 outer-fold 记录使用同一个重抽权重。
- context-task 依赖敏感性：另以 `(dataset, context, perturbation)` 整簇重抽，但不替代更严格的 perturbation-cluster 主区间。
- pooled-median 仅为敏感性：每个 `(dataset, context, perturbation)` 输出一行；重复 outer-fold 记录的分数和端点分别取中位数。它不是 E140 原 estimand，也不是新的 pooled 主分析。
- 两个 estimand 分别进行七研究随机效应：研究效应经 Fisher z 变换；tau² 用 REML，I² 用 Cochran Q，均值区间用 modified Knapp–Hartung，prediction interval 使用 t 分布。
- E140 fold-macro 的研究内标准误来自 perturbation-cluster bootstrap；pooled-median sensitivity 的关联标准误使用经典 Fisher-z 近似、配对差值标准误使用 context-task bootstrap。
- SafeConf 与比较器的效应为 `atanh(r_safeconf)-atanh(r_comparator)`，不可解释为原始 Δrho。
- LODO：对两个 estimand 每次删除一个完整研究后重算相同随机效应模型。
- Nadig 方向风险按 perturbation 整簇同步 HepG2、Jurkat 及其 outer-fold 记录；context-task pooled median 仅作单独敏感性。方向结果绝不并入七研究 absolute 元分析。
- 所有 bootstrap 固定 `3000` 次，主随机种子 `202607146`。

## 冻结规模

| dataset | n_rows | n_folds | n_unique_context_tasks | n_unique_perturbations | row_to_context_task_ratio |
|---|---|---|---|---|---|
| Frangieh | 837 | 3 | 567 | 189 | 1.4762 |
| Lara_exvivo | 345 | 5 | 155 | 31 | 2.2258 |
| Liang | 612 | 9 | 162 | 18 | 3.7778 |
| Nadig_two_cellline | 256 | 2 | 192 | 96 | 1.3333 |
| Santinha | 255 | 5 | 115 | 23 | 2.2174 |
| Shifrut | 172 | 4 | 80 | 20 | 2.1500 |
| Tian_CRISPRi | 732 | 4 | 396 | 99 | 1.8485 |
