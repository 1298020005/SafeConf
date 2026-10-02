# 新嵌套统计和内容置乱的只读复核

本补充只读代码与既有审计回执，没有调用模型或读取新的表达/TEST。运行结果尚须以完成回执核实。

1. `run_safeconf_publicset_risk_followup_v1.py:feature_frame` 的Universal P恢复了绝对稀疏比例 `abs(p)<=1e-8` 和无floor的raw绝对q95，与原 `seal_mcfaline_dual_memory_risk.py:prediction_features` 一致。新目录隔离旧relative-sparsity的160fits；Public参数/priors可复用，无需神经重训。
2. `paired_statistics` 的观测U20按5fold×4context等权宏，再平均各seed指标；pooled四context另列supp。误差、安全macro与stratum门采用同20分层，worst strata透明报告。逐方向重置同固定随机种子，575gene名单相同，故两个方向也使用相同gene抽样。
3. `weighted_draws` 中 `counts[:,high]` 是 draw×task×score；沿task轴累积次数，`min(max(k-prior,0), count)` 精确截断top ceil(.2N)，包括重复task跨top边界的情况。Oracle分母同规则。每个task先按ID稳定排序，重复copy得分和误差相同，等值边界保持原语义。
4. `gene_counts` 按原rng逐draw整数抽样生成并保存；代码保留前3draw在primary/supp两种聚合与显式重复展开比较≤1e-12的运行检查。静态检查通过；最终证据应绑定实际展开差值与完成5000draw回执。
5. `run_safeconf_publicset_content_null_v1.py:shuffled_domain` 对actualfit bank基因、querygene禁止、same physical context和fit-derived logcells decile逐项设断言。query供体只能来自fit bank；原query parent及当前query context的treated实验不能提供内容。125个B1 scope的实际供体合法性回执为PASS/0坏scope。
6. 所有recipient元数据、control、支持权重及group membership逐值验证保留；替换effect后重新 `Domain.groups`、PCA transform、冻结builder打分、权重、μ、dispersion及discrepancy。risk重拟合核对real训练task及CDF标签一致。此检验是**冻结real生物构建器条件下的内容置乱**，不声称生物训练全流程null；供体支持仅按decile匹配，不能排除全部精度/噪声因素。
7. 原outer B1未保存pickle，null准备从原HGB_INPUT与SCORES复现五个real B1拟合。它们是新增实际生物拟合，应入成本；并非null标签拟合。保存实际max差值，浮点环境差异不能称逐字节完全一致。
8. 论文§5.8–5.10/S7对542 vs543任务、old-guide对齐、Source same-context fallback=B0、强简单规则以及fixed-R_hist STOP边界的解释符合代码。论文负责人已修正§3.3/S7：B0零fit，B1原逐history RMSE/pair-row损失，只有B2/B3共用gene-balanced query reconstruction MSE；§5.9明确B2−B1包括目标函数变化，只有B3−B2隔离集合上下文。再次读取已确认修正。

新版结果、物理置乱、最终方法采用决定仍按完成产物核验；不由本次静态复核提前宣称合同完成。
