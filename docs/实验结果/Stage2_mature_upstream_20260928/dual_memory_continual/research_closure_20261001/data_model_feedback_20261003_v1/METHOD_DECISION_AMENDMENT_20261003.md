# SafeConf 方法采用增补（2026-10-03）

这份文件是实验阶段的采用决定增补，不是论文正文，也不是永久评价集确认。它把已完成的 GWPS、反馈和 ErrorResidualAdapter 结果与原有采用决定接起来；SAMS-VAE 仍在训练，跨预测机制的最后一项证据尚未闭合。

## 当前采用层级

### 1. 主干：幅度 + 合法公共历史距离

公共历史首先经过输出合同、任务排除、背景匹配和支持加权。当前最稳定的风险起点是：

- 当前预测幅度作为无历史时的起点；
- 有合法同背景历史时，使用预测与历史参照的加权均方距离；
- 历史分散度保留为诊断量，不把细胞数直接当质量；
- 没有合法历史时返回幅度和 `no_history` 状态，不伪造零风险。

GWPS 扩展实际增加了 1,147 个相对旧全目标历史的新任务。扩展覆盖任务上，公共 direct-distance 的 U20 点估计为 0.307930，bootstrap 均值增量相对幅度为 0.162896，95% CI [0.034534, 0.291658]。同一范围的固定 HGB 增量 CI 跨零，因此不能把 HGB 写成扩展库的稳定替代。

### 2. Public 学习器：不晋升 DeepSets 或 pointwise 网络

公共生物内容置乱仍显示真实效应向量被利用，但 DeepSets 相对强简单方法没有稳定增量。当前不再扩 Transformer、GNN 或门控；公共学习器作为合同内既有组件保留，主默认使用可解释的历史距离规则。

### 3. Source Error Supervision：条件性组件

源错误监督可在已登记的 TxPert 线路中训练共享风险分数，也保留真实标签和标签置乱对照。现有跨研究结果没有稳定超过强简单历史距离，因此不把 source HGB 宣称为普遍主方法。SAMS-VAE 仅用于检验不同预测机制之间的错误经验迁移，结果出来前不提升主张等级。

### 4. Target feedback：按门启用，不按反馈数自动启用

开发选型得到 F1 + 小样本 HGB（H1），但固定评价上只有 10% 反馈预算相对旧 HGB 通过采用门，25–100% 区间未形成稳定优势，且强无反馈历史规则仍为 U20 0.837668。它可以作为低反馈候选，但当前不替换公共规则。

独立 ErrorResidualAdapter 生命周期已经真实闭合：两批 exact-version Error Memory 追加、adapter v1/v2 注册、预定 DEV gate、失败候选保留和服务重载都通过。adapter v2 的反馈 gate U20 增量为 −0.03586，候选被拒绝；这证明更新机制可运行，不证明更新有性能收益。

## 最终论文主线的证据状态

当前最可靠的贡献候选是：**把合法公共真实实验转成跨任务的后置风险参照，并明确报告其覆盖、背景匹配和无历史边界。** GWPS 提供了新增覆盖和独立顺序增长证据；反馈和源错误层提供条件性扩展及失败边界。

SAMS-VAE 训练结束后只需回答一个问题：不同预测机制的合法错误是否能在强公共距离之上提供可迁移增量。若不能，最终模型仍保持公共距离主干，source/target 作为可更新但受门控的扩展；若能，才把跨预测器错误监督提升为第二贡献。

## 不得做的解释

- 不把重建 loss 当作上游预测能力；
- 不把反馈学习器超过 Target-only 写成超过强公共规则；
- 不把 adapter v2 被拒绝写成 Error Memory 无法工作；
- 不把两个同家族模型的错误相关写成跨家族迁移；
- 不重新打开永久评价集来选择 SAMS 或任何新参数。

## 证据入口

- GWPS：`gwps/ALL_METRICS.csv`、`gwps/ACTUAL_COVERAGE_AMENDMENT.csv`、`gwps/PUBLIC_GROWTH_5_FIXED_ORDERS.csv`
- 反馈：`feedback/ADOPTION_GATES.csv`、`feedback/PAIRED_COMPARISONS.csv`、`feedback/SELECTION_LOCKED_BEFORE_HOLDOUT.json`
- 更新：`error_adapter_replay_v1/COMPLETION.json`、`error_adapter_replay_v1/METRICS.csv`、`error_adapter_replay_v1/INFORMATION_BUDGET_LEDGER.json`
- SAMS：`sams/TRAINING_STATUS.json` 及训练完成后的真预测回执
