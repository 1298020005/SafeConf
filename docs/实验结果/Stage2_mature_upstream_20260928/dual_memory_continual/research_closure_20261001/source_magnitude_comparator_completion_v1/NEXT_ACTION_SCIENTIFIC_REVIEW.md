# 下一项科学动作：只允许一个 Source DEV 目标函数对照

## 当前判断

现有授权下，没有一个尚缺的“不变方法补评”能补成稳定的独立研究增量。强对照、目标反馈、物理内容、控制表达以及实际 PublicBiology 重训闭环均已完成；继续补相似诊断不能替代新确认。建议停止这些分支，只提交下面一个需明确批准的 DEV 方法目标对照。本次审阅未拟合、未采样、未读取新表达或封存真值，未改共享记录。

- Source LearnedHGB−Magnitude 两方向为 +0.034327，CI[-0.009285,0.069899]，以及 +0.037491，CI[-0.002974,0.077648]；另外四个 Manual/P-only 比较也跨零。真实标签优于五个同容量 shuffle 支持源错误标签对应关系，不能据此把 Public 内容、HGB 灵活性或两个同家族架构算成独立迁移增量。证据：本目录 `README.md`、`ALL6_FIXED_PAIRED_COMPARATORS.csv`。
- E170 known-history 的 V2−Direct 为 +0.029478，CI[-0.018512,0.090661]；−DistancePlusDispersion 为 +0.024281，CI[-0.028890,0.084104]。其1920目标 validation errors覆盖与known TEST相同640genes、不同donor，不能认证当前零目标错误标签的 Shared Core。证据：`../e170_legacy_strong_references_v1/README.md`及独立实际审阅。
- Orion 原主差 −0.010052，CI[-0.271236,0.191339]，HCT安全门未过；MC有些增长/内容点和区间为正，但最终仍未超过强历史规则。PublicBio DEV重训取得小幅生物重建改善而风险门拒绝，已经闭合更新操作缺口。以上不等于方法无效或等价，也没有证明“绝对幅度是唯一失败原因”。当前 `METHOD_DECISION.md`、`EXECUTION_STATUS.json`、`CLAIM_EVIDENCE_MATRIX.md`正确保留这些限制。

## 唯一有判别价值的开发问题

`research.rank_labels`按训练集CDF产生中秩标签；`fit_risk`用平方损失学习该标签。它近似学习 E[F_train(E)|X]。原 `metrics` 的 U20 则选最高风险20%，分子计算所选任务的**原始 RMSE**均值；同一评价队列的分母对各方法固定。CDF期望排序与原始RMSE条件期望排序不保证相同。因此“目标函数可能不适配主选择指标”是合法、具体的假设，尚不是实际跨域失败的解释。CDF单调性只保证单个实测误差的次序；不能据此推出两个条件期望同序。单调校准旧分数也不会产生新的排序信息。

## 最小可审阅范围（Root选择的20-fit方案；尚未执行）

1. 只用当前2840 Source DEV：GAT→Exphormer、Exphormer→GAT，五个原全局gene外折；每方向全部1808 tasks/575 genes/四context。训练仅来源架构且排除held-out genes，评价仅目标架构held-out fold。同gene的两个架构不增加独立生物簇或研究数。
2. 输入为 `/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/risk_cache/nested_{0..4}_TxPert_{GAT,Exphormer}_Learned.parquet` 和 `TX_TASK_SPLIT.csv`；复用已保存的嵌套 Learned13 特征，不重训 Public、不加载生物向量、不新建预测器。每外折训练身份、特征和预算两分支完全相同。
3. 固定原HGB参数200/.05/depth3/leaf20/L2=10、seed20260930及mean-one gene weights。10个原rank分支用于复现冻结主矩阵分数，10个预登记正仿射 raw-error 标签分支；不搜索尺度、损失、特征或参数。旧outer风险模型未保存，所以rank重训是控制复现，不是新候选选择。rank复现失败即停止，不能靠放宽容差继续。
4. 仿射的斜率/截距只能由本折允许训练误差确定，必须在运行前写明。若每CDF组使用不同斜率，须称“组内标准化raw-error目标”，不能说与全pool原始RMSE平方损失完全等价，因为组间有效损失权重改变。新分支不以非严格单调clipping制造ties；原rank执行路径保持不变。
5. 两个完整方向均报告原七指标/四context等权macro，固定 affine−rank、affine−Magnitude、affine−同Learned先验距离；不挑context、折或胜者。全部分数封存后才做DEV评价，复用已经保存的5000×575 gene counts及原sortedgene身份，不产生新RNG。保留合法Spearman NA/valid计数。新隔离路径、输入及正式资产before/after SHA，禁止外部MC/Orion/E170评分、发布或覆盖。

预计20个小表HGB的拟合为秒级至分钟级；完整统计可限定10分钟、4 CPU线程、2 GiB、0GPU/下载。一次生命周期raw-target拟合虽更便宜，但混入不同更新版本，不能替代这套与当前迁移主矩阵相同的配对复现；不并行执行它。

## 授权与结论上限

**需要明确批准：**在独立DEV版本中暂时例外于冻结的CDF秩标签规则，允许上述唯一训练目标对照及20次固定风险拟合。实现/预登记可供审阅；真实拟合不能把本审阅或原持续更新授权当成标签规则变更的批准。

即使DEV区间支持改善，它最多说明该固定特征/学习器对目标定义敏感，并提供一个待确认的方法修订；不证明原Orion失败原因、不恢复旧安全门、不构成新的跨研究增量。若证据仍不足，结束本对照，不追加raw/log-error/quantile等变体。任何论文级外部改进结论仍缺一个事前冻结且独立于已开封MC/Orion/E170的新确认；本轮限制下无法补此缺口。
