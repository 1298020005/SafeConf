# Methods 实验合同（证据规格，不是论文正文）

- 主问题：风险层没有当前预测器错误记录时，公共真实扰动实验能提供什么审核能力；反馈回来后还有什么增量。
- McFaline：212 SEEN任务/152基因；331反馈任务/228基因；542开发任务单列准备成本。DecoderOnly和SAMS预测已冻结。
- KOLF：600上游训练、300开发反馈、300确认基因。基于固定scGPT基因表示、仅训练基因PCA50的Ridge通过原能力门；这不是完整scGPT扰动预测模型。MLP未通过方差门。
- KOLF主预测/真值/能力门：1400注册输出基因的pooled-count log1pCP10k delta。GWPS实际缺34输出基因；Public比较仅在实际共同1366轴，预测delta+NTC经inverse-log和相同轴重归一化。缺失不补零，主端点不截轴。负log表达只在Public比较视图投影到0并另作诊断。
- Public：Replogle K562 GWPS；KOLF研究排除。一个历史screen/context每扰动，单历史分散度0不是实测独立重复质量。
- Public可用时用公共距离的训练侧featureCDF；无历史时用幅度的featureCDF。CDF不拟合错误标签，但不是相同百分位即相同真实风险。全300任务唯一排序。
- 上游资格门读取300开发误差，单列准备成本；Public规则风险拟合/选型/校准的KOLF错误使用数0。不要把这写成整个流程未读任何KOLF答案。
- 反馈候选是固定PertEMA代码配方的XGBoost输入适配，包含prediction、control、NTC-onlySVD50、相似性及Public增量。所有臂相同任务/真值/预算/簇权重。尚未完成官方conformal和区间流程。
- 反馈预算10/25/50/75/100%，10个预固定顺序、3个固定种子；它们是算法稳定性不是30个独立生物样本。
- 主复核20%，辅5/10/30%；主误差deltaRMSE，辅1-Pearson和评分侧真实效应Top200RMSE。Top200不进入模型。
- Utility以oracle选出同数量高误差任务的收益归一化。真实top20%严重错误集合与review预算分别定义。宏平均与全量global分别报告。
- 5000次基因簇配对bootstrap，同一抽样同步应用所有背景/算法运行。区间条件于已拟合模型和本研究，不代替新研究间不确定性。
- 风险分数、模型及哈希冻结后一次打开confirmation；当前证据包只读取保存评分，新增拟合0、GPU0、下载0。
