# 投稿目标对齐

首选目标是 IEEE/ACM Transactions on Computational Biology and Bioinformatics（TCBB）。中国计算机学会当前推荐目录把 TCBB 列入交叉/综合/新兴方向，见官方目录：

https://www.ccf.org.cn/Academic_Evaluation/By_category/

TCBB 的官方介绍强调计算生物学与生物信息学中的算法、系统和实证研究，SafeConf 的方法框架、风险排序、信息预算和单细胞扰动验证与这个范围匹配：

https://ieeecs-media.computer.org/assets/pdf/tcbb_intro.pdf

稿件应围绕一个计算问题展开：已有真实扰动实验能否在没有当前预测器错误标签时启动风险审核，以及少量反馈如何改变审核排序。实验部分需要同时呈现：

- PublicRule 相对幅度、支持量和历史能量基线的同任务比较；
- McFaline 两种预测器的主要结果；
- Frangieh GEARS/scGPT 跨家族压力证据及其 predictor competence 边界；
- KOLF 单一合格 predictor 的独立确认、混合历史覆盖和无历史回退；
- Target/Source/PertEMA 候选的监督成本与不采用理由。

当前默认方法、任务划分、统计端点和信息账本已经冻结。后续写作必须保持这些配置，不在确认结果之后重新挑选方法。
