# Source：固定HGB−Magnitude强对照补齐

实际完成，PID924463／session77062，13.195702秒，CPU34.624176秒，峰值519344128bytes。0模型／CDF拟合、0新采样、0GPU、0Orion／MC数值访问、0raw gene vectors。源主矩阵和所有方法参数保持原版本。

两个Source方向，原seed20260930；每条完整1808 tasks／575 genes／4 contexts，固定Learned HGB、Manual HGB、P-only HGB相对Magnitude共6比较。已输出210配对行、280单方法区间，包含4context及等权macro的全部7指标。每项scalar point均对主表和counter helper复算，最大差分别2.22e-16／8.88e-16。

## 固定macro U20结果

| Source方向 | 方法−Magnitude | 点差 | 名义95% paired gene CI |
|---|---|---:|---|
| Exphormer→GAT | Learned HGB | +0.034327 | [-0.009285,+0.069899] |
| Exphormer→GAT | Manual HGB | +0.031741 | [-0.017315,+0.058492] |
| Exphormer→GAT | P-only HGB | +0.010145 | [-0.036005,+0.041538] |
| GAT→Exphormer | Learned HGB | +0.037491 | [-0.002974,+0.077648] |
| GAT→Exphormer | Manual HGB | +0.039755 | [-0.003418,+0.073826] |
| GAT→Exphormer | P-only HGB | +0.006461 | [-0.020953,+0.047175] |

全部6个macro U20区间跨零。真实标签对比五个label-shuffle的已有正证据仍支持标签对应关系有信息；HGB胜同参照距离的同家族结果也保留。但距离不是这个Source域中最强无标签规则，现有证据尚未建立对Magnitude的稳定优势，更不能将全部收益归给Public内容或升级跨家族迁移结论。CI跨零不表示等价或无效。

## 保存draw与访问范围

原main bootstrap仅保存区间，没有保存draw数组。本补项复用已实际存在的`common_public_growth_statistics_20261002_v1/Source_BOOTSTRAP_GENE_COUNTS.npy`，5000×575 uint16，SHA `524ac5396accdf0892196cddcd2694a1cbf159eb0b8aff57f9c99a41ebd3908e`；sortedgeneOrderHash `f6a9d3cb6e0c2e6ba3630fb3df066392324ef8f34ebc3999001154079ebdc6c3`。其原生成seed20261002不同于旧主CI；因此不声称复现旧主CI。两个Source方向及全部方法使用相同已有gene counts，不产生RNG。

合法NA保留并按原macro传播，只对有限draw求区间；本轮全部210配对均有5000有效draw。所有固定比较都展示，没有选择种子、方法、context或子集。共同truth只是原Source观测错误，两个同家族架构的派生error不增加独立生物真值／研究数。

共享MATRIX CSV的字节绑定／扫描包含原文件；在数值解析前依据line、seed、method元数据过滤，只解析允许的Source记录，MC标量未数值解析。输入／代码执行前后SHA与size一致。所有上下文点和macro point复现没有新标签或CDF训练。

执行脚本`tools/scripts/complete_safeconf_source_magnitude_comparators_agent.py` SHA `4d5b97b9bd700a363a16b1874c013ef6e58e025d3bbc7e117763502fa1ae3d53`。实际回执SHA `43f74a73fea082dcaca1889e0510e6f1ca66e0ffea3a0cb03c4be070ed0c0680`；大draw数组位于runtime `source_magnitude_comparator_completion_20261002_v1/ALL_FIXED_SOURCE_METRIC_DRAWS.npz`。这只是SEEN固定强对照完整性补项，不是新主方法或确认。作者不冒称独立复核，Root另行审阅。
