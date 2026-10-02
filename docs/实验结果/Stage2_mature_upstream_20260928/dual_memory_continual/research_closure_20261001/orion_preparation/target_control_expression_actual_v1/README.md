# 固定目标基因TRAIN控制表达强对照：实际结果

已见Orion固定232任务/144gene，单一方向高表达=高风险；用原own-context TRAIN NTC mean_cell_logCP4000[target_gene_id]评分。不是新主方法、变换候选、确认或参数搜索。初始Root消息的CP10000单位已在任何分数生成前纠正为原科学合同CP4000，没有换算已平均的log表达。

## 原控制与身份复算

两context的完整38606基因NC侧car与原MODEL.rds baseline逐位一致；原3285投影也逐位一致。全部232任务保留，其中145个target不在原3285readout，不能缩成87任务或补零。它利用已经存在的完整控制输入，不读raw表达、treated truth或新的实验数据。R只readRDS/export原baseline，0fit/predict；准备阶段1.046706秒。

完整38606是原P6/PUBLIC7之外的实际可用生物坐标。因此本项是相同任务/误差预算下的强生物信息对照，不是完全相同特征条件的算法比较。owncontext控制不是query-specific plate/batch控制。Source/MC current轴缺target，停止其全量版本。

## 固定结果

Control macro U20=0.078304，CI[-0.092418,0.305068]；HCT116=0.153833，HEK293T=0.002775，各区间均跨零。

| 固定方法 | U20 | 相对control点差 | 名义95%配对CI |
|---|---:|---:|---|
| Magnitude | 0.077116 | -0.001187 | [-0.139833, 0.136044] |
| Learned_DirectRMSE | 0.267066 | +0.188762 | [-0.171296, 0.461058] |
| Learned_hgb | 0.204835 | +0.126532 | [-0.216172, 0.416231] |
| B100_PublicTarget_ridge | 0.279075 | +0.200771 | [-0.126186, 0.424130] |
| B100_SharedTarget_ridge | 0.243019 | +0.164715 | [-0.120531, 0.423764] |

全部43方法−control宏U20对照没有正区间；只有NegativeSourceHistorySupport差值区间全负。这是描述性名义区间，不按43结果选赢家，也不作为多重校正后的优越性结论。Public参照/部分学习器的点估计更高，说明该单标量没有重现全部观察到的点差；区间仍允许零和负增量，尚未排除control解释或证明Public/Source稳定增量。

30Target方法继承既有2790validation任务/1661gene的监督和各预算CDF；本统计0fit不代表它们零C标签。原13主方法、Source/LM/NC、HCT安全失败和原科学结论完整保留。不反转符号、不加参数、不分有利子集。

## 配对统计与实际成本

原13＋既有30Target＋1control共44方法，924metric/903paired行。复用保存的同一5000×144排序gene索引：原Control33和StrongRef4指标数组复用，只补6原规则＋control；每draw原LearnedHGB anchor在1e-15内复现。独立代理从保存数组精确复现所有924/903区间，无新采样或拟合。

所有U20是5000有效draw。P_only_hgb在HEK与macro Spearman仅4352有效，因为648次抽样分数退化为同值；NA原样传播，没有补0或删除任务。其余922metric/901paired有5000有效draw。初次Root辅助验收错误要求全部指标5000，已修正验收逻辑为保留合法NA，不改统计数据。

统计实际97.388010秒、CPU97.417870秒、报告ru_maxrss567353344bytes；0Source/upstream/model调用、0新fit/下载/GPU/表达读取/训练误差标签。缓存TEST误差只作已见评价；完整性SHA可能读取bytes，不伪称新的sealed确认。

负责人决定：这组便宜强对照已经完成，停止符号/参数/范围变体，保留Public较高点估计和不确定性。不能由本项宣布方法成熟、修复HCT门或Source监督跨家族迁移。论文正文仍按用户要求暂缓。
