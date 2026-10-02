# Orion固定物理历史内容负对照：实际结论

本项是在已见Orion原232任务/144gene上，按原3项人工/均匀参照规则执行固定5种子的辅助诊断。不改变Source参数、原13主比较、任务集合、指标或确认资格。

## 实际完成

v1准备因分数序列化末位差异而失败，原acba源码/ABORT保留。v2按原17g字符串→pd.to_numeric codec复现464个query×reference，原几何、权重、分散度和输出分数全部bitwise匹配，未放松容差或数学。随后Root正式器先复现3原score，再在canonical study/context/type/effect/gene-space/control/n_batches/Sourcecellcountquartile匹配组内循环交换实际效应向量。

每种子5200/5365bank行移动、全部5365构成一对一排列；不交换已经算好的风险列。原recipient支持、查询历史ID、资格与权重保持。全部232任务保留，3个无可移动历史的任务不删除；全部18分数先落盘封存后才解析已有cached TEST误差。为完整性SHA会提前流式读取该缓存文件bytes，故准确说法是封存前不解析/使用truth数值，不是文件从未打开。

## 结果

| 原固定规则 | 真实U20 | 五次置换U20范围 | 名义正区间数／5 |
|---|---:|---|---:|
| Uniform_DirectRMSE | 0.262120 | [-0.095331, 0.130880] | 2/5 |
| Manual_DirectRMSE | 0.244031 | [-0.176300, -0.007733] | 2/5 |
| Manual_WeightedHistoryDistance | 0.186661 | [-0.039078, 0.179600] | 0/5 |

15项宏U20点差均正，只有4个名义95%配对区间完全为正，来自两Direct规则的种子20261001/20261003。ManualWeighted五次均跨零；HEK293T所有15区间均跨零，其中Weighted两项点差为负。不能挑正种子、将15相关比较算作独立复制，或用本诊断修复原HCT安全门。五个固定置换不是大型随机化显著性检验，名义区间未多重校正。

与既有McFaline真实内容胜5null相比，第二研究的方向相同但精度较弱。它支持固定队列中历史gene-effect对应关系的描述性价值，不能证明生物内容稳定优于数量信息、隔离纯测量/生物因果、Source error跨研究主增量或完整方法成熟。参照构建不同的两个研究不直接合并RawRMSE或CI。

## 支持匹配与统计复核

232查询实际使用338个unique历史recipient。各置换中96.4497–98.2249%的donor/recipient cellcount ratio在[0.8,1.25]，99.7041–100%在[0.5,2]；没有逐值或潜在方差匹配，Source批次数也是记录代理，非生物重复证明。所有ratio及3个不动任务完整保留，不事后扫描caliper。

复用已保存的5000×144排序gene索引，全部方法同一抽样；未生成新的bootstrap抽样。378项metric、315项paired行已完成；Root核对3项原规则的逐次指标数组与上一轮strong-reference补算在1e-15内一致。独立代理从已有draw精确复现全部315配对和378单方法CI，没有重新采样/拟合或读取truth数值。

正式诊断235.777741秒、CPU235.799281秒、报告ru_maxrss567353344bytes；0fit/模型推理/新下载/GPU/表达读取，只有5组明确标注的counterfactual donor映射。大数组/逐任务分数和donor映射保留在runtime。独立实际审阅见 `INDEPENDENT_ACTUAL_REVIEW.md`。

负责人决定：保留全部5种子和有限正证据，停止此诊断的置换/匹配变体；原Source跨研究主比较结论不变。另已完成只读target-gene control-expression简单对照资格审计：仅Orion的完整预测前TRAIN控制域覆盖全队列；Source/MC当前readout不足，停止全量版本、不补零。正式该对照若执行须登记完整38606控制输入及额外信息条件，不能混写为仅P13输入。论文正文暂缓，科学投稿就绪仍未证明。
