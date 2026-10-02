# 实际Source目标对照：停止该变体

用户在具体20fit请求后将技术判断交给负责人。本次仅执行隔离Source DEV诊断，不扩展为正式方法、旧外部TEST评分或新上游。

v1一rank fit因原float32 truth与CSV decimal比较失败停止；模型分数本身复现。原代码/输出保留，v2只修codec，全部3616query恢复float32位级一致。第一rank参数复用，19新＋1原fit，累计20。20模型/全部scores在指标汇总前封存，29输入code/core SHA不变。无新原始表达、外部numeric rows、GPU或上游训练。

两方向U20原rank0.783134/0.783743，raw-affine0.774848/0.780253，差−0.008287/−0.003490；Spearman也均降低。40fold/context有20非负，未达到稳定一致改善。AURC/error@10小幅改善完整保留，但不代替主指标。没有bootstrap，不称显著变差/等价，也不认定跨域失败根因。组内仿射改变pool loss组权重已记录。

负责人决定：原CDF目标及正式core保留，停止raw/log/quantile/尺度目标搜索。数学错配是可能性，当前具体干预未提供主指标收益。该诊断不升级论文成功门，下一步研究取舍仍须以Public内容、Source强规则增量和独立研究的真实证据为准。

独立审阅INDEPENDENT_ACTUAL_REVIEW.md复算全部48context/12macro/240fold-context×7指标、原56macro点、40训练组变换、模型复用、参数/分数SHA及输入边界。v2CPU wall5.784s，v1 wall1.446s。完整预测、参数仍保存在runtime，Git仅摘要与复现代码。
