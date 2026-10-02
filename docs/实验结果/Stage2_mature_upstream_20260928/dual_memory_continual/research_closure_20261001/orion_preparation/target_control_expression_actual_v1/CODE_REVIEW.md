# 固定 target-control 统计器：独立代码审阅

**PASS（限定只读代码审阅）。未发现需要停止当前 session90383 的实质行对应、draw alias 或条件化错误。**

审阅文件：`tools/scripts/evaluate_safeconf_orion_target_control_baseline.py`。
SHA256：`3858b4c111635f648b70ab5a7c74d92ea4047529a8e5320146a5bb4131eebd94`。
同时核对原validation-control、strong-reference completer和dec8指标代码的字段／顺序定义。本次未运行统计器、读取error数值或模型／draw数组，未重采样、拟合或修改既有文件。

## 核验

- **全队列行对应。** 从已封存232 control scores出发，以全部5个query identity字段进行一对一合并，保留左侧顺序；要求232唯一query、13原规则、30原target learners，全部分数及errors finite。没有补零、按性能／readout覆盖筛行或选择方向。
- **保存draw的身份。** 以同canonical sorted144个Ensembl gene IDs构造block，并核对Control的完整task集合；同一gene在两个context的行共同进入每个saved draw。这里只遍历已有5000×144indices，没有创建bootstrap RNG。
- **别名和轴顺序。** 30个Target方法按原`ALL_PREDICTIONS_SEAL.score_columns`索引33-method draw轴。三别名`Learned_hgb→FrozenLearnedSourceHGB`、`Learned_WeightedHistoryDistance→LearnedWeightedDistance`、`NegativeSourceHistorySupport→NegativeSourceSupport`与原control定义相同。四个Direct／ManualWeighted reference轴顺序与strong-reference completer的`NEW_REFERENCES`完全一致；它们原先即使用同一saved5000indices计算，并以原Weighted anchor逐draw核对。
- **缺失原规则的补算。** 仅计算Magnitude、P-only Ridge/HGB、Manual Ridge/HGB、Learned Ridge和新control共7列的metric draws；每draw、每context及macro再次核对原LearnedHGB anchor（原数值复算容差1e-15，不涉及风险分数／方法修改）。其他33＋4方法复用保存draw。
- **固定配对方向与完整性。** 全43 comparators都报告`method−control`，直接相减同draw指标，未相减独立CI。原helper处理两个context各n≥20、等权macro及未定义指标；每行保留valid draws。全部44×3×7=924单方法行、43×3×7=903配对行必须存在，无winner／seed／预算选择。
- **封存和依赖。** 新control preparation的输入、代码、模型、输出绑定先检查，原dec8指标SHA不变，现有score/draw/truth等文件执行前后SHA/size一致检查保留。新分数此前已在0error数值访问阶段完成封存；本统计器只使用已开封SEEN缓存，不重新打开raw TEST表达或产生新预测。

## 信息预算及解释须保持

新control是现有38606-gene own-context TRAIN NTC均值中的target标量，高表达固定为高风险，规范单位为CP4000；不换算均值、不试反号。它使用原3285 readout／P6+PUBLIC7之外的真实可用生物输入坐标，因此比较不能称完全相同feature信息预算或算法优越性。

“0 new fit／new C error labels for training”指本次统计阶段的增量成本。30个Target learners此前已经使用2790个validation tasks／1661 genes，按各固定预算拟合；不是零Target标签方法，也不是单一checkpoint的版本化残差ErrorAdapter。它们是同一published C family两个context checkpoint上的既有目标研究风险学习补充。原13方法、新control和这些target-feedback arms的信息条件必须分别披露。

这是SEEN同任务辅助比较，不改变原13主比较、Source参数或安全gate，不产生新确认资格；名义CI不能用于事后挑最佳control方向、Target预算或方法。本静态PASS不替代实际终态输出核验。
