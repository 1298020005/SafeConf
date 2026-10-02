# E170固定强规则：独立实际核验

**PASS。未发现当前统计器的实质数值、行对应或配对错误。** 本次只读当前代码、registration／score freeze／输出、已保存draws和既有身份接口；0拟合／新采样／模型调用，未读取新表达、validation error数值或新封存truth，未改Root文件。

执行代码SHA：`2093ea2c062997a77e383b5d87da35a9bdc30d5cd7960ca658067868b1349780`。实际28.047073秒、0fits，23macro／276strata／161paired rows。

## 实际完整性与数学核验

- registration全部input/code bindings及实际7个runtime artifacts的SHA／size一致；cohort/method/metric keys无重复。三个cohort保存draws的method轴、7个metric轴和5000次draw形状一致；每panel count row总数分别200／160／40，保持同gene全部三states一起抽样。
- 全部**161个配对CI和配对point**从已有draws逐项精确复现；23个macro point与保存point arrays精确一致。276strata完整保留，没重采样或再次解析query error数值计算指标。
- 原6方法×7个Full macro点与原SUMMARY最大差2.22e-16；Full V2 U20仍`0.2158954463535431`。原2400表、预测与GateA没有被known1920成绩替换。旧CI与新cohort的seed不同，不声称旧CI复现；不同cohort使用各自固定新draws，不跨cohort相减CI。
- 2400 sealed score行与原feature任务顺序相同。按执行代码使用的原default CSV parser复算，Direct=`−negative_model_source_gap`、Weighted=`sqrt(gap²+dispersion²)`及Support=`−log1p(n_source_cells)`全部逐位一致。另用round_trip解析旧feature会引入ULP差，不能把不同parser的差当作执行错误。
- 原history_maps对同gene／condition的两个train-donor effect等权取mean，dispersion是其RMS离散度。Direct和Weighted忠实复用这套native512 panel scalar，不涉及新跨研究向量或Source2840／3285合同。旧Relevance已含prediction-reference cosine和负distance（原confirmation代码149，development RELEVANCE／USR定义），不是新引入的学习特征。

## NA与支持数附加scope

Direct／Weighted恰有480个无历史任务为NA，没有补零或从原2400评价删除。Support全域附加控制在运行前registration已单独明示：0是观测到没有历史，不是零effect，`−log1p(0)=−0`合法。Full／No-history上只增加该元数据规则，不让无历史Direct／Weighted混入有限共同集。

实际每个known-history panel×state的`n_source_cells`也只有一个取值。因此Support在known内没有目标间排序信息；Full主要反映有／无历史的差别，不能称为检验了known内不同样本量或测量精度。Known的V2−Support、Support−Magnitude及No-history的V2−Support三项Spearman均合法NA、0/5000 valid draws，实际输出准确保留。其他158配对项均5000有效。constant score得到的U20可受固定task-ID tie规则影响，不应将其点估计解释为生物排序能力。

## Gene overlap与信息合同实际确认

只读四个既有`pretruth_release/{P01..P04}/tables/PRETRUTH_SCORING_INTERFACE.csv`身份列：每panel validation480tasks／160genes，known test480tasks／相同160genes；column-unseen120tasks／40genes，与validation无重叠。全局640个validation genes与640个known-test genes完全一致，160个unseen genes不交；四panel之间gene ID不重复。validation donor为`CE0008678`，test donor为`CE0010866`。

原授权合同49／72行明确用validation-donor错误拟合风险模型，测试为另一donor；gene重叠不违反这个协议。但旧6个监督风险方法已消费1920条目标ensemble validation errors，不能改称Source-only／零C风险标签或gene-disjoint风险迁移。它也是同一研究native512、已有six-model ensemble的评价，不认证当前冻结Shared Source core、独立新family或新外部确认。

## 配对结果的准确范围

known1920原V2 U20=`0.272010`，Direct=`0.242532`、Weighted=`0.247730`；V2−Direct `+0.029478 [-0.018512,+0.090661]`，V2−Weighted `+0.024281 [-0.028890,+0.084104]`。两区间都跨零，不能从原GateA point PASS推出已经稳定超过强历史规则，也不能把宽区间说成等价。V1／Ridge_USR仍是固定次比较，不升级为新主方法。

原封存V4的证据定位可收紧为：已完成同研究held-donor协议的固定评价，监督模型包含目标验证信息，尚未建立相对同队列历史几何规则的稳定增量。新补评不改变原GateA、方法、模型或历史确认身份。

精确证据：registration `27130f208aea2675257c008cd562078cb161d147c4295b45243bc6e12a7e6331`；score freeze `9e6a71b61b56c2a249146a01233faaafb879fd13b01121e7c09dc87eadd2b31a`；actual result `77d6f3173bfd6f9b90358a949a6e239900d65d349a3c03d779b7d6157caed085`；paired CSV `15527522cac6cbc8b014bde37d6159f65d7b4c61fd62a674dbd27440a3d99fc6`。
