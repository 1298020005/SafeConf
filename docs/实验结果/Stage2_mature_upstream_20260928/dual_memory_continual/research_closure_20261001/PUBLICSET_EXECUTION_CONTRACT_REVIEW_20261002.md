# PublicSet 执行合同：本机核验与负责人修订

依据：用户附件《SafeConf：新架构分阶段实验执行合同》与基准 aeb0e9e。状态：设计已核验到下列范围；本轮没有训练新模型。用户要求先把架构讲通、交由网页版讨论，允许适当放松时间。原72小时记录保留，新阶段实际启动时间单独记账。

## 1. 负责人结论与本机事实

四类信息的安排成立：预测信息（含幅度）＋模型无关的公共生物参照；源错误训练风险映射；目标错误用于后期个性化。首轮只改变Public构建器。Source residual是风险映射的替代训练方式，并不是在已有Source HGB之后再强制叠一层。新gate、GNN、Set Transformer和相对特征本轮不并行开发。

| 本机只读核验 | 实际结果 | 执行处理 |
|---|---|---|
| 硬件 | 2×Quadro RTX6000，24GB/卡；96逻辑CPU，约128GB内存 | 足够承担小网络；全量时长仍按单折实测 |
| 默认Python | Torch2.11/CUDA13与驱动不兼容，cuda_available=False | 不用作GPU工作环境 |
| 已有txpert环境 | `/home/yyf/.venvs/txpert-08d82eea/bin/python`，Torch2.6.0+cu124；GPU前向/反向烟测通过 | 固定PYTHONNOUSERSITE=1复用；无需升级驱动 |
| CPU/GPU库版本 | 默认sklearn1.8/numpy2.4，GPU环境sklearn1.5/numpy1.26 | 原HGB保持原CPU环境；CPU/GPU通过数组与JSON交换，不混读sklearn模型pickle |
| Source合法历史 | 1808任务/575gene；K=1:132，K=2:423，K=3:1253；4737边 | 是小集合；132任务保留评价，不期待重加权产生变化 |
| McFaline开发生物验证 | 542任务/377gene；固定7173条train/validation历史；K=7..14，其中K13=239、K14=219 | 可检验较丰富的集合关系；此计数没有读取TEST结果 |
| 原HGB输入 | Source已有conflict/support_fraction；MC已有真实重复性字段 | B2也保留这些摘要，不把它称为完全不含集合信息 |

实际独立性取决于实验单元/provenance，不以K或细胞数替代独立样本量。原始effect向量是否互不相同、近重复排除和每折有效监督范围，执行时写入HISTORY_SET_AUDIT。

## 2. 第一轮Public实现固定如下

四臂为B0支持加权均值、B1当前HGB、B2逐条MLP、B3带学习集合上下文的DeepSets。B0/B1仅在基因轴、任务、eligibility及数值读出完全一致时复用。B2/B3共享输入、训练目标、初始化种子、迭代上限和预处理。

- 输入：各生物数据域现有合法pair字段＋32维effect PCA；Source与MC分别拟合各自生物模型，同一任务的不同upstream复用一份参照。PCA仅用于评分，μ仍对完整2840维原始effect加权。模型不输入p、upstream ID、当前错误或当前真实效应。
- φ为两层64宽ReLU网络；评分头为128→64→1。B2上下文为64维零向量，B3为有效token编码的masked mean。无dropout/BatchNorm，实际参数量低于10万；有效与名义参数量均报告。
- `w=0.5 softmax(score)+0.5 support_weight`，沿用当前支持权重定义。不能将50:50解释成“质量与内容各一半”。重复实验先排除；padding不进入PCA、标准化、均值或softmax。
- PCA/标量预处理只用当前拟合分区允许的history/query行；PCA使用唯一history行，不能因多个upstream重复采样。各阶段保存拟合ID、scaler与PCA参数。Source未知Quality保持未知。
- loss为完整effect向量的query MSE；每个gene内query权重倒数归一，使每个gene的总训练权重相同。K1及相同向量集合的固定损失计入评价；若批次无可学习梯度，跳过该optimizer step并记录，不能借此删除评价任务。
- AdamW：lr=0.001、weight_decay=0.0001、batch32、最多100epoch、patience10、gradient clip1.0、FP32；种子20260930/20261001/20261002。PCA使用独立固定种子并在同分区的B2/B3间复用。
- 每个拟合范围再按gene划固定内部验证；仅用内部生物MSE定epoch，随后从头在允许全训练区按该epoch重拟合。内部阶段与重拟合均计成本。内存不足只允许减batch并用累积梯度保持有效batch32，禁止按结果删历史或改结构。

接口只新增一个可替换的Public构建器：训练入口接收允许的query/历史ID与训练生物标签；推理入口不接受truth/error字段。输出μ、历史分散度、逐history权重、有效支持、缺失状态及bank/model/preprocess版本。四臂共用该输出合同，原风险特征构造器继续使用这份输出。

## 3. 评价和采用规则

先在Source与MC生物DEV各自原fold0做B2/B3、三个种子。早停＋全训练重拟合合计最多24次小网络fit，全部计入4GPU-hours原型限额。这是实现、数值与耗时筛查，单fold正结果不决定采用，也不因单fold负结果立即否定结构。

完整第一阶段是两域各五个生物outer folds；同一BIO任务不因两个模型而重训。按原划分评价三种子平均及单次结果。只有后续确需训练Source风险层，才增加风险训练区内部Public OOF；不能把单层生物OOF直接拿来当完整嵌套风险训练特征。

固定主读出R_hist，预登记辅助读出direct RMSE；同时报告Magnitude和历史支持量规则。数学上，R_hist²是各历史到p的加权平方距离，而训练loss只优化μ；所以μ重建改善并不保证R_hist改善。若仅direct读出改善，记录READOUT_DEPENDENT，不能改名为主读出成功。只有合法Source训练数据具备时，允许一次固定原HGB读出复核这种情况，不扫描多个读出。

结构证据看B3−B2，实用证据看对B0/B1及强简单规则的结果。采用需完整外层macro ΔU20≥0.005、≥60%分层非负、有效分层≥80%、原AURC/error@10/20/50的5%限制及漏检率0.02限制，并报告配对5000次gene统计。允许明显负效应的区间不称稳定改善。

B2若达到相同实用门而B3相对B2没有0.005增量，优先保留B2，并把贡献限定为学习式参照而非集合交互。两者均无收益则继续使用B1，不自动换Transformer。可做一次固定的DEV凸组合重建上界诊断来区分信息/凸包限制；oracle只解释生物重建空间，绝不能进入预测或充当风险排序上界。

## 4. CPU跨研究线须修正后推进

现有40-fit设想不能仅凭缓存存在就运行。已核E201_K562训练view保留HepG2/Jurkat treated；E201_hepg2保留K562/RPE1/Jurkat treated。风险训练行留出另一研究，不能消除源预测checkpoint参数中的暴露。

执行顺序固定：

1. 为每个预定query context建立教师checkpoint→训练实验单元→合法source-validation错误→query单元表，同时核E205。SourceMean自身context均值含额外信息，仅作描述性对照，不能充当公平competence。
2. 优先使用训练时已排除该query context的既有checkpoint及其source-validation预测。源错误必须held-out，生物历史必须合法、任务/基因轴匹配。当前只发现validation日志，尚未核实完整向量出口。
3. 缺预测时，最多一次既有checkpoint的validation导出：先完成输入/输出合同和单批耗时核验，注册最多2GPU-hours；这是复用旧模型推理，0新上游训练，不读取新的评价真值。若public历史也缺失，必须给出具体可用记录/构建成本；不能把P-only训练偷偷称作P+Public训练。
4. 如教师需按query context隔离，风险模型也须按该范围训练；不能重新混入暴露query的教师。保持两研究方向、两architecture方向、全部query contexts及五gene folds。完整两种HGB对照上限因范围拆分变为80次小拟合，先核实际运行清单；算力增加很小，方法含义必须明确。
5. 只有合同通过才运行数值比较，名称限定为“目标研究错误监督留出、既有合法生物知识条件下的迁移”。不称整个pipeline未见目标研究或新独立家族确认。无法取得合法记录时该数值分支标为INPUT_NOT_READY；Public线及已合法反馈工作继续，不伪造完成、不反复重跑缓存。

这一分支是新的输入资格修正，不能悄悄改写旧Source/McFaline/Orion结果。

## 5. Source残差与目标反馈的明确边界

若合法Source直接HGB已稳定满足所声明增量，不再新增Source结构。若存在负迁移，且合法数据已闭合，则只做一次全预算、等信息的残差比较。

为避免附件的一维Ridge出现负斜率并再次倒置Public规则，基础校准固定为带截距、系数非负的Ridge(alpha10)。校准与Source residual使用统一的训练错误排名目标，所有误差用途入账。独立CDF参照用每个outer-train按固定gene哈希保留的20%簇；剩余80%训练两个比较器。CDF参照与所有内部query簇不交叉，任何分组不足两个有限不同错误值则该组为NA，不能借query补齐；记录分辨率。

在这同一CDF下，对剩余80%做四折gene OOF基础校准，得到b_OOF；残差标签严格为z−b_OOF。同特征（原P/Public＋b）、同拟合行和权重比较直接HGB与残差HGB。残差预测允许为负，最后只对b+δ裁剪；不能复用默认clip=True而把负修正截掉。另保留原100%拟合直接HGB强对照，明确它的训练角色分配不同。此项是独立新DEV实验，不能继承旧确认。

新compatibility gate本轮默认跳过；当前没有已验证的独立gain/regret训练条件。Source residual有无收益都不强制开发gate。

目标反馈保留Shared/Target-only/Public+Target/Shared+Target及PertEMA。输入/模型版本相同才复用原结果；变更组合用相同反馈簇、0/10/25/50/75/100%预算重跑。模型/参照先冻结再计算已见评价结果，均标SEEN补充评价，不据此再选结构。100%指反馈池，CDF/校准/早停用过的错误按并集计预算。Error Adapter版本工程降为P2。

## 6. 验收、预算和交付

必要测试：任务/基因fold隔离；query truth不进入推理/PCA；生物训练不接upstream错误；history排序与padding不改变输出；K0安全返回缺失；K1四臂参照相同；同向量集合无虚假选择收益；B2集合上下文恒零；权重非负/和为1；完整向量读出恒等式；模型reload一致；负残差保留；CDF与早停标签预算可复算；原文件哈希不变。

所有新任务有配置、输入hash、真实PID、完成状态和费用记录。原型4GPU-hours；完整验证先按实测拟合数估时，另设12GPU-hours上限并受累计96GPU-hours剩余额约束。成本表空值不是零，先按作业回执去重对账。双卡占用时间分别相加。原型与合法旧checkpoint导出顺序排队，避免超卖显存/预算。不得默增新上游训练。

时间默认放松为新阶段48–72小时目标：先在24小时内给四臂原型/Source输入资格结论，再推进值得保留的完整验证。旧时钟与新增耗时分别记录；新候选没有未参与设计的资产时，独立确认仍是单列缺口。论文正文继续暂缓。

保留用户附件全部交付物；优先交Public四臂结果、Source范围与结果、同预算反馈、采用/拒绝决定、三张主要图和复现入口。上面的INPUT_NOT_READY、READOUT_DEPENDENT等是具体分支结果，不代替负责人继续执行其余合法任务。科研成功仍要求相对最强合法简单方法的增量，工程验收或漂亮网络图不代表投稿证据闭合。
