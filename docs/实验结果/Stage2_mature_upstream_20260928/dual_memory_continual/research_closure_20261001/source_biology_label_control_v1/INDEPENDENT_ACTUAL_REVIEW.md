# Source生物响应范数标签对照：独立审阅

## 运行前代码审阅

**PASS：最终代码没有发现阻碍此固定Source DEV信息对照的问题。** 执行授权来自Root当前限定任务和用户向lead的研究判断委托；本审阅不生成授权，也不复用已花费的20-fit目标函数许可。

核对runner `tools/scripts/run_safeconf_source_biology_label_control_v1.py` SHA `5c10dbd21bcc4ee2965920ed5a0621a958be02b9b7dec08347f64c8d56a54b68` 及CONFIG的scope/script/input一致绑定：

- 同Common2840轴、1808个Source任务/575gene、两个既有架构的五个gene外折；cached Learned13与任务ID/gene/context/fold匹配，train/query genes不相交。
- 从固定 `SOURCE_TRUE_EFFECTS` 按允许TRAIN task行选择后才计算RMS；原rank CDF helper仅接收这些train norm标签。query真实响应范数不进入特征或CDF，误差列没有进入新臂fit。
- 10个新BioNorm HGB与10个固定旧rank模型使用相同13特征、gene weights、参数和seed；两臂最终均 `clip=True`，保留原[0,1]政策。旧10模型的精确SHA和原Source/core输入链预检，原clipped score门限仍为1e-14。
- 全模型/score-only CSV先记录SHA封存，随后才解析Source缓存actual errors/归档Source行；共享CSV在数字转换前筛掉外部行。没有MC/Orion/E170读取或评分路径。
- 575 sorted genes/5000×575 count文件使用固定SHA，生成seed20261002；全gene blocks共同用于两个方向、四context和两臂，无新RNG。helper仅提取纯counter函数；macro逐指标严格四context均值，NaN传播且保留valid/total，不zero-fill。
- 一CPU线程、Parquet禁用线程、CPU soft590/hard600及wall600秒；10-fit上限，不能覆盖已运行/失败/封存目录，没有发布正式模型或新上游尝试路径。

本段只读代码/配置；没有读取Source向量/parquet数字、执行fit、prediction、统计或额外selfcheck。实际执行及产物审阅待结果出现后追加。

## 科学解释与预算限制

这是**生物响应幅度监督对照**，不是必然失败的negative null，也不是无监督/没有旧模型经验的模型。新臂使用0个actual Source predictor-error训练标签，但使用非零的Source生物真值标签，同时保留旧预测/Public特征与历史Source调用。旧臂仍由错误监督训练。累计10fit的训练行使用数约14464，不是新增14464独立真值；唯一Source生物范围仍是1808任务/575gene，两方向共享真值，两个架构不是两个独立生物研究。

若旧错误监督胜出，只支持其相对这个RMS响应代理的额外预测价值；目标一致性、标签可学习性及优化也影响差值，不能唯一归因于predictor-specific失效机制。若差值小或不一致，也不能证明等价、所有错误经验冗余或风险问题不成立。全部方向/context/指标与既有Magnitude/历史距离点都应保留；不能选择胜者、外部旧TEST重评分或升级新确认/Core/Gate。已有相关性诊断不能替代本训练对照。

## 实际产物与报告恢复审阅

**PASS：统计完整、训练隔离成立，报告故障已零拟合恢复；没有阻塞。** 原执行完成10新拟合、复用10rank模型并封存全部分数，随后仅因 `pandas.to_markdown` 缺可选tabulate依赖退出1。原FAILED状态及5c10执行代码保持原bytes。耗时16.752635秒、CPU16.740402秒、峰值566534144 bytes。独立render-only恢复退出0，37个保护binding及SUMMARY SHA前后/当前全一致，新增fit/score/statistic/RNG均0。不能把原FAILED抹成成功，也不能把render故障解释为科学失败。

实际独立核验：

- 34输入/核心/旧模型/code bindings前后及当前相同，20保存模型和score-only SHA均精确匹配封存receipt；文件顺序/时间和代码均确认封存先于actual-error解析。
- 10个允许训练fold的Source响应RMS、训练CDF标签SHA逐字节复算一致；40个context组的训练row/gene/record身份哈希匹配，训练与query基因/行互斥。10对preprocessor参数逐字节相同。独立确认14464是重复fit-row实例，独立Source仍1808tasks/575genes，没有额外独立Source error标签。
- 每方向/每方法均完整1808tasks/575genes/四context，14464条保存分数全在[0,1]。十折原rank复现最大差1.11e-16，低于原1e-14；归档误差float32身份恢复而评价沿用decimal的修复链保留。
- 32context、8macro、160fold-context的全部七指标从保存记录复算差0；counterpoint最大差1.22e-15。完整140配对和280边际区间从已有5000draw数组复算点/CI差0，全部5000有效；四context等权macro的每draw计算精确一致，575gene身份/既有counts绑定通过。正反差是同一对比的两个符号，不是两项独立证据，没有新RNG或旧main CI复现主张。

### 固定主对比：原错误监督−生物范数监督

| Source方向 | OldRank U20 | BioNorm U20 | 差值 | 名义95% paired-gene CI |
|---|---:|---:|---:|---|
| Exphormer→GAT | 0.783134 | 0.730973 | +0.052161 | [+0.019170,+0.086591] |
| GAT→Exphormer | 0.783743 | 0.727108 | +0.056636 | [+0.025689,+0.094668] |

**科学决定：支持一条限定的Source DEV信息增量。** 在相同P/Public特征、learner和预算下，错误监督相对这一RMS生物响应标签代理取得正U20区间；该结果补充了仅做相关性或label-shuffle无法回答的监督目标对照。它不证明超越所有生物难度标签、不唯一识别模型失效机制、不添加独立家族/研究，也不改变原Source HGB−Magnitude区间跨零及MC/Orion外部未建立稳定增量的结论。当前overall readiness仍未成立，不发布替代模型、不外部重评分、不继续标签变体。

实际绑定：原FAILED STATUS `c1afb2e289192129ac234ba5d89130b398e6fdf569259f2f5eea956f84007698`；SCORE_FREEZE `9ed57c2430b35c40365a7bb10a25090e355b420e8525b7f2893b903853999f41`；saved metric draws `4089da5ec760054a4c728fc4205c446e3d4d62fcbb1518f543ecdbbcc8b5a9c3`；140paired CSV `b93c7152214b79c10363cb8843de2163fcb4f9eef57bfa8a3a6fa6f46092fc72`；RENDER_ONLY_RECOVERY `7584a1951a5ff69b85f9cfcc9d1066542b8977fbed9fecde4fcd3a7f95398fa4`；SUMMARY `29a1a3eebc2ce757a9e98080580a9cb002a20b2d1101f93f3dcd3abad3923b29`。本独立审阅新增fit0、预测0、RNG0、外部数值访问0；仅只读核验已授权Source训练effect/cache和保存结果，并只修改本审阅文件。
