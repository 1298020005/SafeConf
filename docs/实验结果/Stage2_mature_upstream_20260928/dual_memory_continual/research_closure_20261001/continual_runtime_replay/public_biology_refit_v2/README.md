# 实际 Source DEV PublicBiology 重训闭环

**完成；C被原DEV质量门拒绝，实际服务保留A。** 此包补齐原结构的PublicBiology训练／持久化／更新候选／发布或保留操作，不声明外部增益或新确认。

## 三个预定臂

| 臂 | 检索bank | PublicBiology监督 | Risk Error预算 | 用途 |
|---|---|---|---|---|
| A initial learned | immutable v1／1047 records | folds2,3／225 genes | folds2,3／1372 records | 初始隔离服务版本 |
| B updated bank/errors, old Bio | immutable v2／2008 records | 保留v1模型／225 genes | folds2,3,4／2038 records | 固定诊断，不能发布或选赢家 |
| C updated bank/errors, refit Bio | immutable v2／2008 records | folds2,3,4／336 genes | folds2,3,4／2038 records | 唯一发布候选 |

固定原1699 tasks／563 genes／3398 predictor-task records，同gene全部模型／context同fold；没有重新筛队列。初始PubBio risk-training特征使用2fold OOF，更新C使用3fold OOF。B的fold2／3使用旧相应OOF模型在新bank上推理，fold4使用旧final Bio；全部anchor0／gate1先验由对应final Bio生成。原9 pair features、NumericPreprocessor、Bio seed20260929、Risk seed20260930、HGB200/.05/depth3/leaf20/L2=10、50:50 learned/support和per-budget Error CDF保持。

Public模型只拟合生物`transfer_rmse`，不接上游错误。初始final实际964pairs／686tasks／225genes，更新final2737pairs／1019tasks／336genes；这些是已开放Source生物真值的训练用途，不是新独立风险标签。B／C的bank、完整记录顺序、Error预算、CDF排序数组及物理task权重相同（见`B_C_UPDATED_DATA_IDENTITY_PROOF.json`）。C−B反映整个Bio重训操作，包括新增pair监督／训练行数量；不是相同BIO监督下单纯优化器效应。HGB的原L2／叶约束随训练记录规模的影响也包含在这个操作中。

## 真实运行和发布决定

v2 PID886933／session68399，55.846118秒、CPU64.176626秒，峰值758554624bytes。实际7个Bio＋3个Risk拟合；从原immutable Error revision复用`realised_error`，没有增加2次Error OOF拟合。旧Error记录的shared_risk／residual属于历史Manual重放，仅保留来源，不用于本次学习。

所有3臂模型／先验／权重／完整分数先封存，随后才按task predicate数值读取anchor／new-gate的Source错误。训练query生物真值仅取2／3(/4)；0／1 inference pairs不计算transfer target。final Bio和OOF Bio均持久化于joint bundle；reload从原预测、控制、bank和保存Bio重新生成先验／权重／features／risk，全部逐位一致，服务resolver也真实核对。

原finite_release_gate的C−A结果：new-gate ΔU20=`−0.014099`（阈值≥−.005）；miss增量`+0.025460`（阈值≤.02），均失败。old-anchor AURC变化`−1.2285%`，nonnegative strata=`5/8`。因此C标记REJECTED，未publish；实际服务为A＋Bio训练bank1＋检索bank1。新bank2和C参数继续保留，原永久Public/Error CURRENT及正式Source模型未改。没有gate后再拟合／新候选。

## 保存draw的固定统计

复用既有227gene的5000共同gene-count draws，0新RNG；两个角色各8 architecture×context strata，3臂全保留。共48strata、6macro、42paired rows；NA按原宏规则传播，有限draw计数单列。本轮42项均有5000 valid draws。

| 角色／U20差值 | 点差 | 名义95% paired CI |
|---|---:|---|
| anchor C−A | +0.058258 | [-0.009671, +0.123516] |
| anchor B−A | +0.035041 | [-0.016471, +0.122706] |
| anchor C−B | +0.023217 | [-0.028035, +0.043883] |
| new-gate C−A | −0.014099 | [-0.046894, +0.070436] |
| new-gate B−A | −0.006306 | [-0.047176, +0.059710] |
| new-gate C−B | −0.007793 | [-0.031081, +0.038584] |

所有U20差值区间跨零，未建立Bio重训的稳定性能收益；原点判定质量门独立于CI。更新能力与收益分开：本次证明真实PubBio重训、版本绑定、重载及拒绝发布能力。不能由Source DEV结果判定跨域失败根因、绝对幅度方案已修复或问题本身不成立。

## Root独立复核及生物重建诊断

Root独立实际复核已通过：106项input/output/code保护，B／C训练身份和CDF相同，48个risk strata与42个保存draw区间可复算，CURRENT真实为A。代码作者没有将自己的实施验证冒称独立审阅。

Root在封存后额外使用已有先验／Source生物真值和同5000gene计数，0fit／RNG地核对680个unique biological tasks／227gene blocks；四context macro，每task不按两个upstream重复。C−B的微小生物重建变化为：

| 角色／指标 | C−B | 名义95% gene CI |
|---|---:|---|
| anchor 平均RMSE | −0.000280 | [-0.000538, −0.000068] |
| anchor 平均cosine | +0.003755 | [+0.000552, +0.007641] |
| new-gate 平均RMSE | −0.000355 | [-0.000614, −0.000122] |
| new-gate 平均cosine | +0.004994 | [+0.002112, +0.008204] |

这提供了该固定DEV更新操作具有小幅生物重建收益的证据，不能将PublicLearner整体称为失效；但它未转化为已建立的风险排序增量，C仍必须拒绝发布。该postseal补充描述不用于选择版本／新终点或撤销gate，也不是新的外部确认。生物重建使用观测Source centroid，未分离潜在生物信号与完整测量噪声。

实际risk与生物重建结果共同约束后续判断：尚无证据证明问题本身不成立，也没有证明外部失败由绝对幅度特征单独造成。原更新能力缺项已经补齐，当前不再追加refit／参数／种子变体；独立新确认资产与核心修订仍缺合格合同，不能用旧Orion／MC确认新修复。

## 保留失败和信息边界

v1 PID883842／session66815，真实1.337314秒、0fits：split元数据缺native控制行字段而fail closed。实际执行代码`4aee...`在`../public_biology_refit_v1/EXECUTED_CODE_SNAPSHOT.py`完整保留，原runner／registration／ABORT不覆。技术v2以task_id一对一关联原primary metadata控制行；核完整1808 ID／gene／context／canonical数组行序，并恢复原math.log1p路径，未更改公式／容差／cohort／gate。完整原inputs、gene轴、CURRENT和code执行前后SHA保护通过。

这是回顾性Source DEV操作：两个TxPert架构属同一家族；初始Bio／Error训练已包含两个Source研究，不能写为研究按时间首次到达。0Orion／McFaline访问，0新上游调用、下载或GPU；正式13风险主方法和已见外部结果不变。Code作者不将本包称为独立审阅；Root另行复核。大模型／矩阵仅在runtime `continual_public_biology_refit_20261002_v2`。
