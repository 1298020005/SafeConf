# 当前唯一必要动作：补Model Error Adapter实际生命周期

依据HEAD `292ec6b096ca76ff045fdcbc059c4a561f615aeb`、当前完成矩阵/真实回执和已授权的持续更新要求，只读判断如下。没有拟合、下载、新表达/真值读取或修改共享记录。

## 两条不应重跑的分支

**E190/fixed-w gate已经完成。** `Stage2_mature_upstream_20260928/gate_mechanism_value/REPORT.md`与 `tools/scripts/analyze_gate_mechanism_value.py`实际覆盖w=0/.25/.5/.75/1及learned，复用相同inner-OOF校准分支和outer folds。learned−最佳fixed：GAT +0.001475 CI[-0.015419,0.021079]，Exphormer −0.003530[-0.024524,0.021109]，E190 −0.039015[-0.250848,0.102147]，没有稳定机制增量。最佳fixed由同一held-out OOF结果选择，其CI条件于该选择，属于DEV描述性强包络，不是可部署TRAIN选择证据；全部固定权重已保存，这项限制不构成重跑/挑w理由。E190仅512gene、692batch×gene任务/47gene，不能充当当前3285新确认。

**Source范围/强基线已经充分完成一次诊断。** SourceMean未补低幅度范围，固定scale压力测试明确样本内/排序也随干预改变；输出变小没有使Public HGB完全失去区分力。raw-affine目标对照两方向U20均降，已停止。新生物范数标签对照支持错误监督相对该代理的Source增量(+0.052161/+0.056636、名义CI均正)，但旧HGB−Magnitude仍CI跨零，MC/Orion强历史比较未建立主要优势。不能由这些结果把幅度认定为外部根因、增加scale/gate/label变体或挑旧TEST赢家。

## 真正未闭合的一项具名要求

**Error bank有记录，Model Error Adapter没有真实持久化更新→发布/保留→reload/rollback闭环。**

- `run_mcfaline_error_memory_simulation.py`确实调用ErrorMemoryRegistry.append并做多个反馈预算的residual fits、release-decision CSV；但代码没有adapter保存、ServingModelRegistry/current指针、publish或rollback调用。它是实际数值的静态retrospective曲线，不能替代模型生命周期日志。
- Source `continual_runtime_replay/actual_v2`只更新Shared risk参数；随后 `public_biology_refit_v2`已经真实重训并持久化PublicBiology，但仍未训练/管理residual λ。因此SYS-A原“未Public retrain”是陈旧描述，不是重训理由；λ生命周期才是剩余操作缺口。
- 已有 `ErrorResidualAdapter(kind='ridge',kappa=50,seed=20260929)`、`VersionedErrorRuntime`与`ServingModelRegistry`可直接复用，新增结构或第三上游均非必要。

## 一个最小Source DEV执行范围，等Root定具体版本后实施

在新的隔离runtime中，只固定**一个已登记upstream release版本**，例如TxPert_GAT/e201_official_frozen；只读原 `continual_runtime_replay_20261002_v2` 的 `MODEL_v1.joblib`、其初始CDF、`FEATURES_PUBLIC_v1.parquet`和Error r1/r2 revisions。Public bank、θ及Source CDF永远固定，不新增Public/Shared拟合。复用原1699tasks/563gene cohort和gene roles：2/3初始feedback，4追加feedback，0旧anchor，1新gate；只取该版本对应行，不筛效果。

新增**最多2个Ridge adapter fits**：λ-v1用该版本2/3反馈，λ-v2用相同反馈加4；原κ50、14项P+PUBLIC+Shared特征及原缩减公式不变。必须从同一个冻结θ和CDF重建每条残差，**不能把现有Error记录中的composite-OOF shared_risk误当该冻结θ的输出**。actual realised_error/provenance可复用，派生残差另行绑定。

实际Error append触发λ更新；保存两个λ及θ/CDF/Public版本bindings；所有anchor/gate预测封存后按原有限值守卫及ΔU20≥−.005、旧anchor AURC≤5%、miss≤+.02、≥60%context strata的DEV门判断。通过才更新隔离serving pointer，否则保留v1而新增记录保留；真实reload/参数hash/预测复算及行政rollback drill独立记录。没有效果胜者选择、永久McF holdout/Orion访问或自动正式发布。一个版本四Sourcecontexts即可操作；不把它扩写成独立研究效果。

预计2个小Ridge拟合为秒级，运行/验证可限定数分钟、4CPU/1GiB、0GPU/下载/upstream calls。这改变“系统是否实际接收反馈并更新adapter版本”的完成判断，**不会自行改变论文级跨研究收益判断**。

## 必须先明确的checkpoint语义

当前Source replay原代码237–239和499–500行、Error provenance明确：`e201/e205_official_frozen`是**composite OOF family，不是individual checkpoint**。按现成缓存做上述循环只能证明“精确登记的ensemble-release版本隔离”，不得写成单checkpoint实证。虽有 `E201_SEED_CENTROIDS.npy`/`E205_SEED_CENTROIDS.npy`等已存member资产（本审阅只核文件存在），仅换version名字不能得到其匹配P/Public输入、truth/CDF和checkpoint绑定。

Root应批准一个可审阅的小scope：接受ensemble-release版本这一诚实操作范围，或坚持严格single checkpoint并先完成其已存member的身份/特征接口绑定。后者没有闭合前不拟合，不新增推理/上游尝试来掩盖。原计划的版本更新要求及用户lead委托可以支持前者的隔离操作；它不授权更改正式核心或声称singleton/新确认。若要求严格singleton，本次资产检查只提供路径线索，不能宣称已经ready。

除此之外，未发现需要再补的一项不变方法实验能建立稳定外部增量。六个官方预测发布族未提供验证完整3285链条的fresh预测；指定六个本地候选又为SEEN、chemical或mouse/UNKNOWN，不存在当前可直接接入的新确认包。科学缺口仍是强简单规则之外的独立研究收益；在现有边界内它不能由重复旧TEST诊断、运维闭环或登记措辞补成。
