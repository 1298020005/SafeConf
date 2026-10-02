# 当前主张—证据矩阵

2026-10-01。论文正文暂缓。本表区分原合同结果与共同 2840 基因的已见数据复核，不自动将复核方法视为经过原确认。

| 问题或主张 | 当前实际证据 | 判定 |
|---|---|---|
| 公共信息能预测观测 centroid 误差 | McFaline 历史距离 U20 0.729672，Prediction HGB -0.008533；固定主评价与完整任务集合 | 成立于当前观测误差定义；必须解释数量/采样精度 |
| 历史效应内容超过历史数量 | 历史数量规则 0.827129，学习历史距离 0.729672；后者差值 -0.097457，95% CI [-0.162306,-0.031880] | 外部主张未成立 |
| 公共参照与目标真值的平均方式是否影响风险 | 固定 Manual guide 等权 0.711295，cell 等权 0.834314；Δ+0.123019，CI [0.056150,0.198610]；对数量规则 Δ+0.007185，CI [-0.039087,0.055330] | 估计目标对齐有作用；仍不能证明内容超过数量，也不替代原确认 |
| 对齐后的真实历史内容相对同支持量置换是否有作用 | 五个 cell 等权 physical-content null U20 0.602882–0.678152；真实 0.834314，全部五个差值 CI 为正 | 该规则依赖真实内容/对应关系；不等于证明内容超数量的独立增量 |
| 对齐历史内部平均方式能否救回外部 Source 模型 | 精确重放原 Manual Source HGB，旧 0.648814；同模型仅换 cell query 参照为 0.575302；对 cell 距离 Δ−0.259012，CI [-0.371281,-0.127813] | 不能；保留失败，不开发新模型解释本次已见结果 |
| Source 学习超过同参照历史规则 | TxPert 两条方向 HGB 0.783134/0.783743，历史距离 0.593518/0.592336 | 同 family 的 held-gene 任务上成立；独立研究不能照搬 |
| Source 学习跨独立研究优于简单历史规则 | McFaline HGB 0.695253，历史距离 0.729672；差值 -0.034419，95% CI [-0.132144,-0.008644] | 不成立，保留明确负结果 |
| 真 Source 标签对应关系有信息 | 五个簇块打乱、同特征/模型/权重/CDF；逐线路完整报告 | 需要逐线路看区间；不能替代强规则或外部迁移 |
| 更多 Source 行/预测器必然更好 | 五个监督顺序、五预算、单来源/合并/等记录/平均；既有曲线不单调 | 不支持必然改善；新共同轴与原轴分别标注 |
| Public 增长必然更好 | 五个记忆顺序、稀疏覆盖记录、逐任务预测 | 不支持单调改善；系统能更新与性能收益分开 |
| 目标反馈有额外价值 | 212 个同一 holdout 任务、152 簇，严格反馈预算；HGB/PertEMA 原生控制特征适配及 Public/Shared 组合 | 若干条件下改善观测误差；收益并非普遍或单调 |
| Shared 降低当前部署调用成本 | 验证预测已存在；合法目标验证风险训练可使用 230 任务/123 簇，新增模型调用为 0 | 当前资产不能宣称 Shared 必然省调用 |
| Source 相对便宜的目标验证学习有优势 | PublicValidation HGB 0.755390，Source HGB 0.695253；差值 +0.060137，CI [-0.017049,0.163774]；Shared 额外特征差值 -0.016076，CI [-0.030772,0.041165] | 未证明额外 Shared 收益；不同信息用途明确记账 |
| 无历史安全回退 | 当前主线外部任务均有同基因其他状态历史；旧无历史组未证实安全排序 | 未证明，不写成核心既成事实 |
| 广义首次后置错误学习 | Risk Advisor、ConfidNet、PertEMA 已有相关方法 | 不允许 |
| 广义首次跨预测器失败监督 | FailureScope、Fail-Fast/Restart-Smart 等跨模型研究已有先例 | 不允许；生物任务的特定信息合同仍可研究 |
| 广义首次利用旧模型评估帮助新模型 | 2024 generic assessor 以旧模型结果加新模型少量参考任务预测未见任务；2025 PredictaBoard 评估逐任务风险预测 | 不允许；零目标风险标签与生物历史参照须分别证明 |
| 噪声和共同失败首次发现 | PertEMA 与 2026 实验可靠性/评价工作已经讨论 | 不允许；需要明确不同的新发现 |
| 性能反映生物模型特异错误 | McFaline 误差与测试细胞数 rho 约 -0.96 至 -0.98；固定细胞数与跨半组诊断使强风险排序消失 | 当前证据不足；不能把观测采样风险等同模型生物失败 |
| 本轮可以宣布达到投稿就绪 | 强规则、信息预算、合同和多数实验已完成，但核心增量/噪声归因与创新范围尚未闭合 | 尚未证明；实验负责人继续工作 |

## 关键边界

- C validation 在上游选择/校准、Public 生物学习、风险训练和 CDF 的用途分开登记。新强对照的 230 条错误只来自与全部 test 基因簇不重叠的验证记录；这是明确增加目标监督的信息条件。
- 验证错误来自未参加上游梯度训练的记录，但 checkpoint/alpha 曾用完整 validation 做选择；不能称为 pristine upstream OOF。
- 固定 20-cell 真值更不精确，属于敏感性诊断。它不替代原 RMSE，也不作为新的赢家选择规则。
- Source 的采样量关联有 context 差异：K562/hepg2 约 -0.02，Jurkat/RPE1 约 -0.38 至 -0.50。不能把 McFaline 的机制泛化到全部结果。
- PertEMA 使用固定官方模型工厂及明确适配特征/目标/校准预算；不是原 CD4 实验的完整复现。

实验、代码、已确认资产与失败记录持续保留。稿件主张需要由以上证据约束，目标保持计算机方向二区/CCF-B，不能用实验数或最高分替代主张核查。


## 反馈池上的无错误标签强规则

在完全相同的 212 个 holdout 任务上，历史数量规则 U20 为 0.795264，学习历史距离为 0.689213，Shared 为 0.571055。Feedback 的增量必须同时相对这些无需错误反馈的规则报告，不能只相对较弱 Shared 得分。全部 5000 次簇抽样、七个指标的配对差在 `common_gene_axis/results/feedback_strong_baselines/` 中，原目标学习器的分数不变。

## Source training-copy and objective-weight diagnosis

Fixed common2840 features/CDF/parameters were replayed in12 table fits; all six original score arrays reproduced within9.72e-17. Copying existing GAT or Exphormer training rows adds no new errors or independent tasks. Learned standard pooled U20 exceeds copied GAT by0.1069 (95%CI0.0113–0.1681), while its0.0650 gap over copied Exphormer remains uncertain (CI−0.0182–0.1341). Matching total loss weight to a single-source budget retains a pooled point advantage, but weight scaling, quantile binning, leaf-count effects and two nearly identical same-family predictors remain attribution limits. These are SEEN diagnostics; no new model is promoted and frozen Orion source parameters remain unchanged. Full fixed results and5000 joint gene draws are in common_gene_axis/results/source_row_weight_diagnostic/.

## Conditional Source scaling uncertainty

Paired5000gene-cluster draws on543fixedMcFaline tasks/380genes now cover all source budgets and diversity comparisons. Learned100%-10% supervision ΔU20=0.339716,95%CI[0.223499,0.390742]; Manual=0.205204,CI[0.112537,0.277519]. Learned fullpool exceeds equal-record pool by0.128708,CI[0.021106,0.179955], and separate-risk average by0.102166,CI[0.018369,0.164578]. All Manual diversity pair intervals crosszero, and adjacent budget changes are not universallypositive. These are conditional fixed-prediction SEEN results, not new families or independent training seeds. Budget-specific CDF resolution and copy/weight controls remain necessary interpretation limits.

## Actual isolated DEV continual-operation evidence

Real Source DEV replay now appends Public records1047→2008 and Error records1372→2038, rebuilds Manual priors, refits the shared risk core and checks actual serving/version pointers. Candidate new-task ΔU20=-0.017995 and3/8strata nonnegative fail the fixed release gate; old risk-v1 remains serving and new records remain stored. Both model reloads, prior reconstruction and administrative rollback are independently verified byte-identical. Seven total tabular fits include the preserved failed technical run; no upstream training or Orion/target-holdout access.

This establishes ingestion/update/quality-rejection/retention and administrative rollback capability, not universal continual benefit. No Public Biology Learner or residual adapter was retrained; Public and Source errors changed jointly, and error-training tasks already span both studies. Two architectures remain one TxPert family. Administrative rollback used the same approved v1 artifact marker; it is not a quality-failing v2 publication. Evidence and the immutable-ledger family-label correction are in `continual_runtime_replay/README.md` and `independent_actual_v2_review/FINAL_REVIEW.json`.

## Available target-validation information and external claim limits

The actual Orion competence check has already computed 2790 target-predictor validation errors across 1661 genes. They were used only to qualify the published upstream, not to train Source risk, target CDFs or adapters. Nevertheless, these errors are available information; obtaining them requires no additional upstream calls beyond the completed qualification workflow. The frozen 13-method Orion comparison does not include a risk learner trained on those target-validation errors.

Consequently, a Source-risk win over fixed historical distance would support that registered contrast; it would not establish label efficiency or call-cost superiority over an available target-validation learner. No target-validation learner is added after TEST opening to promote a new method or replace the fixed primary comparison. Existing McFaline target-validation comparisons remain separate SEEN evidence. The exact usage and missing-comparison limits are recorded in `scientific_readiness_v1/AVAILABLE_VALIDATION_INFORMATION_LIMIT.json`.

The independent readiness audit `scientific_readiness_v1/SCIENTIFIC_READINESS.json` retains both positive and negative fixed comparisons. Public-plus-Source beating a weaker Source prediction-only learner does not by itself establish a biological-content increment over support or a Source-supervision increment over the same-prior distance rule. New Orion efficacy remains unproven until its original truth reader and fixed comparisons complete.

## Orion target-validation control: strongest simple rules completed

The later scoped SEEN validation-reuse control is now compared with all five frozen historical rules, not only weighted distance. All150macroU20 intervals contain zero. Full-budgetPublicRidge−LearnedDirectRMSE=+0.012009,CI[-0.188223,0.179268]; no superiority or Shared/target label-efficiency claim is established. This authorized postconfirmation supplement does not change the earlier13primary comparisons. The pooled target-study control across two checkpoint versions is not a single-checkpoint residual adapter. Exact existing5000indices were reused; same seeds across different gene orderings would not be paired. Evidence: `orion_preparation/validation_strong_reference_completion_v1/`.

## Current2840 Public target-coverage growth

Fixed275HGBfits and5000paired gene draws complete the previously old-contract-only growth evidence. On all543McFaline tasks, mean100−10% U20=+0.680116[0.522410,0.786253]; Source directions intervals crosszero. This is whole-target coverage plus reference/risk refitting, with Source labels100% fixed; no PublicBiology learner update. Same-covered-task direct distances are identical, all25snapshot common-history cohort empty, and low-n cells remainundefined. Full MC HGB0.648814 still below same-prior distance0.711295. This conditional gain does not establish content beyond support/measurement, universal update benefit or RouteBready. Evidence in `common_gene_axis/results/public_growth_uncertainty_v1/`; independent interpretation in `next_scientific_action_review_v1/REVIEW.md`.

## Fixed Orion historical-effect correspondence negative control

Fivefixedphysical effect derangements complete on232tasks/144genes; sameSource support/weights/query links retained. All15macroU20 actual-minus-null points positive, only4nominal CIs positive (Uniform/ManualDirect seeds20261001/20261003); Weightedall5 andHEKall15 intervals crosszero. MC showedstronger content-null evidence; this is a second-study descriptive direction, not robustcontent beyondcount/measurement or SourceCore superiority. CoarseSupport/technicalbatch grouping doesnot match latentvariance. Same3originalrules reproducebits and prior5000draw metrics; original13primary/hashes preserved. Evidence:`orion_preparation/physical_content_null_actual_v1/`.

## Full TRAIN target-control scalar comparator

OnefixedCP4000positive-direction proxy onall232Orion tasks yieldsmacroU20 .078304CI[-.092418,.305068]. LearnedDirect-minus-control+.188762CI[-.171296,.461058],SourceLearnedHGB+.126532CI[-.216172,.416231],fullPublicTargetRidge+.200771CI[-.126186,.424130]. All43macrocomparisons lackpositiveCIlowerbounds. The full38606 controlinputisadditionalbiologybeyondP13; extraCerrortraining0doesnot eraseexisting30targetfitbudgets. No exclusion/signselection/Sourceparameterchange, no newconfirmation. Exactcontrol/modelbaseline checksandall903/924intervals independentlyverified.

### 真实PublicBiology重训更新（Source DEV）

7个PublicBio和3个Risk拟合、存储OOF/final参数、jointBioRisk-bank重建位级一致、原开发发布门拒绝C并保留A已完成。C−B的生物重建小幅收益有DEV名义区间支持；全部6风险U20差值CI跨零。680unique biological tasks按227gene抽样，不以两模型重复行计独立样本。该操作不是新外部确认，不证明单checkpoint ErrorAdapter或普遍增量改善。

## Source强幅度基线与E170信息边界补齐（2026-10-02）

Source两方向LearnedHGB−Magnitude +0.034327[-0.009285,0.069899]、+0.037491[-0.002974,0.077648]；Manual和PredictionHGB另四CI也跨零。HGB超过历史距离/真实标签优于打乱的既有证据保留，不能据此称全面胜过最强零错误标签规则。复用实际5000gene counts，0fit/新RNG，原40点复现。

旧E170用了1920目标validation errors，known640gene与validation同gene/不同donor，不是当前SharedCore的无C错误跨家族确认。known V2−Direct +0.029478[-0.018512,0.090661]、−DistancePlusDispersion +0.024281[-0.028890,0.084104]；missing−Magnitude −0.030054[-0.180912,0.183269]。旧GateA点门通过保留，强规则/无历史稳定增量未证明。known数量panel×state常量，不能当成有排序信息的强数量对照。

停止这两组winner搜索，完整保留原资产、23macro/276strata/161pairs与独立审阅。本轮是证据/预算修正，不是方法修复或投稿就绪。

## Source训练目标诊断已执行并停止

固定20fit（v1有效rank1复用＋v2新19）已完成，旧模型/原外部评价不改。raw-affine U20两方向0.774848/0.780253，相对原rank0.783134/0.783743下降0.008287/0.003490；Spearman也均下降，AURC/error@10小幅改善如实保留。40fold/context仅20非负。初轮无CI，不宣称显著退化/等价或外部根因。独立复算全指标/40训练组/20params/SHA通过。原CDF规则保留，停止此目标变体，不转向调尺度/损失。数学可能性未变成当前收益。用户具体委托和Source-only范围记录，0新上游/原始外部truth/GPU。

## Source错误标签相对生物强弱监督的实际对照（292ec6b）

同13特征、HGB、训练行/权重、gene外折、训练区CDF与原裁剪，旧错误标签相对Source真效应RMS标签的U20增量分别+0.052161[0.019170,0.086591]、+0.056636[0.025689,0.094668]。独立实际核验通过；范围1808任务/575簇、2架构1family。保留源预测输入和公共/生物真值，不能称无监督、所有生物难度代理均排除、独立家族/研究确认或全面优于Magnitude。Source HGB−Magnitude的CI仍跨零；外部主增量和总体投稿就绪不升级。次指标error@10的BioNorm优势保留。见`source_biology_label_control_v1/INDEPENDENT_ACTUAL_REVIEW.md`。
