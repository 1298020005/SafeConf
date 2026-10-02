# SafeConf: Public Experimental References for Task-Level Reliability of Perturbation Predictions

Author names and affiliations: [AUTHOR NAMES], [DEPARTMENT / INSTITUTION], [CITY / COUNTRY]. Corresponding author: [NAME], [EMAIL].

## Abstract

Single-cell perturbation predictors produce thousands of expression profiles, whereas experimental follow-up is limited. A useful reliability system should rank likely prediction errors even when a newly deployed predictor has not accumulated its own error labels. We introduce SafeConf, a modular framework that separates three information sources: real biological experiments used to construct a public reference, errors of existing predictors used to learn a shared risk function, and subsequent feedback from the target predictor used for personalization. Public reference learning uses biological response supervision rather than predictor errors; the risk layer combines prediction statistics, reference discrepancies, and evidence support. A weighted historical-distance rule provides an interpretable, label-free comparator. Experiments on a common 2,840-gene axis include 1,808 biological tasks for two TxPert architectures and 543 tasks for an external McFaline predictor. Shared risk reaches Utility@20 of 0.7831 and 0.7837 for the two cross-architecture directions, compared with 0.5935 and 0.5923 for the corresponding historical-distance rule. Gains over the stronger magnitude baseline remain uncertain. In McFaline, public distance is informative, but source-supervised risk does not improve on it. A separately frozen Orion evaluation further exposes limited public-history coverage and uncertain cross-study transfer. Target feedback improves risk ranking when combined with public evidence, without demonstrating consistent additional benefit from the shared score. A completed five-fold reference comparison finds gains from pointwise neural weighting on Source, but no consistent increment from learned set context; in McFaline, same-context support aggregation matches or exceeds the neural point estimates. These results support reference-based reliability with separately assessed source transfer and target adaptation.

Keywords: single-cell perturbation prediction; reliability estimation; public experimental evidence; error supervision; selective review; target adaptation.

## 1. Introduction

Perturbation prediction connects an intervention, a cellular context, and a predicted transcriptional response. Such predictions are increasingly used to prioritize experiments. Average predictive accuracy, however, does not identify the individual tasks that require additional review. A predictor may perform acceptably on a benchmark while failing on specific interventions or unfamiliar contexts. Conversely, a moderate predictor may still provide useful predictions when its errors can be ranked. This distinction makes task-level reliability a practical problem separate from building another perturbation predictor.

A common solution is to learn a post-hoc error estimator from past predictions and observed outcomes. This requires suitable errors from the predictor being deployed. At first deployment, a new predictor may have produced current predictions but have few or no locally available risk-training labels. Meanwhile, other information can already exist: public perturbation experiments in related conditions and past errors of other predictors. These resources have different meanings. Biological experiments describe what happened in the underlying system; predictor errors describe how a computational model departed from that system. Combining them without separating their roles can obscure whether a method benefits from biological content, additional supervision, or simply more precise measurements.

SafeConf organizes these sources into a two-stage estimator with an optional target-feedback extension. A public module constructs a biological reference from eligible historical experiments. A risk model then learns how a prediction, the reference, and their disagreement relate to realized error. The public module is trained on biological reconstruction, whereas the risk model is trained on errors of source predictors. At target inference, source errors are represented by the fitted risk function; the current implementation does not retrieve individual source errors for each query. As target outcomes arrive, a target-specific learner can use public features and, optionally, the shared risk score.

This design yields three falsifiable questions. First, does public evidence improve ranking beyond features of the current prediction? Second, do source errors improve the use of the same public evidence beyond a direct historical-distance rule? Third, after target feedback becomes available, does the shared score still add information beyond public features and target errors? These are information comparisons. They differ from comparing two learners with identical features and labels.

We study these questions using cross-architecture tasks, an external predictor, a separately frozen cross-study evaluation, and fixed-budget feedback experiments. The resulting picture is heterogeneous. Source supervision adds useful information within the current model family, while the cross-study results favor a simpler public reference or remain inconclusive. Public support is itself a strong signal in one study, requiring biological content to be tested against support-matched controls. We therefore develop SafeConf as a reusable reliability interface and empirically examine the conditions under which its components contribute.

The contributions of this manuscript are: (i) a public-reference formulation for task-level risk estimation with an explicit separation between biological supervision and predictor-error supervision; (ii) an interpretable weighted-distance comparator and controlled tests of the incremental value of source errors; and (iii) a target-feedback and versioned-update protocol that measures personalization at a common error-label budget. The public-reference comparison additionally separates data-contract alignment from representation choice, and tests a pointwise network and a set-conditioned network against fixed context, condition and support references. Neural set interactions are treated as an empirical component choice; the general set-learning operator is established prior work.

## 2. Related Work

### 2.1 Post-hoc reliability and target-specific error learning

Risk Advisor learns a post-hoc meta-model to characterize failures of trained black-box classifiers and supports applications including abstention and shift diagnosis [@lahoti2023risk]. This establishes that learning reliability outside an upstream model is not itself a new contribution. SafeConf addresses vector-valued perturbation effects and separates errors from other predictors from target-specific feedback.

PertEMA is a public software implementation of post-hoc perturbation reliability estimation. It uses out-of-fold errors, a gradient-boosted estimator, isotonic calibration, and split-conformal intervals. Its documentation instructs users to refit for a new screen, and currently specifies a software citation rather than an accompanying paper [@shrestha2026pertema]. We use its fixed estimator factory as an adapted target-feedback comparator. SafeConf's candidate distinction is the use of reusable public biological references and source-predictor error supervision before target risk labels become available, not the invention of target-specific error learning.

### 2.2 Biological evidence and uncertainty

PRESCRIBE models perturbation-response uncertainty through a multivariate evidential framework that combines local training evidence and response variability [@cheng2025prescribe]. It is closely related to the interpretation of similarity and experimental quality, but requires a specialized predictive model rather than a generic post-hoc interface to frozen predictions. Accordingly, conceptual comparison and any compatible native uncertainty experiments are distinguished from equal-information post-hoc baselines.

A recent perturbation-reliability study examines the reproducibility of measured responses, separates shared from perturbation-specific effects, and studies the consequences of quality filtering [@wang2026reliable]. This work makes experimental reliability a substantive competitor to purely model-based explanations. SafeConf preserves support and repeatability as distinct quantities and evaluates biological content against support-only rules. It does not treat quality labels or shared task difficulty as original concepts.

### 2.3 Evaluation and set-based reference learning

PerturBench provides datasets, prediction interfaces, and evaluation tools for perturbation analysis [@wu2024perturbench]. Complementary evaluation work shows that reusing a control population can inflate correlation and cosine-based comparisons of differential responses [@nicol2026spurious]. Our primary outcome is task RMSE on a documented effect contract; direction and correlation remain secondary outcomes whose control construction must be reported.

Deep Sets provides permutation-invariant representations for unordered collections [@zaheer2017deepsets], while Set Transformer adds attention-based interactions [@lee2019settransformer]. We use these works as architecture precedents. The evaluated small public set builder is an application to biological reference selection, not a new general set-learning operator. Its value must be established against the existing per-history tree scorer and simple reference aggregation.

## 3. SafeConf

### 3.1 Problem setting and available information

A biological task q identifies a study, cellular context, perturbation, condition, and control/effect definition. A frozen upstream predictor m produces an aligned effect vector p(m,q) over G genes. The observed effect y(q), when available, defines realized task error:

$$
e(m,q)=\sqrt{\frac{1}{G}\|p(m,q)-y(q)\|_2^2}.
$$

The risk estimator returns a scalar r(m,q), with larger values indicating a higher expected ranking of error. It does not change the upstream prediction. Its output is a ranking score, not automatically a calibrated error probability.

We distinguish three information sets. Public evidence contains real experimental effects h(i), context and perturbation metadata, support, repeatability measurements when available, and provenance. Source supervision contains errors of registered predictors on eligible historical tasks. Target feedback contains outcomes that have returned for the specific deployed model and version. The latter two are separated by predictor identity for storage and budget accounting, even when the final risk features omit identity.

Direct-effect outputs are used directly. Treated-state outputs are converted using a matched predicted or observed control as specified by the output contract. Gene identifiers, ordering, normalization, and missing-gene policy are fixed before error computation. Missing genes are not filled with artificial zero effects. The common-axis experiments use 2,840 genes; the fixed Orion experiment uses a distinct 3,285-gene contract. Raw error values from these contracts are not pooled.

### 3.2 Public biological references

For task q, the eligibility operator returns a set H(q) of historical experiments. It excludes the current task and prohibited experimental relatives; information about independent study, replicate group, batch, and prior exposure is retained. The implemented Source references mainly reuse the same perturbation in other cellular contexts. This supports related-history deployment and does not by itself solve new-perturbation tasks with no suitable history.

Each eligible history has a normalized weight w(i,q). The reference mean and its weighted dispersion are:

$$
\mu_q=\sum_i w_{iq}h_i,\qquad
\sigma_q^2=\sum_i w_{iq}\frac{\|h_i-\mu_q\|_2^2}{G},\qquad
\sum_iw_{iq}=1.
$$

Uniform aggregation, cell-support-weighted aggregation, and a learned per-history scorer are evaluated. The completed learned builder predicts a biological transfer error from features of a query-history pair. These include control-state similarity, historical effect summaries, support fractions, conflict, and available quality measurements. Its supervision is the discrepancy between a historical effect and a held-out biological response, independently of upstream prediction errors. Predicted transfer errors are converted to nonnegative weights and mixed equally with the existing support weights. The original conversion uses the spread of candidate scores; the previously tested out-of-fold residual-temperature variant did not satisfy the adoption criterion and is not substituted into the reported results.

Support and quality are different. Counts describe how much material supports a record; repeat consistency describes how reproducible its effect is. Unknown quality is retained as missing. In particular, a weight concentrated on one historical record can reduce weighted dispersion without resolving disagreement in the original set. The risk interface therefore retains a summary of the original eligible-history conflict separately from the chosen weights.

### 3.3 Matched public-reference builders

The four-arm comparison changes the public weight scorer while preserving eligible records, the biological reconstruction target and the full-effect output contract. B0 uses the support-weighted mean, B1 uses the per-history HGB scorer, B2 uses a pointwise network, and B3 uses a set-conditioned network. The latter encodes each history and its relationship to the query using a small network, then conditions a nonlinear score on the mean of all valid history encodings:

$$
u_i=\phi(x_i,q),\qquad c_q=\frac{1}{|H(q)|}\sum_i u_i,
\qquad a_i=\operatorname{softmax}_i\{g([u_i,c_q])\}.
$$

The reference weights remain w(i,q)=0.5a(i,q)+0.5s(i,q), where s denotes the existing support distribution. A pointwise-network comparator uses the same inputs and training protocol with the learned set context removed. This comparison isolates the incremental use of learned collection-level information without claiming that the pointwise comparator lacks all manually constructed conflict summaries.

The two neural builders use 32 training-fitted principal components of the historical effect for scoring, a two-layer width-64 encoder, and a 128-to-64-to-1 nonlinear score head. Aggregation is permutation invariant and masks padded records. The output mean remains a weighted combination of the full aligned effect vectors. Biological training minimizes reconstruction mean squared error, with equal total query weight per biological cluster. AdamW uses learning rate 0.001, weight decay 0.0001, effective batch size 32, and at most 100 epochs. Internal grouped validation selects the stopping epoch, followed by a fresh fit on the permitted training partition.

The fixed mixture restricts the attainable references. A development-only constrained oracle therefore measures how well the same allowed mixtures could reconstruct each biological response. This is a diagnostic for representational headroom, not a feasible risk predictor or an upper bound on Utility@20. If biological reconstruction improves without better risk ranking, the fixed history rule and the predeclared HGB reader are evaluated separately. All four builders are evaluated on the same outer query folds. The primary fixed reader is the weighted historical rule R_H; direct prediction-reference RMSE is an auxiliary reader. A separate nested risk-reader comparison assesses each builder through supervised risk estimation, so failure of a fixed reference rule does not decide the builder-reader combination.

### 3.4 Source-supervised risk mapping

The risk input comprises six prediction summaries and seven public-reference features. Prediction summaries are magnitude, mean absolute value, signed mean, standard deviation, absolute 95th percentile, and sparsity. Public features are prior magnitude, prediction-prior RMSE, cosine similarity, prior dispersion, log support, effective number of sources, and original history conflict. Training-derived preprocessing and missing indicators follow the frozen implementation; identity and optional model-native uncertainty indicators are not introduced as model identifiers.

The shared risk function is a histogram gradient-boosted regressor with 200 iterations, learning rate 0.05, depth 3, minimum leaf size 20, and L2 regularization 10. Ridge with alpha 10 is a simple supervised comparator. Source errors are converted to training-distribution midrank empirical-CDF labels within their documented study, predictor/version, context, and output-contract groups. Every budget uses only the labels permitted by that budget to construct its CDF. The CDF is a training normalization device; the resulting score is not the unknown target predictor's actual error percentile.

For a new predictor C, the shared function uses C's present predictions and the public reference features. C's realized risk labels are not required for this inference path. Source errors have influenced model parameters, rather than being directly queried at inference. Target validation outcomes used for upstream predictor calibration are reported separately from risk-model training labels.

### 3.5 A strong reference-only risk rule

A simple comparator uses the discrepancy from the reference and its dispersion:

$$
R_H(q)=\sqrt{\frac{\|p_q-\mu_q\|_2^2}{G}+\sigma_q^2}
=\sqrt{\sum_iw_{iq}\frac{\|p_q-h_i\|_2^2}{G}}.
$$

The equality follows from the weighted bias-variance decomposition on the same gene axis, mask, and weights. It does not require source error labels. Comparing a source-supervised learner with this rule asks whether learned error information improves on the historical discrepancy already available. Comparison with an otherwise identical learner trained on permuted source labels further isolates the correspondence between source tasks and their errors. Neither comparison alone proves cross-study transfer.

When H(q) is empty, a historical rule is unavailable; it does not receive an artificial zero effect. History-conditioned comparisons use the common evaluable cohort, and prediction-only methods additionally report whole-cohort coverage. A one-record history has zero empirical dispersion but no demonstrated absence of biological uncertainty.

### 3.6 Target feedback and versioned updating

Once C has legitimate held-out errors, we fit target-specific learners in four information conditions: Shared, Target-only, Public+Target, and Shared+Target. The latter adds the previously available shared score to the public and prediction features. Existing residual adaptation is evaluated as an alternative, not a required final layer. A model need not benefit from both source transfer and residual correction to be useful.

Public memory is append-only and shared across predictors; error memory is scoped by predictor and checkpoint/version. A public update can change the reference even without changing network parameters. Consequently, a released version binds the memory snapshot, eligibility rules, preprocessing, public builder, risk model, and affected target adapters. A candidate is trained using previously available records, checked on registered development and anchor tasks, and published as a compatible version or rejected while new records remain stored. Permanent evaluation outcomes do not select a release. The continual claim requires performance on subsequent tasks, beyond demonstrating that a retraining operation executed.

## 4. Experimental Design

### 4.1 Cohorts and evidence roles

Table 1 lists the cohorts used here. Counts refer to biological tasks and gene clusters, not replicated rows across predictors or seeds. Source and McFaline common-axis results are completed development/seen analyses. Orion's original fixed comparison was frozen before its once-only test evaluation; subsequent Orion diagnostics are seen analyses. Historical E170 confirmation belongs to an earlier method and used target validation errors, so it is not presented as independent confirmation of the current shared model.

| Table 1. Cohort | Tasks / gene clusters | Gene axis | Evidence role |
|---|---:|---:|---|
| TxPert Source, four contexts | 1,808 / 575 | 2,840 | Cross-architecture development/seen |
| McFaline, full risk cohort | 543 / 380 | 2,840 | External predictor, seen evaluation |
| McFaline, feedback evaluation pool | 212 / 152 | 2,840 | Fixed target-feedback comparison |
| Orion, all eligible test tasks | 2,993 / 1,750 | 3,285 | Fixed external prediction cohort |
| Orion, common historical-reference cohort | 232 / 144 | 3,285 | Original frozen primary comparison |
| Frangieh, retrospective cross-family tasks | 567 / 189 | 512 | Seen stress analysis; upstream competence failed |
| McFaline, reference-builder comparison | 542 / 377 | 2,840 | Five-fold development/seen; aligned bank |

For Source public learning, tasks have one to three eligible historical records. The McFaline biological development cohort has seven to fourteen per task from a 7,173-record bank. These records share genes and conditions and are not counted as that many independent experiments. A newly completed Frangieh retrospective branch uses existing scGPT and GEARS predictions on their native 512-gene output axis [@cui2024scgpt] [@roohani2024gears]. Its source errors are original upstream held-out errors, its public history is restricted to original upstream training/validation experiments, and risk-training folds are separated by gene. The old upstream test has become seen risk-development evidence, rather than a new confirmation set.

### 4.2 Splits, teacher exposure, and nesting

Biological-cluster splits keep all predictor outputs for the same true task together. An outer fold assesses risk; inner folds generate public features used for risk training. Public preprocessing, learned reference construction, CDF fitting, and risk fitting use their permitted partitions. In the new PublicSet branch, public builders share an outer gene fold across Source contexts, with query genes excluded from fitting queries, fitting histories and PCA. McFaline excludes every outer evaluation experimental record globally, including its use as another query's history; inner validation follows the same rule. Source inference is a separately declared rotating-context deployment: same-perturbation evidence from other contexts is available, while the current query experiment is excluded.

For the new Source risk follow-up, error supervision is additionally scoped to the current inference context. Each source-architecture-to-target-architecture reader trains only on source errors in that context and outer training genes, and evaluates target errors in that context and outer test genes. Its empirical CDF is fitted only to those source-training errors. It does not mix teacher errors across contexts. The teacher audit checks 32 training/prediction records across E201/E205, four contexts and four seeds: target treated training and target perturbed access must both be zero, the current context must be absent from the teacher's training context list, and the final checkpoint hash must match. A perturbation gene may have been observed in another context; this does not make the current biological query experiment an upstream training example. These checks and the per-fit CDF records bind the interpretation of the new risk results.

An independent experimental unit cannot enter the training history of another query merely because that query has a different identifier. Holding out risk labels does not erase a teacher's prior exposure to the evaluated biological response, so upstream exposure and risk-fold membership are reported separately.

The historical common-axis matrices retain their original versioned development/seen scope. The new record-level history audits and context-scoped Source readers do not retrospectively certify those older mixed-context matrices. The context-scoped reader comparison is reported as a separate version, while the older results provide an empirical baseline. No study-holdout experiment is implied by gene grouping or by a context-specific teacher audit.

### 4.3 Comparators and label budgets

The completed core matrix contains 16 specified methods: magnitude, manual-prior magnitude and dispersion, prediction-only Ridge/HGB, three public aggregators crossed with direct distance/Ridge/HGB, and two weighted historical-distance rules. Source-label permutation, matched-support physical history permutations, and a biological-magnitude label control test alternative explanations. A support-only rule and prediction-plus-support learner are retained as strong comparators.

Target-feedback budgets are 10%, 25%, 50%, 75%, and 100% of the registered feedback pool. The evaluation pool is disjoint. Error labels used for CDF estimation, fitting, or calibration are counted in the budget. Shared uses no newly revealed target risk labels; target-only learners are unfit at zero feedback. McFaline upstream validation calibration is separately recorded, including 542 validation records; thus lack of target risk-training errors is not equated with the entire upstream pipeline having never used a target validation outcome.

The PertEMA-derived comparator uses the official factory at commit `43c09a32e23d0ee2ae5dfbab21b2deeab27f1803`. The adapter changes feature sets, uses the budget-specific task-RMSE rank target, applies biological-cluster weights and registered seeds, and optionally uses grouped out-of-fold isotonic calibration. No separate conformal sample is allocated, so this is an estimator adaptation rather than a reproduction of all original uncertainty guarantees. An additional native-control feature version is reported separately.

### 4.4 Metrics and statistical analysis

For n tasks, let k=ceil(0.2n), S be the k tasks with highest predicted risk, and O the k tasks with highest realized error. Utility@20 is:

$$
U_{20}=\frac{\overline e_S-\overline e}{\overline e_O-\overline e}.
$$

Ties use task identifiers. A stratum with fewer than 20 tasks or denominator at most 1e-12 is not evaluated. Larger Utility@20 means that a fixed review budget captures more of the attainable error enrichment. It is not classification accuracy or a percentage of correctly identified tasks. Spearman correlation, area under the risk-coverage curve, retained error at 10/20/50% coverage, and high-risk miss rate complement the primary endpoint.

We macro-average the registered contexts and report valid versus planned strata. Primary differences use 5,000 paired biological-gene-cluster bootstrap draws, sampling all outputs of a biological unit together. Point differences are computed from the actual estimates; bootstrap means are not substituted for them. Repeated training seeds describe fitting sensitivity, not new independent biological observations. The displayed primary-seed results use the first preregistered seed, 20260930; all three seeds are included in the accompanying evidence tables. Intervals condition on the fitted models and do not include a complete bootstrap refitting of training data.

## 5. Results

### 5.1 Public evidence and source supervision contribute differently across settings

Table 2 presents the completed common-axis matrix. Shared error learning reaches high review utility in both architecture-transfer directions. Relative to the same learned weighted historical rule, point increments are 0.1896 and 0.1914. However, magnitude alone is already strong in Source. The corresponding shared-minus-magnitude differences are 0.0343 (95% paired interval -0.0093 to 0.0699) and 0.0375 (-0.0030 to 0.0776). Thus source supervision is useful relative to historical distance, while an advantage over the strongest simple Source comparator remains unresolved.

| Table 2. Utility@20, common 2,840-gene axis | Exphormer to GAT | GAT to Exphormer | TxPert to McFaline |
|---|---:|---:|---:|
| Prediction magnitude | 0.7488 | 0.7463 | -0.2047 |
| Prediction-only HGB | 0.7590 | 0.7527 | -0.0085 |
| Support-weighted historical distance | 0.5698 | 0.5686 | 0.7113 |
| Learned weighted historical distance | 0.5935 | 0.5923 | 0.7297 |
| Manual reference + source HGB | 0.7805 | 0.7860 | 0.6488 |
| Learned reference + source HGB | 0.7831 | 0.7837 | 0.6953 |

McFaline has a different information regime. Prediction-only HGB is nearly uninformative, while public historical distance attains 0.7297. The source-trained risk learner attains 0.6953, a difference of -0.0344 from learned weighted distance, with a paired interval of -0.1321 to -0.0086. The portable resource is therefore not necessarily the entire source-trained error relationship. A useful public reference may survive a study shift even when the learned risk map does not improve its interpretation.

These differences also motivate separating the public builder from its reader. Source learned direct reference distance attains only 0.2507 and 0.2788, whereas adding historical dispersion yields 0.5935 and 0.5923. Learning a more accurate conditional mean and constructing an effective risk ranking are distinct tasks.

### 5.2 Source errors contain information beyond a biological-magnitude proxy

We hold the features, HGB, folds, weights, and CDF procedure fixed and replace source error labels with the RMS magnitude of the true biological response. Table 3 shows that the original error labels outperform this proxy in both directions. This is a controlled positive result about the content of source supervision, within the current architecture pair.

| Table 3. Source supervision control | Error-label HGB | Biological-magnitude-label HGB | Difference | Paired 95% interval |
|---|---:|---:|---:|---|
| Exphormer to GAT | 0.7831 | 0.7310 | 0.0522 | [0.0192, 0.0866] |
| GAT to Exphormer | 0.7837 | 0.7271 | 0.0566 | [0.0257, 0.0947] |

Both directions share biological outcomes and belong to one model family. The comparison rejects equivalence to this particular magnitude proxy; it does not establish that all task-difficulty proxies are inferior or that the supervision transfers to an independent family. Permuted-label and source-coverage analyses are retained in the supplemental evidence rather than counted as additional independent datasets.

### 5.3 Support is a strong alternative explanation in McFaline

The negative log-history-support rule reaches Utility@20 0.8271 in McFaline, above both learned public distance and source HGB. In Source the same rule reaches only 0.0961 and 0.0899. Consequently, a universal claim that support explains all studies would be as inaccurate as attributing every public-evidence gain to biological content.

Public evidence bundles response content, measurement precision, availability, and experimental support. Matched-support history permutations and the source prediction-plus-support model help separate these contributions. The strong McFaline support result remains in the main comparison, since omitting it would exaggerate the necessity of a learned public representation. The completed reference-builder comparison therefore retains support and fixed matched-context aggregation alongside the learned references (Section 5.9).

### 5.4 A frozen cross-study experiment reveals a coverage and transfer boundary

The original Orion evaluation uses a published fixed linear predictor and 2,993 eligible tasks spanning 1,750 genes. Only 232 tasks, covering 144 genes, have the history required for the common primary comparison: 7.75% of all eligible tasks. The predictor passed the registered noninferiority competence rule while remaining close to the training-mean baseline, rather than demonstrating strong superiority over it.

| Table 4. Orion historical-reference cohort, 3,285-gene contract | Utility@20 |
|---|---:|
| Learned direct reference distance | 0.2671 |
| Learned weighted historical distance | 0.2149 |
| Learned reference + source HGB | 0.2048 |

The preregistered HGB-minus-weighted-distance difference is -0.0101, with a paired 95% interval of -0.2712 to 0.1913. HCT116 retained error at 10% coverage worsens by 7.598%, and its high-risk miss rate rises by 0.0909. The original safety criterion is therefore not met. This fixed result is retained as an external boundary rather than replaced with the best secondary comparison.

Subsequent seen analyses reuse 2,790 existing validation errors to train target-specific estimators. Public+Target Ridge reaches 0.2791 compared with 0.2671 for the fixed direct reference rule; the paired difference interval is -0.1882 to 0.1793. Across 150 registered macro comparisons to strong rules, the intervals include zero. These results show that simply obtaining additional target labels is not sufficient when public support is sparse and transferable ranking information remains limited.

### 5.5 Target feedback benefits from public features, but extra shared scores are not uniformly useful

The strict McFaline feedback evaluation uses 212 tasks and 152 genes. Table 5 compares fixed HGB variants at the same feedback budgets. These scores must not be subtracted from the full 543-task results because the evaluation cohort differs.

| Table 5. Feedback budget | Shared | Target-only HGB | Public+Target HGB | Shared+Target HGB | Existing residual HGB |
|---|---:|---:|---:|---:|---:|
| 10% | 0.5711 | 0.0807 | 0.0807 | 0.0807 | 0.5711 |
| 25% | 0.5711 | 0.0884 | 0.7848 | 0.7848 | 0.7348 |
| 50% | 0.5711 | 0.0984 | 0.7498 | 0.7153 | 0.7408 |
| 75% | 0.5711 | 0.1515 | 0.8167 | 0.7501 | 0.8054 |
| 100% | 0.5711 | 0.0833 | 0.7747 | 0.7742 | 0.7846 |

The small-budget tree models can be constant, for which Spearman is undefined; the deterministic tie rule still yields a numerical U20. Public features create a large improvement once the learner can use them. At full feedback, adding the shared score to Public+Target changes the point estimate from 0.7747 to 0.7742, providing no point improvement. Curves are nonmonotonic, and the strongest no-feedback support rule remains a required comparator. These observations support treating target adaptation as a separate component whose information requirements and adoption conditions are evaluated rather than assumed.

At full feedback, the same-information PertEMA-factory adaptation obtains 0.0233 with prediction-only features, 0.7560 with public features, and 0.7704 with the added shared score. These comparisons reinforce the importance of input information across two tree-learning implementations; they do not establish universal superiority of HGB over the original PertEMA pipeline. The native-control adaptation and isotonic variants are preserved in the supplement with their different feature budgets.

### 5.6 Updating a biological learner is not equivalent to improving risk

A completed Source development replay compares an initial release, new memory with the old biological builder, and new memory with a refitted builder. On old anchor tasks, refitting reduces biological reconstruction RMSE by 0.000280; on new-task checks, the reduction is 0.000355. Both nominal paired intervals exclude zero. The corresponding risk-utility differences do not.

Relative to the initial release, the new-task U20 difference is -0.014099 and the high-risk miss rate increases by 0.025460. The candidate is rejected, and the old serving version is retained. This is a concrete demonstration of versioned update and release behavior, with an informative distinction between biological reconstruction and operational reliability. This completed replay establishes an executed release decision; it does not establish an improvement on future tasks or certify an updated neural builder.

### 5.7 Native-axis cross-family stress analysis

The new Frangieh analysis reuses scGPT and GEARS caches and fits 120 fixed risk estimators without new upstream training or GPU computation. There are 567 distinct biological tasks and 189 genes across three original context-holdout folds. Those folds overlap biologically; synchronized gene bootstrap preserves their dependence. Most perturbation genes had been seen by the upstream model in another context, so the experiment concerns held-out tasks and risk-training genes rather than entirely unseen upstream genes.

| Table 6. Frangieh held-out contexts, native512 | Public HGB U20, 567 tasks | Magnitude U20, 567 tasks | HGB minus weighted history, paired 471-task cohort | Paired 95% interval |
|---|---:|---:|---:|---|
| GEARS to scGPT | 0.7014 | 0.1087 | -0.0582 | [-0.1446, 0.0155] |
| scGPT to GEARS | 0.7401 | 0.3789 | -0.1239 | [-0.2410, -0.0552] |

Public HGB exceeds magnitude in both directions: paired U20 differences are 0.5927 [0.4392, 0.7713] and 0.3612 [0.1246, 0.4973]. Only 471 of the 567 tasks have usable history, so comparisons against historical rules are recomputed on that identical supported cohort rather than obtained by subtracting full-cohort point estimates. Against weighted historical distance, the source-supervised HGB does not show an increment. This extends the observation that public evidence can be valuable while a learned source-error reader remains unnecessary or harmful.

The upstream capability check selects the strongest zero/global-mean/context-mean baseline by training-gene out-of-fold error and evaluates it on the original validation tasks. All six model/fold assets have relative macro RMSE disadvantages of 8.13% to 14.10%, failing the registered competence gate. These results therefore serve as cross-family stress evidence, not qualified main validation. They concern these checkpoint/split assets and do not rank the general capability of the published model families.

A saved-score diagnosis finds that scGPT-to-GEARS Ridge predictions saturate under hard clipping in all fifteen risk-fold evaluations. The completed, separately registered technical repair applies the fixed strictly increasing map 0.5 + arctan(z - 0.5)/pi to raw Ridge output, preserving raw ordering while bounding the score. It performs zero new fits and preserves the original models, predictions, primary results, and input hashes. This is a post-primary seen diagnostic, not percentile calibration or external confirmation.

On scGPT-to-GEARS held-out contexts, repaired prediction-only Ridge reaches U20 0.0673 from -0.0562, a gain of 0.1236 [0.0085, 0.3424]; repaired Public Ridge reaches 0.0999, a gain of 0.1562 [0.0284, 0.3484]. Both remain below magnitude 0.3789: their paired differences are -0.3115 and -0.2789, with both intervals entirely below zero. Public Ridge AURC worsens from 0.0578 to 0.0587, so the recovered top-tail ranking does not uniformly improve selective risk. GEARS-to-scGPT U20 remains 0.1278 for prediction-only Ridge and 0.7069 for Public Ridge. On the identical history-covered cohort, repaired Public Ridge still loses to weighted historical distance in both directions: -0.0608 [-0.1459, -0.0008] and -0.7445 [-0.8961, -0.6306]. The branch stops after this one fixed repair. Table 6 retains the original HGB results and the failed competence qualification.

### 5.8 Data alignment changes the reference before changing the algorithm

A completed McFaline diagnostic isolates one data-contract change: within each public experiment, replace the equal-guide effect mean with a cell-weighted mean. The prediction, canonical query truth, eligibility, and between-history support weights stay fixed. Only registered training/validation expression is aggregated: 7,173 public experiments and 615,059 cells, with zero test cells aggregated. The mean discrepancy between the two historical-effect definitions is RMSE 0.018226. This is a seen post-confirmation analysis of the 543-task, 380-gene cohort, not a new external test.

| Table 7. Fixed support-weighted reference, McFaline common2840 | Utility@20 | Difference from equal-guide bank | Paired 95% interval |
|---|---:|---:|---|
| Original equal-guide historical effects | 0.7113 | Reference | Not applicable |
| Cell-weighted historical effects | 0.8343 | 0.1230 | [0.0561, 0.1986] |
| Cell-weighted reference versus support-only | 0.8343 versus 0.8271 | 0.0072 | [-0.0391, 0.0553] |

The positive comparison in Table 7 belongs to data alignment: it does not establish a gain from a set network or from source error labels. The support-only comparison remains unresolved. A fixed source-HGB replay, with one source refit and no new target error-training labels, reproduces the original score before substituting the aligned public inputs. Its Utility@20 changes from 0.6488 to 0.5753; the paired interval for the difference is [-0.1853, 0.1088]. It loses to the aligned historical rule by -0.2590 [-0.3713, -0.1278] and to support-only by -0.2518 [-0.3640, -0.1130]. Aligning a biological reference can therefore improve a direct rule while invalidating the relationship learned by a fixed reader. The five-fold reference comparison uses the aligned bank as its common baseline, so representation increments are measured after this data-contract correction. On its separate 542-task/377-gene evaluation, old-guide B0 has U20 0.6899 and cell-weighted B0 has 0.8491, a paired increase of 0.1592 [0.0418, 0.2144]. These stratum-macro results differ from Table 7 because task eligibility, folds and aggregation differ; the two estimates are retained under their own contracts.

### 5.9 Reference representation improves some comparisons without establishing a set-interaction contribution

The complete four-arm experiment uses five outer gene folds, three registered neural training seeds, and the same eligible public histories. Source contains 1,808 biological tasks/575 genes evaluated under each of two upstream predictors; McFaline contains 542 tasks/377 genes. All arms cover every planned task. Within each context-by-outer-fold stratum, metrics are calculated separately per fitting seed and then averaged; strata receive equal weight. Thus Table 8 is not a best-seed comparison or a ranking of seed-averaged scores. Source has 20 strata per predictor and McFaline has 15. The paired bootstrap synchronizes Source gene multiplicities across predictors, folds and seeds; McFaline is sampled independently.

| Table 8. Fixed historical-reader Utility@20 | Source GAT | Source Exphormer | McFaline DecoderOnly |
|---|---:|---:|---:|
| B0 support mean | 0.5446 | 0.5589 | 0.8491 |
| B1 per-history HGB | 0.5906 | 0.5939 | 0.8270 |
| B2 pointwise network | 0.6308 | 0.6357 | 0.8699 |
| B3 set-conditioned network | 0.6328 | 0.6291 | 0.8648 |
| Prediction magnitude | 0.7524 | 0.7285 | -0.0086 |
| Negative history support | 0.1137 | 0.1069 | 0.8375 |
| Nearest-control reference | 0.4590 | 0.4849 | 0.7958 |
| Same-context support reference | 0.5446 | 0.5589 | 0.8722 |
| Same-condition support reference | 0.5446 | 0.5589 | 0.7064 |

The matching rules are fixed before their evaluation: nearest control selects the eligible record with minimum control-state RMSE; same-context and same-condition rules use support means within that match when available and otherwise use the full eligible set. They require neither biological outcome selection nor risk training. Source eligibility already restricts history to other contexts, so the same-context fallback is exactly B0. McFaline permits same-context histories in other conditions, creating a stronger distinct comparator.

| Table 9. Paired reference increments, fixed historical reader | Difference in U20 | Paired 95% interval | Nonnegative strata |
|---|---:|---|---:|
| Source GAT: B2 minus B1 | 0.0402 | [-0.0044, 0.0867] | 65.0% |
| Source Exphormer: B2 minus B1 | 0.0418 | [-0.0022, 0.0892] | 75.0% |
| McFaline: B2 minus B1 | 0.0429 | [-0.0132, 0.0523] | 80.0% |
| Source GAT: B3 minus B2 | 0.0020 | [-0.0149, 0.0142] | 55.0% |
| Source Exphormer: B3 minus B2 | -0.0067 | [-0.0154, 0.0153] | 40.0% |
| McFaline: B3 minus B2 | -0.0051 | [-0.0099, 0.0090] | 66.7% |
| McFaline: B2 minus same-context support | -0.0023 | [-0.0794, 0.0362] | 60.0% |
| McFaline: B3 minus same-context support | -0.0074 | [-0.0813, 0.0364] | 53.3% |

B2 improves on B1 by about 0.04 in both Source views and satisfies those pairwise practical gates. However, both neural historical readers remain below magnitude in Source, and their McFaline intervals do not establish the required increment over B1 or the stronger same-context reference. B3 supplies no consistent added value over B2. Its practical structure gate fails in every view. This result supports pointwise representation gains relative to the existing historical scorer in Source; it does not support a new set-interaction claim.

The required adoption gate combines actual U20 gain of at least 0.005, paired lower limit of at least -0.005, at least 60% nonnegative strata and at least 80% valid strata. It additionally limits macro AURC and error at 10/20/50% coverage deterioration to 5%, and high-risk miss-rate increase to 0.02. A global replacement must pass against B0, B1, magnitude, support and the three fixed matching rules in all three predictor/domain views; B3 additionally requires its B3-versus-B2 structure gate. Neither neural candidate passes this full fixed-reader requirement, so B1 remains the registered reference-only baseline. This decision is conditional on R_H and does not select the final supervised builder-reader combination.

The biological results explain why reference reconstruction alone is insufficient. On Source, macro reconstruction RMSE decreases from 0.062596 for B0 to 0.061179 for B1, 0.059746 for B2 and 0.059739 for B3. On McFaline, corresponding values are 0.029723, 0.029755, 0.029640 and 0.029639. The virtually identical B2/B3 reconstruction values accompany inconsistent risk differences. Moreover, McFaline same-context support has worse reconstruction RMSE, 0.032065, while yielding the strongest fixed-reader U20 point estimate. Alignment, reconstruction and reliability ranking therefore remain separate empirical outcomes.

### 5.10 Fully nested risk readers and source-error correspondence

The jointly evaluated readers are reported separately from Table 8. The result table binds prediction-only, Public, and UniversalP information definitions to each biological builder, with inner-fold public preprocessing, context-specific teacher errors and training-only CDFs. The source-label null analysis additionally compares genuine source errors, gene-cluster permutations and a biology-only error proxy. These comparisons determine the increment from source supervision under a common reader and input budget.

[[REGISTERED_READER_RESULTS]]

## 6. Discussion

SafeConf makes the information available to reliability estimation explicit. The completed evidence suggests that public references and source errors should be treated as distinct resources. In the source architecture pair, error supervision contains useful ranking information beyond a biological-magnitude proxy and historical distance. Across studies, a public reference can remain useful while the source-trained risk mapping loses its advantage. When target feedback arrives, public features can remain valuable even when an additional shared score does not improve the target learner.

This pattern argues for a modular implementation. A public builder should produce an interpretable biological reference together with support and conflict information. The risk learner should be evaluated against an unsupervised rule that already exploits that reference. Target learning should be allowed to use public evidence directly rather than being forced through a chain of residual corrections. A deployment can then select a validated component combination for its information condition without treating each stage as a guaranteed positive correction.

The completed builder comparison answers a specific representation question. Pointwise neural weighting improves the Source historical reader relative to its existing scorer, but adding learned set context does not consistently improve on pointwise weighting. These observations do not identify a universally preferred builder. A convex combination cannot reconstruct a response outside the representable history set, and a highly accurate biological mean is not necessarily the best statistic for ranking prediction errors. The constrained reconstruction diagnostic and the fixed risk readers make these failure modes distinguishable.

Several limits determine the scope of the present evidence. The mature source pool contains two architectures from one family. Public history predominantly covers previously observed perturbations in other contexts. Evaluation error contains measurement variation, and public support is unusually predictive in McFaline. Orion supplies an independent frozen result but has low history coverage. The neural reference and registered nested-reader results retain development/seen scope; they do not inherit the independent evaluation status of an older component version. This restricts generalization claims even when a within-cohort comparison is positive.

## 7. Conclusion

SafeConf treats public biological responses, source-predictor errors, and target feedback as three different resources for task-level reliability. A simple historical-distance identity supplies a strong comparator, while shared error learning and target adaptation are evaluated at explicit information budgets. Completed experiments show useful source-supervision signal within a model family and substantial public-evidence value in an external predictor setting, together with limits on cross-study transfer and additive feedback benefits. The matched four-arm comparison further shows that a data-alignment gain can exceed a representation increment, and that learned set context need not improve reliability beyond pointwise weighting or simple matched aggregation. The framework supports explicit component selection and rejects updates that fail their risk criteria.

## Data and Code Availability

Code and result artifacts are maintained in the SafeConf repository: https://github.com/1298020005/SafeConf, branch `exp/e220-reviewer-closure-20260921`. The accompanying `evidence/RESULT_MANIFEST.json` binds every table reproduced here to its original result file and SHA-256 hash. Large prediction arrays and model checkpoints are stored in the documented runtime locations and are not silently bundled into the manuscript source. Public dataset terms and upstream software licenses apply. The source snapshot and per-file hashes are supplied with the submission package. The final public release identifier and author declarations require author completion.

## Acknowledgments and Declarations

Author contributions: [AUTHOR-APPROVED CONTRIBUTIONS]. Funding: [FUNDER / GRANT OR AUTHOR-CONFIRMED NONE]. Competing interests: [AUTHOR-CONFIRMED DECLARATION]. Ethics and data-use declarations: [AUTHOR / INSTITUTION-CONFIRMED STATEMENT]. These fields require author completion before submission.

## Appendix A. Evidence files for the reported comparisons

The companion supplement provides full information ledgers, all fitting seeds, fixed versus auxiliary readers, the constrained reconstruction diagnostic, exposure audits and source-label null comparisons. `evidence/RESULT_MANIFEST.json` maps each copied result table to its original result path and SHA-256. The old 543-task McFaline matrix, the new 542-task fold-based comparison, native512 Frangieh stress experiment and frozen Orion3285 evaluation retain their distinct cohorts and roles. No unexecuted experiment supplies a reported numerical result.
