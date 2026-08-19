---
bibliography: references.bib
csl: styles/biomed-central.csl
link-citations: true
reference-section-title: References
lang: en-US
---

# Supplementary Information {.unnumbered}

**SafeConf: registered-family error certificates for auditable single-cell perturbation prediction**

**Yifan Yang and colleagues**

# Supplementary Methods

## S1. Scope and terminology

A **task** is one prediction vector paired with one experimental effect vector on the frozen 512-gene panel. Depending on the source study, tasks correspond to donor–state combinations, technical groups, or individual guides. A **target cluster** contains every task assigned to one held-out perturbation identity within the relevant split. Calibration and coverage calculations keep the complete cluster together.

A **registered family** is a list of model checkpoints, seeds, output genes, normalization rules, and aggregation rules fixed before evaluation truth is accessed. The main family contains five scGPT fits and five GEARS fits with seeds 3407–3411. A family certificate describes this finite collection; it is not a claim about every possible model trained with the same architectures.

For a task with predictions \(p_1,\ldots,p_m\) and experimental truth \(y\), \(R_F\) is the root mean of the members' squared RMSE values and \(W_F\) is the largest member RMSE. The **family lower certificate** \(D_F\) measures RMS dispersion around the family centroid. The **worst-member lower certificate** \(\Delta_F/2\) is half the maximum pairwise prediction distance. The upper certificates combine a reference-centroid conformal event with an observable centroid-shift penalty.

## S2. Frozen study designs

| Study | Train targets | Validation targets | Calibration targets | Evaluation targets | Evaluation tasks | Cluster content |
|---|---:|---:|---:|---:|---:|---|
| Primary CD4 | study-specific prospective panel | study-specific prospective panel | 160 total, 40 in each donor rotation | 640 | 1920 | Rest, 8-h stimulation, and 48-h stimulation |
| Sunshine | 54 | 10 | 30 | 50 | 400 | eight `gem_group` technical groups |
| XuCao | 65 | 32 | 29 | 27 | 73 | all eligible guides for one target |
| GSE225807 | 28 | 9 | 19 | 20 | 40 | two guides for one RBP target |

The Primary CD4 count of 640 evaluation clusters comprises four donor rotations with 160 evaluation targets each. Donor-specific rotations are not presented as independent studies. In Sunshine, `gem_group` labels are technical rather than biological contexts. In XuCao, cell-cycle phase is observed after perturbation and therefore was not used as a pre-perturbation context variable. GSE225807 was the only study executed as a new, fully preregistered registered-family confirmation.

## S3. Derivation of the family RMS identity

Let \(\bar p=m^{-1}\sum_i p_i\). For one family member,

\[
\lVert p_i-y\rVert^2
=\lVert(p_i-\bar p)+(\bar p-y)\rVert^2.
\]

Expanding the inner product gives

\[
\lVert p_i-\bar p\rVert^2+\lVert\bar p-y\rVert^2
+2\langle p_i-\bar p,\bar p-y\rangle.
\]

Averaging over members removes the cross term because
\(m^{-1}\sum_i(p_i-\bar p)=0\). Hence

\[
\frac{1}{m}\sum_i\lVert p_i-y\rVert^2
=\frac{1}{m}\sum_i\lVert p_i-\bar p\rVert^2
+\lVert\bar p-y\rVert^2.
\]

With the definitions in the manuscript,

\[
R_F^2=D_F^2+\lVert\bar p-y\rVert^2.
\]

The gene-normalized Euclidean norm differs from the usual Euclidean norm by the constant \(G^{-1/2}\), so the same inner-product identity holds. The deterministic inequality \(D_F\leq R_F\) follows because the remaining term is non-negative.

This is the classical ambiguity decomposition for squared loss [@krogh1994ambiguity]. SafeConf uses the identity to define an auditable lower certificate for a frozen heterogeneous prediction family; it does not claim the identity as a new theorem.

## S4. Worst-member lower certificate

Choose family members \(a\) and \(b\) attaining the diameter
\(\Delta_F=\lVert p_a-p_b\rVert\). The triangle inequality gives

\[
\Delta_F
\leq \lVert p_a-y\rVert+\lVert p_b-y\rVert
\leq 2\max_i\lVert p_i-y\rVert.
\]

Therefore \(\Delta_F/2\leq W_F\). As with the family RMS lower bound, no calibration sample or target truth is needed to calculate the left side.

## S5. Transfer of the reference upper event

Let \(c\) be the frozen reference centroid and let the reference calibration event be
\(\lVert c-y\rVert\leq U\). The registered-family centroid \(\bar p\) may differ from \(c\). Because

\[
\lVert\bar p-y\rVert
\leq \lVert c-y\rVert+\lVert\bar p-c\rVert,
\]

the reference event implies
\(\lVert\bar p-y\rVert\leq U+s\), where
\(s=\lVert\bar p-c\rVert\). Combining this with the identity in S3 yields

\[
R_F\leq \sqrt{(U+s)^2+D_F^2}.
\]

For each family member,

\[
\lVert p_i-y\rVert
\leq\lVert p_i-\bar p\rVert+\lVert\bar p-y\rVert
\leq r_F+U+s,
\]

which gives the worst-member upper certificate. The shift \(s\) is essential: omitting it would silently apply a guarantee calibrated for one centroid to a different prediction.

## S6. Target-cluster split conformal rule

For target \(j\), all associated tasks \(t\in\mathcal T_j\) contribute one score

\[
S_j=\max_{t\in\mathcal T_j}\{e_t-b(x_t)\},
\]

where \(e_t\) is reference-centroid RMSE and \(b(x_t)\) is a frozen non-negative base. With \(n\) calibration targets and nominal miscoverage \(\alpha\), the correction is the ordered score with rank

\[
k=\operatorname{ceil}\big((n+1)(1-\alpha)\big).
\]

The target is covered only when every task in the target is below its task-level upper bound. This prevents two guides or repeated contexts of the same target from being counted as independent calibration examples. Under exchangeability of target clusters, the usual split-conformal marginal coverage argument applies [@lei2018conformal]. The procedure does not establish conditional coverage for every molecular class or transport under arbitrary distribution shift.

## S7. Reference bases and efficiency comparison

| Study | Frozen reference base | Selection status |
|---|---|---|
| Primary CD4 | donor-specific predicted-magnitude base | frozen before the registered-family reformulation |
| Sunshine | predicted magnitude plus model-disagreement terms | frozen before calibration |
| XuCao | constant, magnitude, magnitude-plus-lower, and ExtraTrees compared under the same cluster correction | adaptive method failed the registered efficiency gate |
| GSE225807 | constant centroid bound only | fixed in the preregistration |

In XuCao, the ExtraTrees model used 18 prediction-derived features and validation-target errors. It covered all 27 evaluation targets but produced a mean upper bound of 0.2315 RMSE, compared with 0.2168 for the constant comparator. The mean difference was +0.0147, with a target-cluster bootstrap 95% interval of +0.0107 to +0.0188. The adaptive method was therefore rejected before the GSE225807 design was finalized.

## S8. Family-composition analyses

The registered ten-member family was compared with two architecture-level means: the mean of five scGPT predictions and the mean of five GEARS predictions. The seed-level family increased median lower-bound tightness by 0.0826 in Primary CD4, 0.0857 in Sunshine, and 0.1011 in XuCao. Every paired task favored the seed-level family.

A no-change prediction was added only as a post-evaluation negative-control stress test. It did not improve tightness in any XuCao task and changed the family centroid in every study. The upper-certificate transfer consequently paid a nonzero shift penalty. This analysis illustrates why member identities and seeds must be fixed independently of evaluation outcomes.

The family centroid calculated from all ten seed-level vectors agreed with the mean of the two five-seed architecture means to within 1.89 × 10⁻⁹ RMSE in the retrospective studies. This check detects inconsistent weighting or member loss.

## S9. Fully preregistered GSE225807 confirmation

### S9.1 Eligibility and split

Eligible RBP targets had exactly two assigned guides, at least five cells per guide, at least 20 cells across the two guides, and complete coverage on the frozen expression and vocabulary axes. Salted SHA-256 target identities were assigned within cell-count strata. The split contained 28 training targets, 9 validation targets, 19 calibration targets, and 20 evaluation targets.

Negative-guide controls contained 91 cells. The logical access record contained 1,348 supervised-training rows, 456 validation rows, 926 calibration rows, and 1,051 evaluation rows. Calibration and evaluation expression rows were absent from the pretruth scoring interface. Evaluation rows were opened only after the conformal threshold and all gates had been committed to both GitHub and Gitee.

### S9.2 Registered gate

The 90% constant centroid threshold was the 18th ordered maximum centroid RMSE among 19 calibration targets:

\[
U=0.18978567801751686.
\]

The registered success rule required simultaneous family upper coverage for at least 17 of 20 evaluation targets, zero deterministic lower violations, ambiguity-identity residual no greater than 10⁻¹⁰, and no access violation. The observed family result was 16/20; the registered overall status is therefore `FAIL`.

Four targets failed simultaneous family coverage: HNRNPC, DDX6, SLTM, and DDX42. DDX42 also failed the worst-member upper bound. All 40 guide tasks satisfied both deterministic lower bounds. The maximum identity residual was 2.26 × 10⁻¹⁷.

### S9.3 Finite-calibration reference calculation

For continuous exchangeable scores, selecting the 18th order statistic among 19 calibration scores gives a conditional future coverage probability distributed as \(\mathrm{Beta}(18,2)\). Mixing a binomial count of 20 evaluation targets over this distribution gives

\[
P(K\leq16)=
\sum_{k=0}^{16}\binom{20}{k}
\frac{B(k+18,20-k+2)}{B(18,2)}
=0.186813.
\]

This calculation explains the sampling variability induced by a small calibration set. It was computed after the failed registered gate and does not change the gate, the threshold, or the reported 16/20 outcome.

## S10. Cross-study synthesis

| Study | Tasks | Targets | Family lower violations | Family target upper coverage | Median family lower tightness |
|---|---:|---:|---:|---:|---:|
| Primary CD4 | 1920 | 640 | 0 | 579/640 (90.47%) | 0.240 |
| Sunshine | 400 | 50 | 0 | 44/50 (88.00%) | 0.170 |
| XuCao | 73 | 27 | 0 | 27/27 (100%) | 0.255 |
| GSE225807 | 40 | 20 | 0 | 16/20 (80.00%) | 0.513 |
| Combined, descriptive | 2433 | 737 | 0 | 666/737 (90.37%) | not pooled |

The combined exact target interval is 88.00%–92.40%. It is reported descriptively because the synthesis combines different studies, task structures, bases, and analysis timing. It is not a new conformal calibration and cannot turn the GSE225807 gate into a success.

## S11. Reproducibility and access chronology

The GSE225807 release was preserved in the following order:

| Stage | Commit | Information fixed or opened |
|---|---|---|
| F1 metadata freeze | `b4701ac` | eligibility, salted identity split, task contract |
| F1 runtime hardening | `fc0de74` | implementation checks before truth release |
| F2 pretruth release | `2f4f148` | model family and prediction-derived geometry |
| F3 calibration lock | `420e536` | calibration truth and threshold |
| F4 final evaluation | `593f663` | one-time evaluation truth and registered outcome |
| post-F4 synthesis | `744b5ef` | retrospective four-study aggregation |

The minimal-release validator recomputes task and target counts, deterministic inequalities, upper-coverage flags, identity residuals, the GSE225807 gate, and the beta-binomial probability from committed release tables. It ran 12,033 checks with zero failures under Python 3.9.25 and Python 3.12.4. A separate pre-submission integrity audit contained 18 checks, all passing.

## S12. Model specificity and pretruth provenance

The following distinctions were fixed to prevent a family-level quantity from being interpreted as confidence in one model:

| Question | Operational answer | Uses evaluation truth? | Reporting boundary |
|---|---|---|---|
| What does \(D_F\) certify? | RMS error across the complete registered family | no | does not certify any named member separately |
| What does \(\Delta_F/2\) certify? | error of the worst member in the registered family | no | does not identify which member is worst |
| What is predicted magnitude? | RMSE of a predicted effect vector relative to the zero-effect vector | no | it is not the measured perturbation effect |
| Does a small scGPT–GEARS distance imply safety? | no | no | both architectures may share the same error |
| Can a cross-dataset upper bound be reported without target-domain calibration? | no | not applicable | only the deterministic lower certificate transfers algebraically |

The five seed-level predictions were averaged within architecture before this diagnostic. In Primary CD4, scGPT and GEARS task errors had Spearman \(\rho=0.9747\), and their highest-error quintiles had Jaccard overlap 0.8916. In Sunshine, the corresponding values were 0.9924 and 0.9512. Their distance was associated with scGPT and GEARS RMSE at \(\rho=-0.2112\) and \(-0.1739\) in Primary CD4, and at 0.0586 and 0.0594 in Sunshine. Thus, error difficulty was largely shared, while the between-architecture distance was not a monotone single-model confidence score.

These correlations were calculated after truth release and serve only as interpretation checks. The deployment-time lower certificates use prediction vectors alone. The target expression used to calculate RMSE never enters the prediction-distance, predicted-magnitude, context-similarity, or training-support features.

## S13. Difficulty ladder and direct-transfer stress test

The difficulty audit reused frozen prediction records; it did not select a predictor after viewing the new certificate result. Four multi-context matrices were included: Frangieh genetic perturbations, Lara ex vivo genetic perturbations, Santinha genetic perturbations, and Cui cytokine stimulation. For each matrix, two inductive reference predictors were evaluated under:

1. a random missing context–perturbation pair;
2. a complete unseen context row;
3. a complete unseen perturbation column;
4. a cell whose context and perturbation were both unseen.

Only 25%, 50%, 75%, or 100% of the permitted training submatrix was made available. Across the four fractions, each setting contributed 276, 879, 690, and 204 task instances, respectively, for a total of 8,196. Every task satisfied both deterministic inequalities. At full fraction, the perturbation-cluster bootstrap summary was:

| Held-out structure | Macro median tightness | 95% bootstrap interval |
|---|---:|---:|
| Random missing pair | 0.328 | 0.315–0.342 |
| Unseen context | 0.260 | 0.252–0.269 |
| Unseen perturbation | 0.175 | 0.163–0.183 |
| Context and perturbation both unseen | 0.148 | 0.133–0.158 |

Bootstrap resampling kept every context associated with one perturbation identity together. Dataset medians were then averaged, preventing large matrices from dominating the summary. Repeated training fractions and folds were not counted as independent experiments.

The direct-transfer component added 553 sciPlex3-to-OpenProblems tasks and 28 sciPlex3-to-sciPlex4 tasks. Family-RMS and worst-member lower violations were both zero. Median family-RMS tightness was 0.703 and 0.641. The first ratio is high because the two reference predictors diverged while neither beat the zero-effect baseline, not because transfer was accurate. Target-domain calibration was absent, so no reference-centroid upper bound or cross-dataset coverage claim was constructed.

The full stress audit therefore contains 8,777 task instances. It extends the deterministic validity check beyond random missing pairs, while leaving the ten-member scGPT–GEARS four-study analysis as the main certificate evidence.

## S14. Adaptive upper baselines and native PRESCRIBE comparator

### S14.1 Nested target-level upper-baseline development

Primary CD4 and Sunshine were each resplit 50 times by complete perturbation identity into model-fitting, conformal-calibration, and evaluation partitions. Every candidate received the same target-level correction. The candidates were constant split conformal, predicted magnitude, magnitude plus the two-model lower bound, seed spread, Ridge, ExtraTrees, random forest, and 0.80-quantile gradient boosting.

| Study | Method | Mean target coverage | Mean upper RMSE | Constant mean | Relative reduction | Paired repeat 2.5%–97.5% |
|---|---|---:|---:|---:|---:|---:|
| Primary CD4 | ExtraTrees | 0.894 | 0.2062 | 0.2119 | 2.72% | −1.52%–7.39% |
| Sunshine | ExtraTrees | 0.905 | 0.5389 | 0.5466 | 1.41% | −6.60%–7.72% |

The resplits overlap and quantify development stability rather than independent replication. The selected ExtraTrees configuration was then frozen for XuCao, where it produced a wider upper bound than the constant method by 0.0147 RMSE (target-cluster bootstrap interval 0.0107–0.0188). This failed confirmation motivated the constant-only GSE225807 preregistration.

### S14.2 PRESCRIBE at its own endpoint

PRESCRIBE was evaluated on 48 tasks from two Norman panels using its paper's predictor-specific evidential confidence and prediction–truth correlation endpoint [@cheng2025prescribe]. The two-panel macro association of combined confidence with Pearson effect accuracy was \(\rho=0.3170\) (bootstrap interval 0.0324–0.5669). Predicted magnitude reached \(\rho=0.3104\). Their paired difference was 0.0065 (−0.0317 to 0.0534), while the confidence score and predicted magnitude were almost rank-equivalent (\(\rho=0.9952\), 0.9746–0.9996).

This is a native-endpoint contextual comparator. It does not convert PRESCRIBE confidence into an upper or lower bound, and it does not compare the same estimand as SafeConf. The result is retained because it tests whether a named uncertainty method adds ordering information beyond a simple pretruth baseline in the setting for which it was designed.

# Supplementary Data Files

The following files are included in the submission package:

- `tables/Table_S1_family_comparisons.csv`: paired registered-family composition comparisons;
- `tables/Table_S2_gse225807_targets.csv`: all 20 GSE225807 target-level outcomes;
- `tables/Table_S3_gse225807_tasks.csv`: all 40 GSE225807 guide-level certificates;
- `tables/Table_S4_difficulty_setting_summary.csv`: dataset-, fraction-, and setting-level difficulty summaries;
- `tables/Table_S5_difficulty_macro_bootstrap.csv`: perturbation-cluster bootstrap intervals for the difficulty ladder;
- `tables/Table_S6_cross_dataset_summary.csv`: direct-transfer certificate summaries and claim boundaries;
- `tables/Table_S7_difficulty_task_certificates.csv`: all 8,196 Cartesian stress-test tasks;
- `tables/Table_S8_cross_dataset_task_certificates.csv`: all 581 direct-transfer tasks;
- `tables/Table_S9_model_error_concordance.csv`: scGPT–GEARS shared-error diagnostics;
- `tables/Table_S10_score_model_specificity.csv`: score associations with model-specific and family error objects;
- `tables/Table_S11_nested_upper_baselines.csv`: 50-repeat target-level upper-baseline summary;
- `tables/Table_S12_prescribe_native_endpoint.csv`: PRESCRIBE native-endpoint associations;
- `tables/Table_S13_prescribe_incremental.csv`: paired increments over predicted magnitude;
- `tables/Table_S14_prescribe_redundancy.csv`: score-to-magnitude rank associations;
- `tables/Table_2_certificate_results.csv`: study-level certificate results used in the manuscript;
- `figures/Figure_1_method_and_protocol.*` through `Figure_5_reproducibility_chain.*`: 600-dpi PNG, vector PDF, and editable SVG exports.

The complete main-certificate release remains in `docs/实验结果/E181_*` through `E186_*`; secondary audits are stored in `E145_*`, `E178_*`, `E179_*`, and `E187_*`. The source-data validator is invoked from the repository root with:

```bash
python tools/scripts/validate_current_certificate_release.py
```
