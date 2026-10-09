# Public perturbation evidence enables cold-start risk auditing for single-cell perturbation prediction

## Abstract

Single-cell perturbation predictors are usually evaluated after the experiment has been performed, while practical use requires deciding which predictions deserve limited experimental review. Existing post-hoc reliability methods learn from the target predictor's own out-of-fold errors. This creates a cold-start problem: a new predictor has no target-screen error history when its first predictions need to be audited. We introduce SafeConf, a risk-auditing framework that uses protocol-compatible public perturbation experiments as an external evidence layer and adds predictor-specific feedback only when it is available. The public layer produces a frozen risk ranking without fitting target-screen error labels; source-model errors and target feedback are tracked as separate information budgets and are not forced into the default system.

On a fixed McFaline benchmark, public experimental evidence identified 22 and 24 of the 43 most erroneous predictions for two different predictors, compared with 4 and 5 using prediction magnitude alone. The remaining mean error after the same review budget was reduced by 7.07% and 6.84%. In a retrospective Frangieh cross-family stress test, public evidence improved risk utility over magnitude in both GEARS-to-scGPT and scGPT-to-GEARS directions, with paired U20 increments of 0.460 and 0.309 respectively. A frozen KOLF study supplied an independent evaluation contract with 600 upstream-training, 300 development, and 300 confirmation genes. One predictor passed the pre-registered competence gate; the public rule covered 228 of 300 evaluation tasks and used a deterministic magnitude fallback for the remaining tasks. The external point estimate favored public evidence over magnitude, while the gene-cluster interval remained wide. These results support public perturbation history as a practical cold-start risk layer and define the conditions under which predictor-specific adaptation should be used.

## 1. Introduction

Perturbation prediction is increasingly used to prioritize biological hypotheses before experimental validation. A prediction error is therefore not only a model-quality statistic; it is also a resource-allocation problem. When only a fraction of predictions can be checked experimentally, a useful reliability layer should place the most likely failures near the top of the review queue.

Post-hoc reliability estimators such as PertEMA learn a mapping from predictor outputs and historical errors to a later risk score [@officialbishal2025pertema]. This is appropriate after a screen has accumulated enough out-of-fold errors, but it leaves a practical gap for a new predictor or a new screen. The first predictions of a new screen must be reviewed before the target model has produced the error labels required to fit a target-specific estimator.

PRESCRIBE estimates epistemic and aleatoric uncertainty from the predictor and its training evidence [@cheng2025prescribe]. SafeConf addresses a different information regime: the risk layer can use protocol-compatible measured responses from other experiments before target-screen error labels exist. Recent evaluation work also shows that reusing control cells can inflate perturbation metrics [@nicol2026spurious]; our primary comparisons therefore keep the response contract, control construction, task grouping, and review budget fixed rather than treating a raw correlation score as a reliability certificate.

SafeConf addresses this gap by treating public perturbation experiments as evidence available before target-screen errors. The public evidence layer is model-independent: it summarizes protocol-compatible historical responses, their support, and their distance from the current prediction. When target errors become available, a target learner can use the same evidence, but its gain is evaluated under the same task and feedback budget rather than assumed. Errors from another predictor are tracked as a separate source of supervision and are retained only when they add information beyond strong public baselines.

This use of public responses is distinct from response-transfer methods such as PerturbMap, which aim to reconstruct a response in a recipient context [@cui2026perturbmap]. SafeConf uses public responses as a post-hoc audit signal and evaluates whether that signal improves limited-budget error discovery.

The paper asks three questions. First, can public experiments start risk auditing without fitting target-model error labels? Second, how much does predictor-specific feedback add after this public audit has started? Third, does the public signal survive a change of predictor family or study context? The experiments are designed so that each question has a corresponding information ledger, strong simple baselines, and a separate failure boundary.

## 2. SafeConf framework

For a query perturbation, let (p) be the frozen prediction and let (h_1,ldots,h_m) be eligible public responses. The public layer forms a response-space reference from the historical effects and computes a prediction-to-history distance. Support, historical energy, conflict, and missing-history status remain separate fields. A task without an eligible history does not receive a fabricated zero-effect reference; it receives the registered magnitude fallback.

All channels are converted with training-side empirical CDFs so that tasks with and without history can enter one risk ranking. The CDF transforms scores only; it does not estimate channel weights from target errors. The adopted default is the public rule. Source and target learners are evaluated as supervised candidates with their own error-label budgets.

The system has a single scoring interface but two kinds of scoring logic. Rule-based scores preserve the zero target-error supervision property. Supervised scores use fold-out predictions and registered source or target errors. A final ranking is produced only after its configuration, score hashes, missing-history behavior, and feedback budget are frozen.

### 2.1 Information contracts and splits

We separated three information ledgers. Public biological evidence records the study, context, perturbation identity, control source, response contract, independent experimental unit, support, and conflict. Source supervision records the upstream predictor version and its eligible held-out errors. Target supervision records the current predictor's feedback records and every use made of them for fitting, calibration, selection, or release. A record used to evaluate a task was excluded from the corresponding fitting or selection path. Public histories from the target study were excluded from the target-study public memory.

The primary McFaline analysis used 212 evaluation tasks grouped into 152 perturbation-gene clusters. The feedback pool contained 331 records from 228 clusters, and the separate 542-task development pool was counted as preparation cost. DecoderOnly and SAMS-VAE predictions were frozen before risk scoring. Feedback budgets were 10%, 25%, 50%, 75%, and 100%, with ten pre-fixed feedback orders and three learner seeds. These repeated runs measure algorithmic stability; they are not independent biological samples.

The KOLF2.1J evaluation used a fixed 600-gene predictor-training partition, 300-gene development/feedback partition, and 300-gene evaluation partition. A competence gate was applied before evaluation responses were opened. Risk scores and all configuration hashes were frozen before the evaluation truth was read. This ordering distinguishes predictor preparation, risk-layer development, and final scoring.

### 2.2 Public reference and score construction

For a task with eligible historical responses (h_i), we constructed a weighted response reference and computed the prediction-to-history distance

\[
D(p,h)=\sqrt{\sum_i w_i\,\mathrm{mean}_g\{p_g-h_{ig}\}^2}.
\]

The weights and aggregation object were fixed in the data contract. Historical support, response energy, conflict, and missingness were retained as separate fields. The adopted PublicRule uses the registered distance and its history-dispersion term; it is not a learned error regressor. A single history has zero empirical dispersion by definition, which is recorded rather than interpreted as proof of experimental reliability.

To combine tasks with and without history in one review queue, PublicRule and magnitude scores were transformed by empirical CDFs fitted only on the relevant training side. The CDF is a scale transformation, not a learned risk weight. A task with no eligible history receives the magnitude channel and an explicit `NO_HISTORY_AMPLITUDE_FALLBACK` status. It never receives a fabricated zero-effect history. For KOLF, the registered prediction axis had 1,400 coordinates, while the Replogle public resource supplied 1,366 common coordinates. The Public comparison used the explicit common axis with identical normalization and no zero filling.

### 2.3 Supervised extensions and PertEMA comparison

Source and Target learners were evaluated as separate candidates rather than forced into the default. The Target candidate used the fixed PertEMA XGBoost recipe with the available prediction, control, similarity, and Public additions. It is reported as a PertEMA-aligned/native-feature adaptation because the complete official calibration and conformal interval workflow was not reproduced. Every supervised training score was generated out of fold; target error labels were never used by the PublicRule arm.

The primary comparison is the same task, predictor, error target, feedback budget, and cluster weighting across all arms. We report prediction magnitude, support and history-energy controls, PublicRule, Target-only, Native plus Public, and the fixed PertEMA-aligned candidate. Source errors are listed separately and are not silently merged with Target errors.

### 2.4 Evaluation and uncertainty

The primary practical endpoint is utility at a 20% review budget. We also report 5%, 10%, and 30% review budgets, AURC, Spearman association, high-risk miss rate, selected-task error, and remaining mean error. The true highest-error set is defined from evaluation truth only; it never enters a score or model input. EffectTop200-RMSE is a secondary evaluation endpoint whose 200 genes are selected from truth only at scoring time.

All confidence intervals use 5,000 paired bootstrap draws with perturbation-gene cluster as the resampling unit. The same draws are applied to every method in a comparison. Feedback-order and learner-seed variation are reported separately from biological uncertainty. Point estimates and bootstrap means are not conflated. Results from McFaline and Frangieh are marked SEEN; KOLF is a frozen independent evaluation, not a claim that superiority was established in every external setting.

## 3. Results

### 3.1 Public evidence starts the audit

We first evaluated two frozen predictors on 212 McFaline tasks spanning 152 perturbation gene clusters. A review budget of 20% corresponds to 43 tasks. For DecoderOnly, PublicRule selected 22 of the 43 tasks in the true highest-error set, whereas prediction magnitude selected 4. For SAMS-VAE, the corresponding counts were 24 and 5. After removing the reviewed predictions, the remaining mean error was 7.07% lower than the magnitude baseline for DecoderOnly and 6.84% lower for SAMS-VAE.

The public result is not explained by a single score interpretation. Support-only, history-energy, magnitude, direct distance, and content-matched permutation controls were reported separately. In the strict 20-permutation content analysis, the nominal randomization result did not establish a gene-specific content increment beyond the registered controls. This negative result is retained because it identifies a measurement and representation boundary rather than allowing a post-hoc mechanism claim.

### 3.2 Feedback improves some global rankings but does not replace the public start

We compared prediction-only, Native control, Native plus Public, and feedback learners at 10%, 25%, 50%, 75%, and 100% feedback budgets. Ten feedback orders and three learner seeds were treated as algorithmic stability variation; the biological bootstrap unit remained the perturbation gene cluster.

On the pooled McFaline ranking, Native plus Public improved over Public at the registered feedback budgets in the global diagnostic. At full feedback, the global U20 increments were 0.198 for DecoderOnly and 0.141 for SAMS-VAE, with paired 95% intervals above zero. The registered context-macro primary endpoint was reported separately and did not justify replacing the no-error-label public rule. Target-only did not reach the public rule's non-inferiority level at any tested budget, so the label-equivalent budget remains right-censored rather than being converted into an interpolated saving.

This distinction matters operationally. Public evidence can start the audit immediately. Feedback can improve a particular pooled ranking, but the improvement depends on task aggregation and is not guaranteed to dominate the public rule in every context.

### 3.3 Cross-family stress evidence

The Frangieh native512 retrospective analysis used the same task truth for GEARS and scGPT [@frangieh2021perturbcite; @roohani2023gears; @cui2024scgpt] and tested both directions of error transfer. PublicHGB exceeded magnitude on all-test risk utility by 0.460 (95% CI 0.317–0.704) for GEARS-to-scGPT and 0.309 (0.079–0.440) for scGPT-to-GEARS. Heldout-context summaries showed the same direction.

All six upstream predictor competence checks failed the registered predictor-quality gate. We therefore use Frangieh as a cross-family stress analysis, not as a qualified independent confirmation. The result demonstrates that the public signal can remain informative across predictor families under a fixed response contract; it does not prove that an unqualified upstream predictor is suitable for deployment.

### 3.4 Frozen independent KOLF evaluation

KOLF2.1J was frozen before reading confirmation responses with 600 upstream-training genes, 300 feedback/development genes, 300 confirmation genes, and a source-defined 1,400-gene response panel, following the genome-scale KOLF2.1J perturbation atlas [@nourreddine2026kolf]. Ridge passed the predictor competence gate. The MLP failed the variance gate because its predicted between-task variance was too small; the failure was retained and no threshold was relaxed to force a second predictor into confirmation.

Risk scores were frozen before confirmation values were read. Public history excluded the KOLF study. The Replogle GWPS [@replogle2022perturbseq] contained 1,366 of the 1,400 requested response coordinates; the missing 34 coordinates were recorded explicitly and were not zero-filled. Public comparison was performed on the actual common axis with the same NTC normalization view for prediction and history.

Public history was available for 228 of 300 confirmation tasks. The other 72 tasks used the pre-registered magnitude fallback, and the fallback scores were exactly identical to the magnitude scores. At 20% review, PublicRule had a U20 of 0.184 versus 0.048 for magnitude; the point difference was 0.136 with a 95% gene-cluster interval of approximately -0.048 to 0.366. Both methods selected 17 of the 60 true highest-error tasks. Equal hit counts do not imply equal rankings: U20 also uses the positions and error magnitudes of the selected tasks, while remaining-error and AURC summarize the rest of the queue. We therefore interpret KOLF as a frozen independent evaluation of feasibility and information accounting, with a positive but statistically broad point estimate rather than a confirmed universal superiority claim.

Native plus Public had a positive point increment over Native at full KOLF feedback, but its 95% interval crossed zero. This result prevents us from presenting the current PertEMA-aligned/native-feature adaptation as a complete reproduction of the official conformal workflow or as an externally confirmed replacement for the public rule.

## 4. Discussion

The main contribution of SafeConf is an evidence protocol for risk auditing under cold-start supervision. Public perturbation history can be used before the current predictor has generated a screen-specific error memory. The same protocol also makes clear when supervised extensions add information and when they merely change the learner or the score scale.

Three design choices support this interpretation. First, public evidence, source errors, and target errors are kept in separate information ledgers. Second, all no-history behavior is part of the full ranking rather than being silently excluded from evaluation. Third, predictor competence is checked before an external risk result is treated as a confirmation claim.

The experiments also set limits. Public history is not automatically independent biological truth: one screen is one historical unit, and a large cell count does not create independent screens. Frangieh shows strong cross-family stress evidence but fails the upstream predictor gate. KOLF supplies a qualified predictor and a frozen confirmation contract, yet its public-versus-magnitude interval remains broad. These limits define the conditions under which SafeConf should be used and motivate future larger independent studies.

SafeConf is therefore best understood as a reliability layer and evaluation framework rather than a replacement perturbation predictor. It turns public experimental history into an actionable review ranking, keeps target feedback optional, and reports the cost and boundary of every additional source of supervision.

## 5. Data and code availability

All frozen task tables, score hashes, information ledgers, figures, source data, and reproduction commands are archived in `evidence_freeze_v1`. The default system is PublicRule with an empirical-CDF magnitude fallback for missing history. The repository branch and final commit are recorded in the project entrypoint.

## References

The BibTeX file `REFERENCES.bib` contains the primary GEARS, scGPT, PertEMA, Replogle, Frangieh, PerturbMap, KOLF, and SafeConf entries. A recent perturbation-evaluation study is retained as a discussion citation candidate and will be included only if its endpoint definition is used in the final manuscript.
